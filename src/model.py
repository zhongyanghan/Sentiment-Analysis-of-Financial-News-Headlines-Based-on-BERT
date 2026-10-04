# -*- coding: utf-8 -*-
"""
模型定义与数据预处理模块。

包含:
- ClassifyModel: BERT + Dropout + Linear 的文本分类模型
- DataProcessForSingleSentence: 单句分类任务的数据预处理（分词/截断/填充/id化）
- load_data: 加载 Excel 数据并构建 DataLoader
- evaluate_accuracy: 在测试集上评估并生成 classification_report
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pytorch_pretrained_bert import BertModel, BertTokenizer


class ClassifyModel(nn.Module):
    """BERT 句子分类模型。

    结构: 预训练 BERT -> [CLS] 池化向量 -> Dropout -> 全连接分类层。

    :param pretrained_model_name_or_path: 预训练模型名称或本地路径,
        如 ``bert-base-chinese``
    :param num_labels: 分类类别数
    :param is_lock: 是否冻结 BERT 主体参数(仅训练 pooler 与分类层)
    """

    def __init__(self, pretrained_model_name_or_path: str,
                 num_labels: int = 2, is_lock: bool = False):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        config = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        if is_lock:
            # 冻结 BERT 主体参数, 仅保留 pooler 参与训练
            for name, param in self.bert.named_parameters():
                if not name.startswith("pooler"):
                    param.requires_grad_(False)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled = self.bert(
            input_ids, token_type_ids, attention_mask,
            output_all_encoded_layers=False,
        )
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class DataProcessForSingleSentence:
    """单句分类任务的文本预处理。

    多线程并行完成: 分词 -> 截断/填充 -> id 化 -> 构建 mask 与 segment。

    :param bert_tokenizer: BERT 分词器
    :param max_workers: 线程池大小
    """

    def __init__(self, bert_tokenizer: BertTokenizer, max_workers: int = 10):
        self.bert_tokenizer = bert_tokenizer
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def get_input(self, dataset: pd.DataFrame, max_seq_len: int = 30) -> TensorDataset:
        """将包含 ``comment`` 与 ``sentiment`` 两列的 DataFrame 转为 TensorDataset。

        :param dataset: 第 1 列为文本、第 2 列为标签的 DataFrame
        :param max_seq_len: 序列最大长度(含 [CLS]/[SEP], 不超过 512)
        """
        sentences = dataset.iloc[:, 1].tolist()
        labels = dataset.iloc[:, 2].tolist()
        # 分词
        token_seq = list(self.pool.map(self.bert_tokenizer.tokenize, sentences))
        # 截断与填充
        result = list(self.pool.map(
            self.truncate_and_pad, token_seq, [max_seq_len] * len(token_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]

        return TensorDataset(
            torch.tensor(seqs, dtype=torch.long),
            torch.tensor(seq_masks, dtype=torch.long),
            torch.tensor(seq_segments, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def truncate_and_pad(self, seq, max_seq_len: int):
        """序列截断、添加 [CLS]/[SEP]、id 化并填充至定长。"""
        if len(seq) > (max_seq_len - 2):
            seq = seq[: (max_seq_len - 2)]
        seq = ["[CLS]"] + seq + ["[SEP]"]
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        padding = [0] * (max_seq_len - len(seq))
        seq_mask = [1] * len(seq) + padding
        seq_segment = [0] * len(seq) + padding
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment


def load_data(filepath: str, pretrained_model_name_or_path: str,
              max_seq_len: int, batch_size: int):
    """加载包含 train/test 两个 sheet 的 Excel 并构建 DataLoader。

    :param filepath: Excel 文件路径
    :param pretrained_model_name_or_path: BERT 预训练模型名称或本地路径
    :param max_seq_len: 序列最大长度(不超过 512)
    :param batch_size: 批大小
    :return: (train_iter, test_iter, train_batch_count, test_batch_count)
    """
    io = pd.io.excel.ExcelFile(filepath)
    raw_train_data = pd.read_excel(io, sheet_name="train")
    raw_test_data = pd.read_excel(io, sheet_name="test")
    io.close()

    bert_tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path, do_lower_case=True)
    processor = DataProcessForSingleSentence(bert_tokenizer=bert_tokenizer)
    train_data = processor.get_input(raw_train_data, max_seq_len)
    test_data = processor.get_input(raw_test_data, max_seq_len)

    train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    train_batch_count = math.ceil(len(raw_train_data) / batch_size)
    test_batch_count = math.ceil(len(raw_test_data) / batch_size)
    return train_iter, test_iter, train_batch_count, test_batch_count


def evaluate_accuracy(data_iter, net: nn.Module, device, batch_count: int) -> str:
    """在给定数据集上评估, 返回 sklearn 风格的 classification_report。"""
    prediction_labels, true_labels = [], []
    net.eval()
    with torch.no_grad():
        for batch_data in tqdm(data_iter, desc="eval", total=batch_count):
            batch_data = tuple(t.to(device) for t in batch_data)
            labels = batch_data[-1]
            output = net(*batch_data[:-1])
            predictions = output.softmax(dim=1).argmax(dim=1)
            prediction_labels.append(predictions.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())
    return classification_report(
        np.concatenate(true_labels), np.concatenate(prediction_labels))
