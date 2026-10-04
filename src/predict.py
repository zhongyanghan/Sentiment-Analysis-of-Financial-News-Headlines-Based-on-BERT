# -*- coding: utf-8 -*-
"""
单句预测脚本: 加载微调后的 BERT 模型, 对单条新闻标题进行情感预测。

用法:
    python src/predict.py --model fine_tuned_chinese_bert_cpu.bin \
        --text "海能达与非洲某国公共安全客户签订万美元项目合同"
"""

import argparse
import time

import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer

from model import DataProcessForSingleSentence

LABELS = ["消极的金融新闻", "积极的金融新闻"]


def predict_single_sentence(model, tokenizer, sentence: str,
                            max_seq_len: int, device: str):
    """对单条句子进行情感预测, 返回 (预测类别, 各类别概率)。"""
    data_test = pd.DataFrame({"id": [0], "comment": [sentence], "sentiment": [0]})
    processor = DataProcessForSingleSentence(bert_tokenizer=tokenizer)
    input_data = next(iter(processor.get_input(data_test, max_seq_len)))
    input_data = tuple(t.unsqueeze(0).to(device) for t in input_data)

    model.eval()
    with torch.no_grad():
        logits = model(*input_data[:-1])
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description="金融新闻标题情感分析 - 单句预测")
    parser.add_argument("--model", default="fine_tuned_chinese_bert_cpu.bin",
                        help="微调后的模型文件路径")
    parser.add_argument("--bert", default="bert-base-chinese",
                        help="分词器对应的预训练模型名称(仅用于加载词表)")
    parser.add_argument("--max-seq-len", type=int, default=200, help="序列最大长度")
    parser.add_argument("--text", required=True, help="待预测的新闻标题")
    args = parser.parse_args()

    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(args.model, map_location=device)
    tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=True)

    predicted_class, probabilities = predict_single_sentence(
        model, tokenizer, args.text, args.max_seq_len, device)

    print(f"标题：{args.text}")
    print(f"预测结果：{LABELS[predicted_class]}，"
          f"置信度：{probabilities[predicted_class]:.4f}")
    print(f"各类别概率：{dict(zip(LABELS, probabilities.round(4)))}")
    print(f"耗时 {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
