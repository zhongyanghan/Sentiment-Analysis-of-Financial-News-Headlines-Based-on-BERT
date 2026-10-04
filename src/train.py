# -*- coding: utf-8 -*-
"""
训练脚本: 微调 BERT 进行金融新闻标题情感分类。

每轮训练结束后在测试集上评估, 将 classification_report 追加写入日志文件,
并保存最终模型。评估日志可用 visualize.py 绘制指标曲线。

用法:
    python src/train.py --data data/train_test_data.xlsx --bert bert-base-chinese
"""

import argparse
import time

import torch
from torch import nn
from tqdm import tqdm

from pytorch_pretrained_bert.optimization import BertAdam

from model import ClassifyModel, evaluate_accuracy, load_data


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---------- 数据加载 ----------
    train_iter, test_iter, train_batch_count, test_batch_count = load_data(
        args.data, args.bert, args.max_seq_len, args.batch_size)

    # ---------- 模型构建 ----------
    model = ClassifyModel(args.bert, num_labels=args.num_labels, is_lock=args.lock)
    model.to(device)
    optimizer = BertAdam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    # ---------- 训练循环 ----------
    train_loss_history, train_acc_history = [], []
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for _, batch_data in tqdm(
                enumerate(train_iter),
                desc=f"train epoch:{epoch + 1}/{args.epochs}",
                total=train_batch_count):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data

            logits = model(batch_seqs, batch_seq_masks, batch_seq_segments)
            loss = loss_func(logits, batch_labels)
            loss.backward()
            train_loss_sum += loss.item()
            preds = logits.softmax(dim=1).argmax(dim=1)
            train_acc_sum += (preds == batch_labels).sum().item()
            n += batch_labels.shape[0]
            optimizer.step()
            optimizer.zero_grad()

        train_loss_history.append(train_loss_sum / n)
        train_acc_history.append(train_acc_sum / n)

        # 每轮在测试集上评估
        report = evaluate_accuracy(test_iter, model, device, test_batch_count)
        print(f"epoch {epoch + 1}, loss {train_loss_sum / n:.4f}, "
              f"train acc {train_acc_sum / n:.3f}, time: {time.time() - start:.3f}")
        print(report)
        with open(args.log, "a", encoding="utf8") as f:
            f.write(report + "\n")

    # ---------- 保存模型与训练曲线数据 ----------
    torch.save(model, args.output)
    print(f"模型已保存至: {args.output}")
    with open(args.curve, "w", encoding="utf8") as f:
        f.write("epoch\ttrain_loss\ttrain_acc\n")
        for i, (l, a) in enumerate(zip(train_loss_history, train_acc_history), 1):
            f.write(f"{i}\t{l:.6f}\t{a:.6f}\n")
    print(f"训练曲线数据已保存至: {args.curve}")


def main():
    parser = argparse.ArgumentParser(description="金融新闻标题情感分析 - 模型训练")
    parser.add_argument("--data", default="data/train_test_data.xlsx",
                        help="训练/测试数据 Excel 路径")
    parser.add_argument("--bert", default="bert-base-chinese",
                        help="预训练模型名称或本地路径")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=512, help="批大小")
    parser.add_argument("--lr", type=float, default=4e-5, help="学习率")
    parser.add_argument("--max-seq-len", type=int, default=200,
                        help="序列最大长度(不超过 512)")
    parser.add_argument("--num-labels", type=int, default=2, help="分类类别数")
    parser.add_argument("--lock", action="store_true",
                        help="冻结 BERT 主体参数(默认冻结, 与原实验一致)")
    parser.add_argument("--no-lock", dest="lock", action="store_false",
                        help="不冻结 BERT 参数, 进行全量微调")
    parser.set_defaults(lock=True)
    parser.add_argument("--output", default="fine_tuned_chinese_bert.bin",
                        help="模型保存路径")
    parser.add_argument("--log", default="results/result.txt",
                        help="评估报告日志路径")
    parser.add_argument("--curve", default="results/train_curve.txt",
                        help="训练 loss/acc 曲线数据保存路径")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
