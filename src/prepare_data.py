# -*- coding: utf-8 -*-
"""
数据准备脚本: 将原始 train_dict.txt 转换为 Excel 训练/测试集。

原始数据格式: 每行一个 Python 字典 ``{"title": ..., "label": ...}``,
其中 label 为字符串 "1"(积极) 或 "0"(消极)。

用法:
    python src/prepare_data.py --input data/train_dict.txt --output data/train_test_data.xlsx
"""

import argparse
import random

import pandas as pd


def prepare_data(input_path: str, output_path: str,
                 test_ratio: float = 0.2, seed: int = 42) -> None:
    """读取原始数据, 打乱后按比例切分并保存为 Excel(train/test 两个 sheet)。"""
    with open(input_path, encoding="utf8") as f:
        data = [eval(line) for line in f if line.strip()]
    print(f"读取样本总数: {len(data)}")

    # 打乱数据顺序
    random.Random(seed).shuffle(data)

    # 按比例切分
    split_index = int((1 - test_ratio) * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]
    print(f"训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")

    train_df = pd.DataFrame(train_data, columns=["title", "label"])
    test_df = pd.DataFrame(test_data, columns=["title", "label"])
    # 统一列名: comment(文本) / sentiment(标签)
    train_df.columns = ["comment", "sentiment"]
    test_df.columns = ["comment", "sentiment"]

    with pd.ExcelWriter(output_path) as writer:
        train_df.to_excel(writer, sheet_name="train", index=False)
        test_df.to_excel(writer, sheet_name="test", index=False)
    print(f"已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="金融新闻标题情感分析 - 数据准备")
    parser.add_argument("--input", default="data/train_dict.txt",
                        help="原始数据路径(train_dict.txt)")
    parser.add_argument("--output", default="data/train_test_data.xlsx",
                        help="输出的 Excel 文件路径")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="测试集比例, 默认 0.2 (即 8:2 切分)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    prepare_data(args.input, args.output, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
