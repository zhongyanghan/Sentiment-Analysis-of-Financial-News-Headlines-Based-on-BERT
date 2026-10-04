# -*- coding: utf-8 -*-
"""
可视化脚本:
1. 解析 result.txt 中的评估报告, 绘制各指标随迭代次数的变化曲线;
2. (可选)基于训练集生成中文词云, 需要中文字体文件(如 simkai.ttf)。

用法:
    python src/visualize.py --log results/result.txt --output-dir results
    python src/visualize.py --wordcloud --data data/train_test_data.xlsx --font simkai.ttf
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_report(filename: str) -> pd.DataFrame:
    """解析 classification_report 日志, 提取各轮指标。"""
    with open(filename, "r", encoding="utf8") as file:
        data = file.readlines()

    accuracy = []
    macro_precision, macro_recall, macro_f1 = [], [], []
    weighted_precision, weighted_recall, weighted_f1 = [], [], []

    for line in data:
        if "accuracy" in line:
            accuracy.append(float(line.split()[1]))
        elif "macro avg" in line:
            parts = line.split()
            macro_precision.append(float(parts[2]))
            macro_recall.append(float(parts[3]))
            macro_f1.append(float(parts[4]))
        elif "weighted avg" in line:
            parts = line.split()
            weighted_precision.append(float(parts[2]))
            weighted_recall.append(float(parts[3]))
            weighted_f1.append(float(parts[4]))

    return pd.DataFrame({
        "Iteration": range(1, len(accuracy) + 1),
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1-Score": macro_f1,
        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall,
        "Weighted F1-Score": weighted_f1,
    })


def plot_metrics(df: pd.DataFrame, output_dir: str) -> None:
    """绘制各指标随迭代次数变化的折线图。"""
    sns.set_theme(style="whitegrid")
    metrics = [
        ("Accuracy", "purple"),
        ("Macro Precision", "b"),
        ("Macro Recall", "g"),
        ("Macro F1-Score", "r"),
        ("Weighted Precision", "orange"),
        ("Weighted Recall", "cyan"),
        ("Weighted F1-Score", "magenta"),
    ]
    for name, color in metrics:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="Iteration", y=name, marker="o", color=color)
        plt.title(f"{name} per Iteration", fontsize=16)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel(name, fontsize=12)
        filename = f"{name.replace(' ', '_').replace('-', '_').lower()}_per_iteration.png"
        plt.savefig(f"{output_dir}/{filename}", bbox_inches="tight")
        plt.show()
        plt.close()
    print(f"指标曲线已保存至: {output_dir}/")


def plot_wordcloud(data_path: str, font_path: str, output_dir: str) -> None:
    """基于训练集文本生成词云(需要 jieba 与 wordcloud 库)。"""
    import jieba
    from wordcloud import WordCloud

    df = pd.read_excel(data_path)
    text = " ".join(df["comment"].dropna().astype(str))
    seg_str = " ".join(jieba.cut(text, cut_all=False))

    wordcloud = WordCloud(
        font_path=font_path, width=800, height=400,
        background_color="white",
    ).generate(seg_str)

    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f"{output_dir}/wordcloud.png", bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()
    print(f"词云已保存至: {output_dir}/wordcloud.png")


def main():
    parser = argparse.ArgumentParser(description="金融新闻标题情感分析 - 可视化")
    parser.add_argument("--log", default="results/result.txt",
                        help="评估报告日志路径")
    parser.add_argument("--output-dir", default="results", help="图片输出目录")
    parser.add_argument("--wordcloud", action="store_true",
                        help="同时生成词云图")
    parser.add_argument("--data", default="data/train_test_data.xlsx",
                        help="词云使用的数据文件")
    parser.add_argument("--font", default="simkai.ttf",
                        help="词云使用的中文字体文件路径")
    args = parser.parse_args()

    df = parse_report(args.log)
    print(f"共解析 {len(df)} 轮评估结果")
    print(df.describe().loc[["min", "max"]])
    plot_metrics(df, args.output_dir)

    if args.wordcloud:
        plot_wordcloud(args.data, args.font, args.output_dir)


if __name__ == "__main__":
    main()
