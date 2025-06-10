import pandas as pd
import random

def merge_shuffle_split(csv_files, train_path, val_path, test_path, seed=42):
    # 读取并合并所有csv文件
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # 打乱数据
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 计算划分索引
    total = len(data)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.2)

    # 划分数据集
    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]

    # 保存csv文件
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"数据合并完成，总样本数: {total}")
    print(f"训练集样本数: {len(train_df)}，验证集样本数: {len(val_df)}，测试集样本数: {len(test_df)}")
    print(f"文件保存为：\n 训练集：{train_path}\n 验证集：{val_path}\n 测试集：{test_path}")

if __name__ == "__main__":
    # 这里你可以手动填入CSV文件列表，或者用input输入路径并分割成列表
    csv_list = [
        "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/zh_domain/zh_hm_news.csv",
        "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/zh_domain/zh_llm_news.csv",
        "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/zh_domain/zh_hm_webnovel.csv",
        "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/zh_domain/zh_llm_webnovel.csv",
        "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/zh_domain/zh_hm_wiki.csv",
        "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/zh_domain/zh_llm_wiki.csv"
    ]

    train_file = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/zh_mix/zh_train.csv"
    val_file = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/zh_mix/zh_val.csv"
    test_file = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/zh_mix/zh_test.csv"

    merge_shuffle_split(csv_list, train_file, val_file, test_file)
