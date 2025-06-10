import csv
import random
import os

def ensure_csv_file_exists(file_path, header):
    if not os.path.exists(file_path):
        print(f"{file_path} 不存在，创建新文件...")
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def merge_and_shuffle_csv(csv_file1, csv_file2, output_csv):
    rows = []

    # 假定你的表头是固定的，可以提前定义
    expected_header = ["text", "label", "domain"]

    # 确保输入文件存在，否则创建空文件（只有表头）
    ensure_csv_file_exists(csv_file1, expected_header)
    ensure_csv_file_exists(csv_file2, expected_header)

    # 读取第一个CSV文件（跳过表头）
    with open(csv_file1, 'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        header = next(reader)  # 读取表头
        for row in reader:
            rows.append(row)

    # 读取第二个CSV文件（跳过表头）
    with open(csv_file2, 'r', encoding='utf-8') as f2:
        reader = csv.reader(f2)
        header2 = next(reader)
        if header2 != header:
            raise ValueError("两个CSV文件的表头不匹配！")
        for row in reader:
            rows.append(row)

    # 打乱数据顺序
    random.shuffle(rows)

    # 写入合并并打乱后的CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)  # 写入表头
        writer.writerows(rows)

    print(f"合并并打乱后的CSV文件已保存到：{output_csv}")

if __name__ == "__main__":
    csv1 = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/zh_domain/zh_hm_webnovel.csv"
    csv2 = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/zh_domain/zh_llm_webnovel.csv"
    out_csv = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/zh_domain/zh_webnovel.csv"
    merge_and_shuffle_csv(csv1, csv2, out_csv)
