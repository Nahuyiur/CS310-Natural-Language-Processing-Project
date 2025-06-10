import pandas as pd
import os

def split_csv_by_domain_label(input_csv, output_dir):
    df = pd.read_csv(input_csv)

    os.makedirs(output_dir, exist_ok=True)

    # 这里用字典缓存每个文件对应的数据，减少重复读写
    file_data = {}

    for _, row in df.iterrows():
        domain = row['domain']
        label = row['label']
        if label == 1:
            base_name = "eng_llm"
        else:
            base_name = "eng_hm"

        filename = f"{base_name}_{domain}.csv"
        filepath = os.path.join(output_dir, filename)

        # 把行转换成DataFrame
        row_df = pd.DataFrame([row])

        # 缓存追加
        if filepath in file_data:
            file_data[filepath].append(row_df)
        else:
            file_data[filepath] = [row_df]

    # 写入所有缓存的数据到对应文件
    for filepath, dfs in file_data.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        # 写文件时，如果文件已存在就覆盖
        combined_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"已保存文件：{filepath}, 行数：{len(combined_df)}")

if __name__ == "__main__":
    input_file =  "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/eng_mix/mix_eng_all.csv"
    output_folder = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/eng_domain"
    split_csv_by_domain_label(input_file, output_folder)
