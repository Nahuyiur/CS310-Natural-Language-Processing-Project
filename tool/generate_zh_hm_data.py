import json
import csv

def json_to_csv(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    domain = "news"

    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "label", "domain"])

        for item in data:
            output_text = item.get("output", "").strip()
            output_text = output_text.replace('\n', '').strip()
            # 如果output为空就跳过
            if not output_text:
                continue

            text = output_text
            label = 0  # 固定为0

            writer.writerow([text, label, domain])

    print(f"CSV文件已生成：{csv_path}")

if __name__ == "__main__":
    json_file = "/lab/haoq_lab/cse12310520/NLP/proj/face2_zh_json/zh_unicode/news-zh.json"
    csv_file = "zh_hm_news.csv"
    json_to_csv(json_file, csv_file)
