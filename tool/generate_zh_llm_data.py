import json
import csv

def json_to_csv(json_path, csv_path):
    # 读json文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    label = 1
    domain = "news"

    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "label", "domain"])

        for key, input_text in data.get("input", {}).items():
            output_text = data.get("output", {}).get(key, "").strip()
            input_text = input_text.replace('\n', '').strip()
            output_text = output_text.replace('\n', '').strip()
            if not output_text:
                continue
            full_text = output_text
            writer.writerow([full_text, label, domain])

    print(f"CSV文件已生成：{csv_path}")

if __name__ == "__main__":
    json_file = "/lab/haoq_lab/cse12310520/NLP/proj/face2_zh_json/generated/zh_qwen2/news-zh.qwen2-72b-base.json"
    csv_file = "zh_llm_news.csv"
    json_to_csv(json_file, csv_file)
