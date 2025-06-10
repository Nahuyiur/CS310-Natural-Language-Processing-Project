import json

def append_json_to_txt(json_file_path: str, output_txt_path: str):
    placeholder = "本条数据记录为空，这是占位内容。"

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    inputs = data.get('input', {})
    outputs = data.get('output', {})

    with open(output_txt_path, 'a', encoding='utf-8') as fout:
        for key in sorted(inputs.keys(), key=lambda x: int(x)):
            input_text = inputs[key].replace('\r\n', '  ').replace('\n', '  ').replace('\r', '  ')
            output_text = outputs.get(key, "").strip()

            if len(output_text.strip()) < 3:
                print(f"警告：index {key} 的output长度不足，使用占位内容替代。")
                output_text = placeholder

            else:
                output_text = output_text.replace('\r\n', '  ').replace('\n', '  ').replace('\r', '  ')

            fout.write(output_text + '\n')

    print(f"数据已追加写入到 {output_txt_path}")

def append_json_list_to_txt(json_file_path: str, output_txt_path: str):
    placeholder = "本条数据记录为空，这是占位内容。"

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    with open(output_txt_path, 'a', encoding='utf-8') as fout:
        for idx, item in enumerate(data_list):
            input_text = item.get("input", "").replace('\r\n', '  ').replace('\n', '  ').replace('\r', '  ').strip()
            output_text = item.get("output", "").replace('\r\n', '  ').replace('\n', '  ').replace('\r', '  ').strip()

            if len(output_text) < 3:
                print(f"警告：第 {idx} 条数据的output长度不足，使用占位内容替代。")
                combined_text = placeholder
            else:
                combined_text = input_text + " " + output_text

            fout.write(combined_text + '\n')

    print(f"数据已追加写入到 {output_txt_path}")

# 调用示例
json_path = "/lab/haoq_lab/cse12310520/NLP/proj/face2_zh_json/generated/zh_qwen2/webnovel.qwen2-72b-base.json"
txt_path = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_txt/zh/zh_webnovel_llm.txt"
append_json_to_txt(json_path, txt_path)
