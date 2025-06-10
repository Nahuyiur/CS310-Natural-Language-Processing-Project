import os 
import json
def filter_lines_by_keyword(file_a_path, file_b_path, keywords, out_a_path, out_b_path):
    """
    给定两个文件路径及关键词列表，读取文件并同步过滤含关键词的行，
    并将结果写入新的文件。

    :param file_a_path: 文件A路径
    :param file_b_path: 文件B路径
    :param keywords: 关键词列表或字符串（单个关键词）
    :param out_a_path: 输出过滤后文件A路径
    :param out_b_path: 输出过滤后文件B路径
    """
    # 确保keywords是列表
    if isinstance(keywords, str):
        keywords = [keywords]

    with open(file_a_path, 'r', encoding='utf-8') as fa, open(file_b_path, 'r', encoding='utf-8') as fb:
        lines_a = fa.readlines()
        lines_b = fb.readlines()

    if len(lines_a) != len(lines_b):
        raise ValueError("两个文件行数不一致，无法对齐过滤！")

    filtered_a = []
    filtered_b = []

    for line_a, line_b in zip(lines_a, lines_b):
        # 检查两个文件任一行是否包含任意关键词
        if any(kw in line_a for kw in keywords) or any(kw in line_b for kw in keywords):
            # 该行过滤掉，不加入结果
            continue
        filtered_a.append(line_a)
        filtered_b.append(line_b)

    with open(out_a_path, 'w', encoding='utf-8') as fa_out, open(out_b_path, 'w', encoding='utf-8') as fb_out:
        fa_out.writelines(filtered_a)
        fb_out.writelines(filtered_b)

    print(f"过滤完成，输出文件：{out_a_path}，{out_b_path}")

# 示例调用
file_a = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_txt/zh/zh_wiki_hm.txt"
file_b = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_txt/zh/zh_wiki_llm.txt"
keywords = ["本条数据记录为空，这是占位内容。"] 
output_a = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_txt/zh/zh_wiki_hm(fil).txt"
output_b = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_txt/zh/zh_wiki_llm(fil).txt"

filter_lines_by_keyword(file_a, file_b, keywords, output_a, output_b)
