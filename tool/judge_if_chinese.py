import pandas as pd
import re

def contains_chinese(text):
    return bool(re.search('[\u4e00-\u9fff]', str(text)))

def extract_context(text, window=5):
    # 用空白分词（简单切词，也可用jieba分词更精确）
    words = str(text).split()
    # 找到第一个含中文字符的词的位置
    for i, word in enumerate(words):
        if contains_chinese(word):
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            context_words = words[start:end]
            return ' '.join(context_words)
    return ''

def check_and_print_chinese_context(csv_path, text_column='text'):
    df = pd.read_csv(csv_path)
    for idx, text in enumerate(df[text_column]):
        if contains_chinese(text):
            context = extract_context(text)
            print(f"Line {idx}: {context}")

# 示例用法，替换成你的csv文件路径
csv_file_path = '/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/test.csv'
check_and_print_chinese_context(csv_file_path)
