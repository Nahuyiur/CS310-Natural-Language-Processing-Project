def check_empty_lines(file_path):
    empty_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            if line.strip() == '':
                empty_lines.append(i)

    if empty_lines:
        print(f"文件 {file_path} 存在空行，行号如下：")
        for lineno in empty_lines:
            print(lineno)
    else:
        print(f"文件 {file_path} 中没有空行。")

# 示例用法：
file_path = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_txt/eng/eng_essay_hm.txt"
check_empty_lines(file_path)
