import os
import re

def append_txt_contents_to_file(
    top_folder_path: str, 
    output_file_path: str, 
    newline_replacement: str = '  ',  # 默认2个空格
    exclude_subfolders: list = None      # 要排除的二级文件夹名称列表
):
    """
    遍历一级文件夹下所有二级文件夹（除排除列表里的），按数字顺序读取每个二级文件夹里的所有txt文件，
    将文件内容中所有换行符替换为指定字符串（默认2个空格），然后追加写入到目标文件，
    并保留目标文件之前的内容。

    :param top_folder_path: 一级文件夹路径
    :param output_file_path: 目标文件路径，追加写入
    :param newline_replacement: 用于替换换行符的字符串，默认两个空格
    :param exclude_subfolders: 不处理的二级文件夹名称列表，默认不排除任何
    """
    if exclude_subfolders is None:
        exclude_subfolders = []

    def numeric_key(filename):
        # 提取文件名中的数字部分作为排序关键字
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else -1

    # 获取一级目录下所有二级文件夹（目录），排序
    subfolders = [f for f in os.listdir(top_folder_path) 
                  if os.path.isdir(os.path.join(top_folder_path, f)) and f not in exclude_subfolders]
    subfolders.sort()

    with open(output_file_path, 'a', encoding='utf-8') as fout:
        for subfolder in subfolders:
            print("处理文件夹:", subfolder)

            subfolder_path = os.path.join(top_folder_path, subfolder)
            # 获取二级文件夹下所有txt文件，按数字顺序排序
            txt_files = [f for f in os.listdir(subfolder_path) if f.endswith('.txt')]
            txt_files.sort(key=numeric_key)

            for filename in txt_files:
                file_path = os.path.join(subfolder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as fin:
                    content = fin.read()
                    # 将换行符替换成指定字符串（默认2个空格）
                    content_replaced = content.replace('\r\n', newline_replacement)\
                                              .replace('\n', newline_replacement)\
                                              .replace('\r', newline_replacement)
                    # 如果替换后内容为空，打印文件完整路径
                    if content_replaced.strip() == '':
                        print(f"警告: 文件内容为空，文件路径: {file_path}")
                        fout.write('The original txt is empty, this is just a placeholder.')
                    fout.write(content_replaced + '\n')

    print(f"所有符合条件的二级文件夹中的txt文件内容已追加到 {output_file_path}")


exclude1=['claude','gpt','gpt_prompt1','gpt_prompt2','gpt_semantic','gpt_writing','prompts']
exclude2=['human','prompts']

top_folder = "/lab/haoq_lab/cse12310520/NLP/proj/ghostbuster-data/essay"
output_file = "/lab/haoq_lab/cse12310520/NLP/proj/prepared_txt/eng/eng_essay_hm.txt"
append_txt_contents_to_file(top_folder_path=top_folder, output_file_path=output_file, exclude_subfolders=exclude1)
