import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer
from torch.utils.data import DataLoader

class TextClassificationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512, text_column='text', label_column='label', domain_column='domain'):
        """
        :param csv_file: CSV文件路径
        :param tokenizer: 预训练模型对应的Tokenizer
        :param max_length: 最大序列长度
        :param text_column: 文本所在列名
        :param label_column: 标签所在列名
        :param domain_column: 领域所在列名（可选）
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.domain_column = domain_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.loc[idx, self.text_column])
        if pd.isna(self.data.loc[idx, self.label_column]):
            print(f"[DEBUG] Row {idx} has NaN label")
        label = int(self.data.loc[idx, self.label_column])
        domain = str(self.data.loc[idx, self.domain_column]) if self.domain_column in self.data.columns else ''

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        item['domain'] = domain  # 额外返回domain字段（字符串）

        return item


def test_local_bert():
    csv_path = '/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/eng_domain/eng_hm_essay.csv'  # 替换成你的CSV路径

    # 加载本地bert-base-uncased tokenizer和模型（需要网络或已缓存）
    tokenizer = BertTokenizer.from_pretrained('/lab/haoq_lab/cse12310520/NLP/proj/bert-base-uncased')

    dataset = TextClassificationDataset(
        csv_file=csv_path,
        tokenizer=tokenizer,
        max_length=128,
        text_column='text',
        label_column='label',
        domain_column='domain'
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print("input_ids:", batch['input_ids'])
        print("attention_mask:", batch['attention_mask'])
        print("labels:", batch['labels'])
        print("domains:", batch['domain'])
        print()
        if i >= 10:
            break

if __name__ == "__main__":
    test_local_bert()