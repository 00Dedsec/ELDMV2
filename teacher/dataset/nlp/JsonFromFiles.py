import json
import os
from torch.utils.data import Dataset
import random

from tools.dataset_tool import dfs_search
from utils.bio_lables import get_labels
from transformers import AutoTokenizer
"""
{
    "doc": "*****"
    "label": 00111

}
"""
class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.data = []
        f = open(self.config.get("data", "train_data_path"), encoding='utf-8')
        sum_len = 0
        num = 0
        for line in f:
            doc = json.loads(line)
            words = [c['tokens'] for c in doc['content']]
            labels = [['O']*len(c['tokens']) for c in doc['content']]
            if mode != 'test':
                for event in doc['events']:
                    for mention in event['mention']:
                        labels[mention['sent_id']][mention['offset'][0]] = "B-" + event['type']
                        for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                            labels[mention['sent_id']][i] = "I-" + event['type']

                for mention in doc['negative_triggers']:
                    labels[mention['sent_id']][mention['offset'][0]] = "O"
                    for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                        labels[mention['sent_id']][i] = "O"

            for i in range(0, len(words)):
                self.data.append({
                    "words": words[i],
                    "labels": labels[i]
                })

            for i in range(0, len(words)):
                for word in words[i]:
                    sum_len += len(word)
                num = num + 1

        print(sum_len//num)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)