import json
import os
from torch.utils.data import Dataset
import random

from tools.dataset_tool import dfs_search


class Json_sent_from_files(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding

        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")
        recursive = False

        for name in filename_list:
            self.file_list = self.file_list + dfs_search(os.path.join(self.data_path, name), recursive)
        self.file_list.sort()

        self.data = []
        for filename in self.file_list:
            f = open(filename, "r", encoding=encoding)
            for line in f:
                #每句话以及对应的标签放在一起
                doc = json.loads(line)
                content = doc['content']
                events = doc['events']
                for item in content:
                    item['events'] = []
                for event in events:
                    for mention in event['mention']:
                        content[mention['sent_id']]['events'].append(event)

                for item in content:
                    self.data.append(item)

        if mode == "train":
            random.shuffle(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)