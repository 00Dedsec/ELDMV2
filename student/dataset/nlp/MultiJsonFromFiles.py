import json
import os
from torch.utils.data import Dataset
import random
from copy import deepcopy

from tqdm import tqdm
from utils.logger import Logger
from utils.augmentation import eda
"""
{
    query: query
    candidate: {
        原始candidate + candidate_id
    }
    label: 3~0 最相关~不相关 
}
"""
logger = Logger(__name__)
class MultiJsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.encoding = encoding

        self.data = []

        # 先读取query数据
        if mode == 'train':
            f_query = open(self.config.get("data", "train_query_data_path"), encoding='utf-8')
            for line in f_query:
                query_json = json.loads(line)
                ridx = query_json['ridx']
                # 获取该query的候选案例列表
                candidate_file_name_list = os.listdir(self.config.get('data','train_candidates_data_path') + '/' + str(ridx))
                for candidate_file_name in candidate_file_name_list:
                    data_item = {}
                    candidate_file = open(self.config.get('data','train_candidates_data_path') + '/' + str(ridx) + '/' + candidate_file_name, encoding='utf-8')
                    candidate_json = json.loads(candidate_file.readline(), encoding='utf-8')
                    candidate_json['candidate_id'] = candidate_file_name.split('.')[0]
                    data_item['query'] = query_json
                    data_item['candidate'] = candidate_json
                    self.data.append(data_item)
            f_label = open(self.config.get("data","train_label_top30_data_path"), encoding='utf-8')
            label_json = json.loads(f_label.readline(), encoding='utf-8')
            for data_item in self.data:
                query_label_json = label_json[str(data_item['query']['ridx'])]
                if data_item['candidate']['candidate_id'] in query_label_json.keys():
                    data_item['label'] = query_label_json[data_item['candidate']['candidate_id']]
                else:
                    data_item['label'] = 0
            
            # 数据增广
            length = len(self.data)
            logger.get_log().info("开始数据增广...")
            logger.get_log().info('增广前：' + str(length))
            with tqdm(total = length) as bar:
                for i in range(0, length):
                    if self.data[i]['label'] != 0:

                        # q_aug = eda(self.data[i]['query']['q'])
                        # c_aug = eda(self.data[i]['candidate']['ajjbqk'])
                        # for i in range(0, len(q_aug)):
                        #     data = deepcopy(self.data[i])
                        #     data['query']['q'] = q_aug[i]
                        #     data['candidate']['ajjbqk'] = c_aug[i]
                        #     self.data.append(data)

                        self.data.append(self.data[i])
                    bar.update(1)
                bar.close()
            logger.get_log().info('增广后：' + str(len(self.data)))
            

        elif mode == 'valid':
            f_query = open(self.config.get("data", "valid_query_data_path"), encoding='utf-8')
            for line in f_query:
                query_json = json.loads(line)
                ridx = query_json['ridx']
                # 获取该query的候选案例列表
                candidate_file_name_list = os.listdir(self.config.get('data','valid_candidates_data_path') + '/' + str(ridx))
                for candidate_file_name in candidate_file_name_list:
                    data_item = {}
                    candidate_file = open(self.config.get('data','valid_candidates_data_path') + '/' + str(ridx) + '/' + candidate_file_name, encoding='utf-8')
                    candidate_json = json.loads(candidate_file.readline(), encoding='utf-8')
                    candidate_json['candidate_id'] = candidate_file_name.split('.')[0]
                    data_item['query'] = query_json
                    data_item['candidate'] = candidate_json
                    self.data.append(data_item)
            f_label = open(self.config.get("data","valid_label_top30_data_path"), encoding='utf-8')
            label_json = json.loads(f_label.readline(), encoding='utf-8')
            for data_item in self.data:
                query_label_json = label_json[str(data_item['query']['ridx'])]
                if data_item['candidate']['candidate_id'] in query_label_json.keys():
                    data_item['label'] = query_label_json[data_item['candidate']['candidate_id']]
                else:
                    data_item['label'] = 0
                    
        elif mode == 'test':
            f_query = open(self.config.get("data", "test_query_data_path"), encoding='utf-8')
            for line in f_query:
                query_json = json.loads(line, encoding='utf-8')
                ridx = query_json['ridx']
                # 获取该query的候选案例列表
                candidate_file_name_list = os.listdir(self.config.get('data','test_candidates_data_path') + '/' + str(ridx))
                for candidate_file_name in candidate_file_name_list:
                    data_item = {}
                    candidate_file = open(self.config.get('data','test_candidates_data_path') + '/' + str(ridx) + '/' + candidate_file_name, encoding='utf-8')
                    candidate_json = json.loads(candidate_file.readline(), encoding='utf-8')
                    candidate_json['candidate_id'] = candidate_file_name.split('.')[0]
                    data_item['query'] = query_json
                    data_item['candidate'] = candidate_json
                    self.data.append(data_item)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)