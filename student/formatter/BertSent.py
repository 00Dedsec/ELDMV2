import json
import torch
import os
import numpy as np
import re
from transformers import AutoTokenizer

from formatter.Basic import BasicFormatter

class BertSentFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        # Lawformer需要使用bert相关函数加载.
        # self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.max_len = config.getint("data", "max_seq_length")
        self.max_sent_num = config.getint("data", "max_sent_num")
        self.mode = mode

    def convert_tokens_to_ids(self, text, max_len):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        text = re.split(r'[；;。sA]s*',text)
        for sent in text:
            token = self.tokenizer(sent, padding = 'max_length', truncation = 'longest_first', max_length = max_len)
            input_ids.append(token.input_ids)
            token_type_ids.append(token.token_type_ids)
            attention_mask.append(token.attention_mask)
        
        while len(input_ids) < self.max_sent_num:
            temp = self.tokenizer('', padding = 'max_length', max_length = max_len)
            input_ids.append(temp.input_ids)
            token_type_ids.append(temp.token_type_ids)
            attention_mask.append(temp.attention_mask)
        
        if len(input_ids) > self.max_sent_num:
            input_ids = input_ids[0:self.max_sent_num]
            token_type_ids = token_type_ids[0:self.max_sent_num]
            attention_mask = attention_mask[0:self.max_sent_num]


        
        return input_ids, token_type_ids, attention_mask

    # 传入的是一个batch
    def process(self, data, config, mode, *args, **params):
        query_input_ids = [] #[batch, sent, word]
        query_token_type_ids = []
        query_attention_mask = []

        candidate_input_ids = [] #[batch, sent, word]
        candidate_token_type_ids = []
        candidate_attention_mask = []

        query_candidate_id = []

        label = []

        for temp in data:
            query_json_item = temp['query']
            candidate_json_item = temp['candidate']
            if(mode != 'test'):
                label_json_item = temp['label']

            # 处理query
            query_token = self.convert_tokens_to_ids(query_json_item['q'], self.max_len) #[sent, max_len]
            query_input_ids.append(query_token[0])
            query_token_type_ids.append(query_token[1])
            query_attention_mask.append(query_token[2])

            #处理candidate
            candidate_token = self.convert_tokens_to_ids(candidate_json_item['ajjbqk'], self.max_len)
            candidate_input_ids.append(candidate_token[0])
            candidate_token_type_ids.append(candidate_token[1])
            candidate_attention_mask.append(candidate_token[2])

            #处理label
            if(mode != 'test'):
                label.append(label_json_item)

            # 处理id
            temp = (0,0)
            temp = (query_json_item['ridx'], candidate_json_item['candidate_id'])
            query_candidate_id.append(temp)
        
        query_input_ids = torch.LongTensor(query_input_ids)
        query_token_type_ids = torch.LongTensor(query_token_type_ids)
        query_attention_mask = torch.LongTensor(query_attention_mask)

        candidate_input_ids = torch.LongTensor(candidate_input_ids)
        candidate_token_type_ids = torch.LongTensor(candidate_token_type_ids)
        candidate_attention_mask = torch.LongTensor(candidate_attention_mask)

        if(mode != 'test'):
            label = torch.LongTensor(label)
            return {
                "query_input_ids": query_input_ids, 
                "query_token_type_ids": query_token_type_ids,
                "query_attention_mask": query_attention_mask,
                "candidate_input_ids": candidate_input_ids, 
                "candidate_token_type_ids": candidate_token_type_ids,
                "candidate_attention_mask": candidate_attention_mask,
                "query_candidate_id": query_candidate_id, 
                "label": label,
            }
        else:
            return {
                "query_input_ids": query_input_ids, 
                "query_token_type_ids": query_token_type_ids,
                "query_attention_mask": query_attention_mask,
                "candidate_input_ids": candidate_input_ids, 
                "candidate_token_type_ids": candidate_token_type_ids,
                "candidate_attention_mask": candidate_attention_mask,
                "query_candidate_id": query_candidate_id,
            }