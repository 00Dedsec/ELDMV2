import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
from transformers import AutoTokenizer


class PLIFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        # self.tokenizer = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        
        self.max_para_q = config.getint("data", "max_para_q")
        self.max_para_c = config.getint("data", "max_para_c")

    def convert_tokens_to_ids(self, text, text_pair):
        return self.tokenizer(text, text_pair, padding='max_length', max_length=self.max_len, add_special_tokens=True, truncation=True)

    def process(self, data, config, mode, *args, **params):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        query_candidate_id = []
        label = []
        for x in data:
            text_q = x['query']['q'] #[sent, max_len]
            text_c = x['candidate']['ajjbqk'] #[sent, max_len]
            input_ids_item = []
            attention_mask_item = []
            token_type_ids_item = []

            for m in range(min(self.max_para_q, len(text_q))):
                p1 = text_q[m]
                input_ids_row = []  # q段落m和doc所有段落的pair
                attention_mask_row = []
                token_type_ids_row = []
                for n in range(min(self.max_para_c, len(text_c))):
                    p2 = text_c[n]
                    res_dict = self.convert_tokens_to_ids(p1, p2)
                    input_ids_row.append(res_dict['input_ids'])
                    attention_mask_row.append(res_dict['attention_mask'])
                    token_type_ids_row.append(res_dict['token_type_ids'])
                
                if(len(text_c)<self.max_para_c):
                    for j in range(len(text_c), self.max_para_c):
                        input_ids_row.append([0]*self.max_len)
                        attention_mask_row.append([0]*self.max_len)
                        token_type_ids_row.append([0]*self.max_len)

                assert (len(input_ids_row) == self.max_para_c)
                assert (len(attention_mask_row) == self.max_para_c)
                assert (len(token_type_ids_row) == self.max_para_c)
                
                input_ids_item.append(input_ids_row)
                attention_mask_item.append(attention_mask_row)
                token_type_ids_item.append(token_type_ids_row)

            if len(text_q) < self.max_para_q:   # 补充文本q
                for i in range(len(text_q), self.max_para_q):
                    input_ids_row = []  #
                    attention_mask_row = []
                    token_type_ids_row = []
                    for j in range(self.max_para_c):
                        input_ids_row.append([0]*self.max_len)
                        attention_mask_row.append([0]*self.max_len)
                        token_type_ids_row.append([0]*self.max_len)

                    input_ids_item.append(input_ids_row)
                    attention_mask_item.append(attention_mask_row)
                    token_type_ids_item.append(token_type_ids_row)
            
            assert (len(input_ids_item) == self.max_para_q)
            assert (len(attention_mask_item) == self.max_para_q)
            assert (len(token_type_ids_item) == self.max_para_q)

            input_ids.append(input_ids_item)
            attention_mask.append(attention_mask_item)
            token_type_ids.append(token_type_ids_item)

            if(mode != 'test'):
                label.append(x['label'])
            query_candidate_id.append((x['query']['ridx'], x['candidate']['candidate_id']))

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)
        label = torch.LongTensor(label)

        return {
                "input_ids": input_ids, 
                "attention_mask": attention_mask, 
                "token_type_ids": token_type_ids,
                "query_candidate_id": query_candidate_id, 
                "label": label
                }
