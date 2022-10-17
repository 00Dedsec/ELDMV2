import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
from transformers import AutoTokenizer


class ELDMFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        # self.tokenizer = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.max_para_q = config.getint("data", "max_para_q")
        self.max_para_c = config.getint("data", "max_para_c")

    def convert_tokens_to_ids(self, text):
        return self.tokenizer(text, padding='max_length', max_length=self.max_len, add_special_tokens=True, truncation=True)

    def process(self, data, config, mode, *args, **params):
        input_ids_q = [] #[b, sent, max_len]
        attn_mask_q = []
        token_type_ids_q = []
        input_ids_c = []
        attn_mask_c = []
        token_type_ids_c = []
        query_candidate_id = []
        label = []
        for x in data:
            text_q = x['query']['q'] #[sent, max_len]
            text_c = x['candidate']['ajjbqk'] #[sent, max_len]
            input_ids_q_sent = [] #[sent, max_len]
            attn_mask_q_sent = [] 
            token_type_ids_q_sent = []
            for i in range(0, min(len(text_q), self.max_para_q)):
                re_q = self.convert_tokens_to_ids(text_q)
                input_ids_q_sent.append(re_q.input_ids)
                attn_mask_q_sent.append(re_q.attention_mask)
                token_type_ids_q_sent.append(re_q.token_type_ids)

            input_ids_c_sent = [] #[sent, max_len]
            attn_mask_c_sent = [] 
            token_type_ids_c_sent = []
            for i in range(0, min(len(text_c), self.max_para_c)):
                re_c = self.convert_tokens_to_ids(text_c)
                input_ids_c_sent.append(re_c.input_ids)
                attn_mask_c_sent.append(re_c.attention_mask)
                token_type_ids_c_sent.append(re_c.token_type_ids)
            
            input_ids_q.append(input_ids_q_sent)
            attn_mask_q.append(attn_mask_q_sent)
            token_type_ids_q.append(token_type_ids_q_sent)                    
            input_ids_c.append(input_ids_c_sent)
            attn_mask_c.append(attn_mask_c_sent)
            token_type_ids_c.append(token_type_ids_c_sent)

            if(mode != 'test'):
                label.append(x['label'])
            query_candidate_id.append((x['query']['ridx'], x['candidate']['candidate_id']))
        
        input_ids_q = torch.LongTensor(input_ids_q)
        attn_mask_q = torch.LongTensor(attn_mask_q)
        token_type_ids_q = torch.LongTensor(token_type_ids_q)
        input_ids_c = torch.LongTensor(input_ids_c)
        attn_mask_c = torch.LongTensor(attn_mask_c)
        token_type_ids_c = torch.LongTensor(token_type_ids_c)

        return {
                "input_ids_q": input_ids_q, 
                "attention_mask_q": attn_mask_q, 
                "token_type_ids_q": token_type_ids_q,
                "input_ids_c": input_ids_c, 
                "attention_mask_c": attn_mask_c, 
                "token_type_ids_c": token_type_ids_c,
                "query_candidate_id": query_candidate_id, 
                "label": label
                }
