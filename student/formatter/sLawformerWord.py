import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
from transformers import AutoTokenizer


class sLawformerWordFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        # self.tokenizer = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))
        self.tokenizer = AutoTokenizer.from_pretrained("thunlp/Lawformer")
        self.max_len_q = config.getint("data", "max_seq_length_q")
        self.max_len_c = config.getint("data", "max_seq_length_c")
        self.mode = mode

    def convert_tokens_to_ids(self, text, mode):
        if mode == 'q':
            return self.tokenizer(text, padding='max_length', max_length=self.max_len_q, truncation=True)
        else:
            return self.tokenizer(text, padding='max_length', max_length=self.max_len_c, truncation=True)

    def process(self, data, config, mode, *args, **params):
        q_input_ids = []
        c_input_ids = []
        q_attention_mask = []
        c_attention_mask = []
        query_candidate_id = []
        label = []
        for x in data:
            text_q = x['query']['q']
            text_c = x['candidate']['ajjbqk']
            q_ = self.convert_tokens_to_ids(text_q, 'q')
            c_ = self.convert_tokens_to_ids(text_c, 'c')
            q_input_ids.append(q_.input_ids)
            c_input_ids.append(c_.input_ids)
            q_attention_mask.append(q_.attention_mask)
            c_attention_mask.append(c_.attention_mask)


            if(mode != 'test'):
                label.append(x['label'])
            query_candidate_id.append((x['query']['ridx'], x['candidate']['candidate_id']))

        q_input_ids = torch.LongTensor(q_input_ids)
        c_input_ids = torch.LongTensor(c_input_ids)
        q_attention_mask = torch.LongTensor(q_attention_mask)
        c_attention_mask = torch.LongTensor(c_attention_mask)
        label = torch.LongTensor(label)

        return {"q_input_ids": q_input_ids, 
                "c_input_ids": c_input_ids, 
                "q_attention_mask": q_attention_mask,
                "c_attention_mask": c_attention_mask,
                "query_candidate_id": query_candidate_id, 
                "label": label}
