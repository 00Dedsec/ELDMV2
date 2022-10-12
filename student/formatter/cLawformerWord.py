import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
from transformers import AutoTokenizer


class cLawformerWordFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        # self.tokenizer = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))
        self.tokenizer = AutoTokenizer.from_pretrained("thunlp/Lawformer")
        self.max_len_q = config.getint("data", "max_seq_length_q")
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode

    def convert_tokens_to_ids(self, text_q, text_c):
        text_q = text_q[:self.max_len_q:1]

        return self.tokenizer(text_q, text_c, padding='max_length', max_length=self.max_len, truncation=True)
        
    def process(self, data, config, mode, *args, **params):
        input_ids = []
        attention_mask = []
        query_candidate_id = []
        label = []
        for x in data:
            text_q = x['query']['q']
            text_c = x['candidate']['ajjbqk']
            text = self.convert_tokens_to_ids(text_q, text_c)
            input_ids.append(text.input_ids)
            attention_mask.append(text.attention_mask)
            if(mode != 'test'):
                label.append(x['label'])
            query_candidate_id.append((x['query']['ridx'], x['candidate']['candidate_id']))

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        label = torch.LongTensor(label)

        return {"input_ids": input_ids, 
                "attention_mask": attention_mask,
                "query_candidate_id": query_candidate_id, 
                "label": label}
