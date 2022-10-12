import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
from transformers import AutoTokenizer

from model.LFESM.features import extract_features_and_replace


class LFESMFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        # self.tokenizer = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode

    def convert_tokens_to_ids(self, text):
        return self.tokenizer(text, padding='max_length', max_length=self.max_len, truncation=True)

    def process(self, data, config, mode, *args, **params):
        q_input_ids = []
        c_input_ids = []
        q_attention_mask = []
        c_attention_mask = []
        query_candidate_id = []
        label = []
        q_features = []
        c_features = []
        for x in data:
            text_q = x['query']['q']
            text_c = x['candidate']['ajjbqk']

            text_q, _q = extract_features_and_replace(text_q)
            text_c, _c = extract_features_and_replace(text_c)

            q_ = self.convert_tokens_to_ids(text_q)
            c_ = self.convert_tokens_to_ids(text_c)
            q_input_ids.append(q_.input_ids)
            c_input_ids.append(c_.input_ids)
            q_attention_mask.append(q_.attention_mask)
            c_attention_mask.append(c_.attention_mask)
            q_features.append(_q)
            c_features.append(_c)


            if(mode != 'test'):
                label.append(x['label'])
            query_candidate_id.append((x['query']['ridx'], x['candidate']['candidate_id']))

        q_input_ids = torch.LongTensor(q_input_ids)
        c_input_ids = torch.LongTensor(c_input_ids)
        q_attention_mask = torch.LongTensor(q_attention_mask)
        c_attention_mask = torch.LongTensor(c_attention_mask)
        q_features = torch.LongTensor(q_features)
        c_features = torch.LongTensor(c_features)
        label = torch.LongTensor(label)

        return {"q_input_ids": q_input_ids, 
                "c_input_ids": c_input_ids, 
                "q_attention_mask": q_attention_mask,
                "c_attention_mask": c_attention_mask,
                "query_candidate_id": query_candidate_id, 
                "label": label,
                "q_features": q_features,
                "c_featurs": c_features
                }
