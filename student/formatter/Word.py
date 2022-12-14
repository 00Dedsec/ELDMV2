import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter


class WordFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode

    def convert_tokens_to_ids(self, text):
        arr = []
        for a in range(0, len(text)):
            if text[a] in self.tokenizer.keys():
                arr.append(self.tokenizer[text[a]])
            else:
                arr.append(self.tokenizer["[UNK]"])
        return arr

    def process(self, data, config, mode, *args, **params):
        q = []
        c = []
        query_candidate_id = []
        label = []
        for x in data:
            text_q = x['query']['q']
            text_c = x['candidate']['ajjbqk']

            while len(text_q) < self.max_len:
                text_q.append("[PAD]")
            text_q = text_q[0:self.max_len]
            while len(text_c) < self.max_len:
                text_c.append("[PAD]")
            text_c = text_c[0:self.max_len]

            q.append(self.convert_tokens_to_ids(text_q))
            c.append(self.convert_tokens_to_ids(text_c))

            if(mode != 'test'):
                label.append(x['label'])
            query_candidate_id.append((x['query']['ridx'], x['candidate']['candidate_id']))

        q = torch.LongTensor(q)
        c = torch.LongTensor(c)
        label = torch.LongTensor(label)

        return {"q": q, "c": c, "query_candidate_id": query_candidate_id, "label": label}
