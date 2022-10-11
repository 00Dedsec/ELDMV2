import json
from lib2to3.pgen2 import token
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random


class SentFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))
        # self.max_len = config.getint("data", "sent_len")
        # self.max_sent = config.getint("data", "max_sent")
        self.mode = mode
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        
    def convert_tokens_to_ids(self, text, type):
        arr = [[]]
        if type == 'q':
            self.max_sent = 15
            self.max_len = 30
        else:
            self.max_sent =30
            self.max_len = 30
        
        for a in range(0, len(text)):
            if text[a] in ['，',"。","；","、"]:
                arr.append([])
                continue
            if text[a] in self.tokenizer.keys():
                arr[-1].append(self.tokenizer[text[a]])
            else:
                arr[-1].append(self.tokenizer[self.unk])

        #########paras#########
        paras = [[]]
        for a in range(0, len(arr)):
            if (len(paras[-1]) + len(arr[a]) < self.max_len ):
                paras[-1].extend(arr[a])
            elif(len(arr[a]) < self.max_len ):
                paras.append(arr[a])
        
        while len(paras) < self.max_sent:
            paras.append([])
        ###################
 
        
        paras = paras[:self.max_sent]
        for a in range(0, len(paras)):
            while len(paras[a]) < self.max_len:
                paras[a].append(self.tokenizer[self.pad])
            paras[a] = paras[a][:self.max_len]
        return paras

    def process(self, data, config, mode, *args, **params):
        q = []
        c = []
        query_candidate_id = []
        label = []
        for x in data:
            text_q = x['query']['q']
            text_c = x['candidate']['ajjbqk']

            token_q = self.convert_tokens_to_ids(text_q, 'q') # [sent, max]
            token_c = self.convert_tokens_to_ids(text_c, 'c')

            q.append(token_q)
            c.append(token_c)



            if(mode != 'test'):
                label.append(x['label'])
            query_candidate_id.append((x['query']['ridx'], x['candidate']['candidate_id']))

        q = torch.LongTensor(q)
        c = torch.LongTensor(c)
        label = torch.LongTensor(label)

        return {"q": q, "c": c, "query_candidate_id": query_candidate_id, "label": label}