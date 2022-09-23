from cProfile import label
import json
import torch
import os
import numpy as np

from transformers import AutoTokenizer

from formatter.Basic import BasicFormatter
from utils.bio_lables import bio_labels

class BertSCM(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        # self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model","bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        tokens_batch = []
        labels_batch = []
        attention_mask_batch = []
        if(mode == 'train' or mode == 'valid'):
            for doc in data:
                tokens = []
                labels = []
                # 按照分词将标签准备好，每个标签对应一个词
                words = [c['tokens'] for c in doc['content']] # [句子 * 单词]
                tags = [['O']*len(c['tokens']) for c in doc['content']] # [句子 * 单词]             
                for item in doc['events']:
                    for mention in item['mention']: 
                        # 设置第sent_id的第offset个词为B
                        tags[mention['sent_id']][mention['offset'][0]] = 'B-' + item['type']
                        pass
                for i in range(len(words)):
                    for word, tag in zip(words[i], tags[i]):
                        word2tokens = self.tokenizer.tokenize(word)
                        tokens.extend(word2tokens)
                        if(tag[0] == 'B'):
                            labels.extend([tag] + ['I-'+tag[2:]]*(len(word2tokens)-1))
                        else:
                            labels.extend(['O']*len(word2tokens))

                labels = [bio_labels.index(item) for item in labels]
                # 合并完成后进行tokenize
                
                tokens.insert(0, '[CLS]')
                labels.insert(0, 0)
                tokens.append('[SEP]')
                labels.append(0)
                attention_mask = [1]*len(tokens)
                while len(tokens) < self.max_len:
                    tokens.append('[PAD]')
                    attention_mask.append(0)
                    labels.append(len(bio_labels) + 1)
                tokens = tokens[0:self.max_len]
                attention_mask = attention_mask[0:self.max_len]
                labels = labels[0: self.max_len]
                tokens = self.tokenizer.convert_tokens_to_ids(tokens)
                tokens_batch.append(tokens)
                attention_mask_batch.append(attention_mask)
                labels_batch.append(labels)
                
        tokens_batch = torch.LongTensor(tokens_batch)
        attention_mask_batch = torch.LongTensor(attention_mask_batch)
        labels_batch = torch.LongTensor(labels_batch)
        return {
            "tokens": tokens_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch
        }