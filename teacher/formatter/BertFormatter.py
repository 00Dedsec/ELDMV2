from cProfile import label
import json
import torch
import os
import numpy as np

from transformers import AutoTokenizer

from formatter.Basic import BasicFormatter
from utils.bio_lables import get_labels

class BertFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        # self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.label_list = get_labels()
        self.pad_token_label_id = -100


    def process(self, data, config, mode, *args, **params):
        input_ids = []
        input_mask = []
        segment_ids = []
        if(mode != 'test'):
            labels_ids = []
            label_map = {label: i for i, label in enumerate(self.label_list)}
        
        for example in data:
            tokens = []
            label_item = []

            for word, label in zip(example['words'], example['labels']):
                word_tokens = self.tokenizer.tokenize(word)
                if len(word_tokens) == 0:
                    word_tokens = ['<UNK>']
                tokens.extend(word_tokens)
                if(mode != 'test'):
                    label_item.extend([label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1))
            
            special_tokens_count = 2
            if len(tokens) > self.max_len - special_tokens_count:
                tokens = tokens[:(self.max_len - special_tokens_count)]
                if(mode != 'test'):
                    label_item = label_item[:(self.max_len - special_tokens_count)]

            tokens = self.tokenizer.tokenize('[CLS]') + tokens + self.tokenizer.tokenize('[SEP]')
            if(mode != 'test'):
                label_item = [self.pad_token_label_id] + label_item + [self.pad_token_label_id]

            input_ids_item = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask_item = [1] * len(input_ids_item)
            segment_ids_item = [1] * len(input_ids_item)

            padding_length = self.max_len - len(input_ids_item)

            if padding_length >0:

                input_ids_item += (self.tokenizer.convert_tokens_to_ids(['[PAD]'])* padding_length)
                input_mask_item += ([0] * padding_length)
                segment_ids_item += ([0] * padding_length)

                if(mode != 'test'):
                    label_item += ([self.pad_token_label_id] * padding_length)
            else:
                input_ids_item = input_ids_item[:self.max_len]
                input_mask_item = input_mask_item[:self.max_len]
                segment_ids_item = segment_ids_item[:self.max_len]

                if(mode != 'test'):
                    label_item = label_item[:self.max_len]

            assert len(input_ids_item) == self.max_len
            assert len(input_mask_item) == self.max_len
            assert len(segment_ids_item) == self.max_len

            if(mode != 'test'):
                assert len(label_item) == self.max_len
            
            input_ids.append(input_ids_item)
            input_mask.append(input_mask_item)
            segment_ids.append(segment_ids_item)
            if(mode != 'test'):
                labels_ids.append(label_item)
        
        input_ids = torch.LongTensor(input_ids)
        input_mask = torch.LongTensor(input_mask)
        segment_ids = torch.LongTensor(segment_ids)

        if(mode != 'test'):
            labels_ids = torch.LongTensor(labels_ids)
            return {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'labels': labels_ids 
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
            }
            
            



