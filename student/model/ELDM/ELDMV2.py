import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class CNNMoudle(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(CNNMoudle, self).__init__()

    def forward(self):
        pass

class ELDMMoudle(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ELDMMoudle, self).__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-chinese")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.CNN = CNNMoudle(config, gpu_list, *args, **params)
        self.sep = '[SEP]'

    def forward(self, data, config, gpu_list, mode, *args, **params):
        """
        data  = {
            "query_input_ids": query_input_ids, # [batch_size, sent1, max_len]
            "query_token_type_ids": query_token_type_ids,[batch_size, sent1, max_len]
            "query_attention_mask": query_attention_mask,[batch_size, sent1, max_len]

            "candidate_input_ids": candidate_input_ids, [batch_size, sent2, max_len]
            "candidate_token_type_ids": candidate_token_type_ids,[batch_size, sent2, max_len]
            "candidate_attention_mask": candidate_attention_mask,[batch_size, sent2, max_len]

            "query_candidate_id": query_candidate_id,[batch_size, (query_id, candidate_id)]

        }
        """
        query_input_ids = data['query_input_ids']
        query_token_type_ids = data['query_token_type_ids']
        query_attention_mask = data['query_attention_mask']
        candidate_input_ids = data['candidate_input_ids']
        candidate_token_type_ids = data['candidate_token_type_ids']
        candidate_attention_mask = data['candidate_attention_mask']

        
        query_candidate_sent_input_ids_matrix = []
        query_candidate_sent_attention_mask_matrix = []
        # sent1 与 sent2 排列组合, 得到[batch_size, sent1, sent2, 2 * max_len]
        for batch_index in query_input_ids.shape[0]:
            sent2sent_input_ids_matrix = []
            sent2sent_attention_mask_matrix = []
            for sent1_index in query_input_ids.shape[1]:
                row_input_ids_matrix = []
                row_attention_mask_matrix = []
                for sent2_index in candidate_input_ids.shape[1]:
                    input_ids = query_input_ids[batch_index][sent1_index].append(self.tokenizer.convert_tokens_to_ids(self.sep))
                    input_ids.extend(candidate_input_ids[batch_index][sent2_index])
                    row_input_ids_matrix.append(input_ids)

                    attention_mask = query_attention_mask[batch_index][sent1_index].append(1)
                    attention_mask.extend(candidate_attention_mask[batch_index][sent2_index])
                    row_attention_mask_matrix.append(attention_mask)
                sent2sent_input_ids_matrix.append(row_input_ids_matrix)
                sent2sent_attention_mask_matrix.append(row_attention_mask_matrix)
            query_candidate_sent_input_ids_matrix.append(sent2sent_input_ids_matrix) #[batch_size, sent1. sent2, 2 * max_len + 'SEP']
            query_candidate_sent_attention_mask_matrix.append(sent2sent_attention_mask_matrix)
            
        plm_input_shape = (query_input_ids.shape[0], -1, 2 * config.getint("data", "max_seq_length") + 1)
        query_candidate_sent_input_ids_matrix = query_candidate_sent_input_ids_matrix.view(plm_input_shape) # [batch_size, sent1 * sent2, max_len]
        pooler_out = self.encoder(input_ids = query_candidate_sent_input_ids_matrix, attention_mask = query_candidate_sent_attention_mask_matrix).pooler_out # [batch_size, sent1 * sent2, emb]
        # [batch_size, sent1, sent2, emb]
        pooler_out = pooler_out.view(query_input_ids.shape[0], query_input_ids.shape[1], candidate_input_ids.shape[1], config.getint("model", "hidden_size"))
        print(pooler_out.shape)
        exit()
        # CNN



class ELDMV2(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ELDMV2, self).__init__()
        self.ELDMMoule = ELDMMoudle(config, gpu_list, *args, **params)
        self.loss = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config,  *args, **params):
        self.ELDMMoule = nn.DataParallel(self.ELDMMoule, device_ids=device)

    def forward(self, data, config, gpu_listm, mode, *args, **params):
        re = self.ELDMMoule(data)
        if mode == 'train' or mode == 'valid':
            label = data['label']
            loss = self.loss(re, data)
            return re, loss
        else:
            return re

