import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class ELDMMoudle(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ELDMMoudle, self).__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-chinese")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.max_len = config.getint("data", "max_seq_length")
        self.num_layers=2
        self.lstm = nn.LSTM(input_size=config.getint("model", "hidden_size"), 
                            hidden_size = config.getint("model", "hidden_size"), 
                            num_layers = self.num_layers, 
                            batch_first = True,
                            bidirectional = True
                            )
        self.attn = nn.MultiheadAttention(embed_dim=self.num_layers * config.getint("model", "hidden_size"), num_heads = 1)

        self.fc = nn.Linear(self.num_layers * config.getint("model", "hidden_size"), 4)

    def forward(self, data, *args, **params):
        input_ids_q = data['input_ids_q']
        attn_mask_q = data['attention_mask_q']
        token_type_ids_q = data['token_type_ids_q']
        input_ids_c = data['input_ids_c']
        attn_mask_c = data['attention_mask_c']
        token_type_ids_c = data['token_type_ids_c']

        input_ids_q.view(-1, self.max_len)
        attn_mask_q.view(-1, self.max_len)
        token_type_ids_q.view(-1, self.max_len)
        input_ids_c.view(-1, self.max_len)
        attn_mask_c.view(-1, self.max_len)
        token_type_ids_c.view(-1, self.max_len)


        # [batch_size, sent, max_len, emb]
        q = self.encoder(input_ids_q, attn_mask_q, token_type_ids_q).last_hidden_state.view(input_ids_q.shape[0], input_ids_q.shape[1], -1)
        c = self.encoder(input_ids_c, attn_mask_c, token_type_ids_c).last_hidden_state.view(input_ids_q.shape[0], input_ids_q.shape[1], -1)

        q = torch.max(q, dim=2)[0] #[batch_size, sent, emb]
        c = torch.max(c, dim=2)[0]

        self.lstm.flatten_parameters()
        q, _ = self.lstm(q) #[batch_size, sent, numlayers * emb]
        c, _ = self.lstm(c) #[batch_size, sent, numlayers * emb]

        attn_output, _ = self.attn(q.transpose(0, 1), c.transpose(0,1)).transpose(0,1) #[batch_size, sent, numlayers * emb]
        
        attn_output = torch.max(attn_output, dim=1)[0]

        re = self.fc(attn_output)

        return re

class ELDMV2(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ELDMV2, self).__init__()
        self.ELDMMoule = ELDMMoudle(config, gpu_list, *args, **params)
        self.loss = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config,  *args, **params):
        self.ELDMMoule = nn.DataParallel(self.ELDMMoule, device_ids=device)

    def forward(self, data, config, gpu_list, mode, *args, **params):
        re = self.ELDMMoule(data)
        if mode == 'train' or mode == 'valid':
            label = data['label']
            loss = self.loss(re, label)
            return re, loss
        else:
            return re

