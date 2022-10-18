import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class sLawformerMoudle(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(sLawformerMoudle, self).__init__()
        self.encoder = AutoModel.from_pretrained("thunlp/Lawformer")
        self.fc = nn.Linear(config.getint("data","max_seq_length_q"), 4)

    
    def forward(self, data):
        q_input_ids = data['q_input_ids']
        c_input_ids = data['c_input_ids']
        q_attention_mask = data['q_attention_mask']
        c_attention_mask = data['c_attention_mask']
        
        q = self.encoder(input_ids=q_input_ids, attention_mask=q_attention_mask)
        c = self.encoder(input_ids=c_input_ids, attention_mask=c_attention_mask)
        q_last_hidden_state = q.last_hidden_state
        c_last_hidden_state = c.last_hidden_state

        q_t = torch.transpose(q_last_hidden_state, 1, 2)
        re = torch.bmm(c_last_hidden_state, q_t) # [batch_size, max_len, max_len]
        re = torch.softmax(re, dim=2)
        re = torch.max(re, dim=1)[0] #[batch, max_len]

        re = self.fc(re)
        return re

class sLawformer(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(sLawformer, self).__init__()
        self.sLawformerMoudle = sLawformerMoudle(config, gpu_list, *args, **params)
        self.loss = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config, *args, **params):
        self.sLawformerMoudle = nn.DataParallel(self.sLawformerMoudle, device_ids=device)
    
    def forward(self, data, config, gpu_list, mode, *args, **params):
        re = self.sLawformerMoudle(data)
        if mode == 'train' or mode == 'valid':
            label = data['label']
            # label = label.to(torch.float)
            loss = self.loss(re, label)
            return re, loss
        else:
            return re
    