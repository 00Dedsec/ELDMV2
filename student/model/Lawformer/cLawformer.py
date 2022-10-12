import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class cLawformerMoudle(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(cLawformerMoudle, self).__init__()
        self.encoder = AutoModel.from_pretrained("thunlp/Lawformer")
        self.fc = nn.Linear(config.getint("model","hidden_size"), 4)

    
    def forward(self, data):
        input_ids = data['input_ids']
        attn_mask = data['attention_mask']
        
        pooler_output = self.encoder(input_ids=input_ids, attention_mask=attn_mask).pooler_output
        re = self.fc(pooler_output)
        return re

class cLawformer(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(cLawformer, self).__init__()
        self.cLawformerMoudle = cLawformerMoudle(config, gpu_list, *args, **params)
        self.loss = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config, *args, **params):
        self.cLawformerMoudle = nn.DataParallel(self.cLawformerMoudle, device_ids=device)
    
    def forward(self, data, config, gpu_list, mode, *args, **params):
        re = self.cLawformerMoudle(data)
        if mode == 'train' or mode == 'valid':
            label = data['label']
            # label = label.to(torch.float)
            loss = self.loss(re, label)
            return re, loss
        else:
            return re
    