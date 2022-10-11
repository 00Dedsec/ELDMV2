import torch
import torch.nn as nn
import torch.nn.functional as F
import json
# from model.loss import cross_entropy_loss
from tools.accuracy_tool import single_label_top1_accuracy

class CNNEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(CNNEncoder, self).__init__()

        self.emb_dim = config.getint("model", "hidden_Size")
        self.output_dim = self.emb_dim // 4

        self.min_gram = 2
        self.max_gram = 5
        self.convs = []
        for a in range(self.min_gram, self.max_gram + 1):
            self.convs.append(nn.Conv2d(1, self.output_dim, (a, self.emb_dim)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = self.emb_dim
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]

        x = x.view(batch_size, 1, -1, self.emb_dim)

        conv_out = []
        gram = self.min_gram
        for conv in self.convs:
            y = self.relu(conv(x))
            y = torch.max(y, dim=2)[0].view(batch_size, -1)

            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        return conv_out

class TextCNNModule(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(TextCNNModule, self).__init__()
        self.embedding = nn.Embedding(len(json.load(open(config.get("data", "word2id"), encoding="utf-8"))),
                                      config.getint("model", "hidden_size"))
        self.encoder = CNNEncoder(config, gpu_list, *args, **params)
        self.fc = nn.Bilinear(config.getint("model", "hidden_size"), config.getint("model", "hidden_size"), 4)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_function = single_label_top1_accuracy

    def forward(self, data):
        q = data['q']
        c = data['c']
        # text = torch.cat([q,c], dim=1)
        # x = self.embedding(text)
        # x = self.encoder(x)
        q = self.embedding(q)
        c = self.embedding(c)
        q = self.encoder(q)
        c = self.encoder(c) # b * h
        
        s = self.fc(q, c)

        return s

class TextCNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(TextCNN, self).__init__()
        self.SMASHMoudle = TextCNNModule(config, gpu_list, *args, **params)
        self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.MSELoss()

    def init_multi_gpu(self, device, config, *args, **params):
        self.SMASHMoudle = nn.DataParallel(self.SMASHMoudle, device_ids=device)
    
    def forward(self, data, config, gpu_list, mode, *args, **params):
        re = self.SMASHMoudle(data)
        if mode == 'train' or mode == 'valid':
            label = data['label']
            # label = label.to(torch.float)
            loss = self.loss(re, label)
            return re, loss
        else:
            return re
    