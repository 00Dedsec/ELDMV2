import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from tools.accuracy_tool import single_label_top1_accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Attention, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x, y):
        x_ = x  # self.fc(x)
        y_ = torch.transpose(y, 1, 2)
        a_ = torch.bmm(x_, y_)

        x_atten = torch.softmax(a_, dim=2)
        x_atten = torch.bmm(x_atten, y)

        y_atten = torch.softmax(a_, dim=1)
        y_atten = torch.bmm(torch.transpose(y_atten, 2, 1), x)

        return x_atten, y_atten, a_

class LSTMEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LSTMEncoder, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.bi = config.getboolean("model", "bi_direction")
        self.output_size = self.hidden_size
        self.num_layers = config.getint("model", "num_layers")
        if self.bi:
            self.output_size = self.output_size // 2

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.output_size,
                            num_layers=self.num_layers, batch_first=True, bidirectional=self.bi)

    def forward(self, x):
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        # print(x.size())
        # print(batch_size, self.num_layers + int(self.bi) * self.num_layers, self.output_size)
        hidden = (
            torch.autograd.Variable(
                torch.zeros(self.num_layers + int(self.bi) * self.num_layers, batch_size, self.output_size)).cuda(),
            torch.autograd.Variable(
                torch.zeros(self.num_layers + int(self.bi) * self.num_layers, batch_size, self.output_size)).cuda())

        h, c = self.lstm(x, hidden)

        h_ = torch.max(h, dim=1)[0]

        return h_, h

class DEC(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(DEC, self).__init__()

        self.sentence_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.document_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.sentence_attention = Attention(config, gpu_list, *args, **params)
        self.document_attention = Attention(config, gpu_list, *args, **params)

    def forward(self, x):
        batch = x.size()[0]
        sent = x.size()[1]
        word = x.size()[2]
        hidden = x.size()[3]

        _, h1 = self.sentence_encoder(x.view(batch * sent, word, hidden))

        x, y, a = self.sentence_attention(h1, h1)

        x = x.view(batch, sent, word, hidden)
        x = torch.max(x, dim=2)[0]

        _, h2 = self.document_encoder(x)

        _, y, a = self.document_attention(h2, h2)
        y = torch.max(h2, dim=1)[0]

        return torch.cat([torch.max(x, dim=1)[0], y], dim=1)


class SMASHMoudle(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(SMASHMoudle, self).__init__()

        self.encoder = DEC(config, gpu_list, *args, **params)
        self.fc = nn.Linear(config.getint("model", "hidden_size") * 4, 4)
        self.embedding = nn.Embedding(len(json.load(open(config.get("data", "word2id")))),
                                      config.getint("model", "hidden_size"))

    def forward(self, data):
        q = data['q']
        c = data['c']
        q = self.embedding(q)
        c = self.embedding(c)
        q = self.encoder(q)
        c = self.encoder(c)

        s = self.fc(torch.cat([q, c], dim=1))

        return s

class SMASHRNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(SMASHRNN, self).__init__()
        self.SMASHMoudle = SMASHMoudle(self, config, gpu_list, *args, **params)
        self.loss = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config, *args, **params):
        self.SMASHMoudle = nn.DataParallel(self.SMASHMoudle, device_ids=device)
    
    def forward(self, data, config, gpu_list, mode, *args, **params):
        re = self.SMASHMoudle(data)
        if mode == 'train' or mode == 'valid':
            label = data['label']
            loss = self.loss(re, label)
            return re, loss
        else:
            return re
    
