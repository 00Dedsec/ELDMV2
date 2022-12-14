import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        pass

    def forward(self, feature, hidden):
        # hidden: B * seq len * H, feature: B * H * 1
        ratio = torch.bmm(hidden, feature)  # B, seq len, 1   各个位置打分
        ratio = ratio.view(ratio.size(0), ratio.size(1))  # b, seq len
        ratio = F.softmax(ratio, dim=1).unsqueeze(2)  # b, seq len, 1 分数比例
        result = torch.bmm(hidden.permute(0, 2, 1), ratio)  # [b,  h, seq len] * [b, seq len, 1] = b, h, 1
        result = result.view(result.size(0), -1)  # b, h
        return result

class RNNAttention(nn.Module):
    def __init__(self, max_para_d):
        super(RNNAttention, self).__init__()
        self.input_dim = 768
        self.hidden_dim = 256
        self.dropout_rnn = 0
        self.dropout_fc = 0
        self.direction = 1
        self.num_layers = 1
        self.output_dim = 4
        self.max_para_d = max_para_d

        self.rnn = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, batch_first=True,
                           num_layers=self.num_layers, bidirectional=False, dropout=self.dropout_rnn)

        self.max_pool = nn.MaxPool1d(kernel_size=self.max_para_d)

        self.fc_a = nn.Linear(self.hidden_dim*self.direction, self.hidden_dim*self.direction)

        self.attention = Attention()
        self.fc_f = nn.Linear(self.hidden_dim*self.direction, self.output_dim)
        self.soft_max = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.dropout_fc)

    def forward(self, hidden_seq):
        # hidden_seq: B , seq_len , h
        batch_size = hidden_seq.size()[0]
        self.hidden = (
            torch.autograd.Variable(torch.zeros((self.direction * self.num_layers, batch_size, self.hidden_dim)).cuda()),  # 1, b, h
            torch.autograd.Variable(torch.zeros((self.direction * self.num_layers, batch_size, self.hidden_dim)).cuda())   # 1, b, h
        )# 2 * B * H

        # Inputs: input, (h_0, c_0)     Outputs: output, (h_n, c_n)
        self.rnn.flatten_parameters()
        rnn_out, self.hidden = self.rnn(hidden_seq, self.hidden) # rnn_out: B * seq len * H,  hidden: 2 * B * H

        tmp_rnn = rnn_out.permute(0, 2, 1) # B * H * seq len

        feature = self.max_pool(tmp_rnn)  # B * H * 1
        feature = feature.squeeze(2)  # B * H
        feature = self.fc_a(feature)  # B * H
        feature = feature.unsqueeze(2)  # B * H * 1

        atten_out = self.attention(feature, rnn_out)  # B * H
        atten_out = self.dropout(atten_out)
        y = self.fc_f(atten_out)  # B, 2
        y = self.soft_max(y)  # b, 2
        y = y.view(y.size()[0], -1)  # b, 2

        return y

class PLIMoudle(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(PLIMoudle, self).__init__()
        self.max_para_q = config.getint("data", "max_para_q")
        self.max_para_c = config.getint("data", "max_para_c")
        self.max_len = config.getint("data", "max_seq_length")

        self.encoder = AutoModel.from_pretrained("bert-base-chinese")
        self.maxpool = nn.MaxPool2d(kernel_size=(1, self.max_para_q))

        self.attn = RNNAttention(max_para_d = self.max_para_c)

    def forward(self, data):
        input_ids_item = data['input_ids']
        attention_mask_item = data['attention_mask']
        token_type_ids_item = data['token_type_ids']

        # cls : [b * max_para_q * max_para_d, h]
        # last_hidden_state : [b * max_para_q * max_para_d, max len, h]
        re = self.encoder(input_ids=input_ids_item.view(-1, self.max_len),  # [b * max_para_q * max_para_d, max len]
                                            attention_mask=attention_mask_item.view(-1, self.max_len),  # [b * max_para_q * max_para_d, max len]
                                            token_type_ids=token_type_ids_item.view(-1, self.max_len))  # [b * max_para_q * max_para_d, max len]
        
        last_hidden_state, cls = re.last_hidden_state, re.pooler_output
        pooling = 'cls'
        if pooling == 'cls':
            feature = cls
        else:
            feature = torch.mean(last_hidden_state, dim=1)

        feature = feature.view(self.max_para_q, self.max_para_c, -1)  # [max_para_q, max_para_d, h]

        feature = feature.permute(2, 1, 0)  # [h, max_para_d, max_para_q]

        feature = feature.unsqueeze(0)  # [1, h, max_para_d, max_para_q]
        max_out = self.maxpool(feature)  # [1, h, max_para_d, 1]
        max_out = max_out.squeeze()  # [h, max_para_d]
        max_out = max_out.transpose(0, 1)  # [max_para_d, h]
        max_out = max_out.unsqueeze(0)
        # print(max_out.shape)

        score = self.attn(max_out)  # b,2

        return score
        

class PLI(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(PLI, self).__init__()
        self.PLIMoudle = PLIMoudle(config, gpu_list, *args, **params)
        self.loss = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config, *args, **params):
        if config.getboolean("distributed", "use"):
            self.PLIMoudle = nn.parallel.DistributedDataParallel(self.PLIMoudle, device_ids=device, find_unused_parameters=True)
        else:
            self.PLIMoudle = nn.DataParallel(self.PLIMoudle, device_ids=device)
    
    def forward(self, data, config, gpu_list, mode, *args, **params):
        re = self.PLIMoudle(data)
        if mode == 'train' or mode == 'valid':
            label = data['label']
            # label = label.to(torch.float)
            loss = self.loss(re, label)
            return re, loss
        else:
            return re
    