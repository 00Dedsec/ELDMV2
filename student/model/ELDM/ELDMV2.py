import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from utils.config_parser import create_config
from model.ELDM.teacher import BertCRF
from utils.logger import Logger
from utils.bio_lables import bio_labels

logger = Logger(__name__)
class ELDMMoudle(nn.Module):
    def __init__(self, config, gpu_list, teacher_model, teacher_config, *args, **params):
        super(ELDMMoudle, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.max_len = config.getint("data", "max_seq_length")
        self.hidden_size = config.getint("model", "hidden_size")
        self.teacher_model = teacher_model
        self.teacher_config = teacher_config
        self.num_labels = len(bio_labels)
        # teacher processing
        self.fc_t = nn.Linear(self.max_len, self.hidden_size)

        #ELDM processing
        self.encoder = AutoModel.from_pretrained("bert-base-chinese")
        self.num_layers=2
        self.lstm = nn.LSTM(input_size=config.getint("model", "hidden_size"), 
                            hidden_size = config.getint("model", "hidden_size"), 
                            num_layers = self.num_layers, 
                            batch_first = True,
                            bidirectional = True
                            )
        self.attn = nn.MultiheadAttention(embed_dim=self.num_layers * config.getint("model", "hidden_size"), num_heads = 1)

        self.fc = nn.Linear(self.num_layers * config.getint("model", "hidden_size"), 4)

    def forward(self, data, gpu_list, *args, **params):
        input_ids_q = data['input_ids_q']
        attn_mask_q = data['attention_mask_q']
        token_type_ids_q = data['token_type_ids_q']
        input_ids_c = data['input_ids_c']
        attn_mask_c = data['attention_mask_c']
        token_type_ids_c = data['token_type_ids_c']
        
        # #teacher_model processing
        # with torch.no_grad():
        #     #[b*sent, max, num_labels + 2]
        #     event_q = self.teacher_model({'input_ids': input_ids_q.view(-1, self.max_len), 'attention_mask': attn_mask_q.view(-1, self.max_len)}, self.teacher_config, gpu_list, 'test')
        #     event_c = self.teacher_model({'input_ids': input_ids_c.view(-1, self.max_len), 'attention_mask': attn_mask_c.view(-1, self.max_len)}, self.teacher_config, gpu_list, 'test')
        #     event_q = torch.max(event_q, dim=2)[1] #[b*sent, max]
        #     event_c = torch.max(event_c, dim=2)[1]
        #     event_q = torch.Tensor(event_q).cuda()
        #     event_c = torch.Tensor(event_q).cuda()
        #     event_q = event_q.view(input_ids_q.shape[0], input_ids_q.shape[1], -1) #[b, sent, max]
        #     event_c = event_c.view(input_ids_q.shape[0], input_ids_q.shape[1], -1)
        #     # torch.set_printoptions(profile="full")
        #     # print(event_q)
        #     # print(event_c)
        #     # torch.set_printoptions(profile="default")
        #     # exit()
        # event_q = self.fc_t(event_q) #[b,sent,h]
        # event_c = self.fc_t(event_c)
        # event_q = event_q.unsqueeze(2) #[b,sent,1,h]
        # event_c = event_c.unsqueeze(2)

        # ELDM proccessing
        # q: [batch_size, sent, max_len, emb]
        q = self.encoder(input_ids_q.view(-1, self.max_len), attn_mask_q.view(-1, self.max_len)).last_hidden_state.view(input_ids_q.shape[0], input_ids_q.shape[1], input_ids_q.shape[2], -1)
        c = self.encoder(input_ids_c.view(-1, self.max_len), attn_mask_c.view(-1, self.max_len)).last_hidden_state.view(input_ids_c.shape[0], input_ids_c.shape[1], input_ids_c.shape[2], -1)

        # #teacher_model processing       
        # q = torch.cat([event_q, q], dim = 2)
        # c = torch.cat([event_c, c], dim = 2)

        q = torch.max(q, dim=2)[0] #[batch_size, sent, emb]
        c = torch.max(c, dim=2)[0]

        self.lstm.flatten_parameters()
        q, _ = self.lstm(q) #[batch_size, sent, numlayers * emb]
        c, _ = self.lstm(c) #[batch_size, sent, numlayers * emb]

        attn_output, _ = self.attn(q.permute(1,0,2), c.permute(1,0,2), c.permute(1,0,2))
        attn_output = attn_output.permute(1,0,2) #[batch_size, sent, numlayers * emb]
        attn_output = torch.max(attn_output, dim=1)[0]

        re = self.fc(attn_output)

        return re

class ELDMV2(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ELDMV2, self).__init__()
        # teacher_model, teacher_config = self.load_teacher(config, gpu_list, *args, **params)
        teacher_model, teacher_config = None, None
        self.ELDMMoule = ELDMMoudle(config, gpu_list, teacher_model, teacher_config, *args, **params)
        self.loss = nn.CrossEntropyLoss()
        
    def load_teacher(self, config, gpu_list, *args, **params):
        teacher_config = create_config(config.get("model", "teacher_config"))
        model_t = BertCRF(teacher_config, gpu_list, *args, **params)

        if len(gpu_list) > 0:
            model_t = model_t.to("cuda")
            model_t.init_multi_gpu(gpu_list, config, *args, **params)
        logger.get_log().info("load teacher...")
        parameters_teacher = torch.load(config.get("model", "teacher_path"))
        model_t.load_state_dict(parameters_teacher["model"])
        return model_t, teacher_config

    def init_multi_gpu(self, device, config,  *args, **params):
        if config.getboolean("distributed", "use"):
            self.ELDMMoule = nn.parallel.DistributedDataParallel(self.ELDMMoule, device_ids=device, find_unused_parameters=True)
        else:
            self.ELDMMoule = nn.DataParallel(self.ELDMMoule, device_ids=device)

    def forward(self, data, config, gpu_list, mode, *args, **params):
        re = self.ELDMMoule(data, gpu_list)
        if mode == 'train' or mode == 'valid':
            label = data['label']
            loss = self.loss(re, label)
            return re, loss
        else:
            return re

