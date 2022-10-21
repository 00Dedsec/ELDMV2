import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from utils.config_parser import create_config
from model.ELDM.teacher import BertCRF
from utils.logger import Logger
from utils.bio_lables import bio_labels
from model.ELDM.utils import masked_softmax, weighted_sum, sort_by_seq_lens, replace_masked

logger = Logger(__name__)

class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.
    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        assert issubclass(rnn_type, nn.RNNBase), "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        self._encoder.flatten_parameters()

        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths.cpu(),
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True,
                                                      total_length=512)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs

class ELDMMoudle(nn.Module):
    def __init__(self, config, gpu_list, teacher_model, teacher_config, *args, **params):
        super(ELDMMoudle, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.max_len = config.getint("data", "max_seq_length")
        self.hidden_size = config.getint("model", "hidden_size")

        self.sent_q = config.getint("data", "max_para_q")
        self.sent_c = config.getint("data", "max_para_c")

        #ELDM processing
        self.encoder = AutoModel.from_pretrained("bert-base-chinese")
        self.projection = nn.Sequential(nn.Linear(4 * config.getint("model", "hidden_size"), config.getint("model", "hidden_size")),
                            nn.ReLU())

        self.composition = Seq2SeqEncoder(nn.LSTM,
                                           config.getint("model", "hidden_size"),
                                           config.getint("model", "hidden_size"),
                                           bidirectional=True)
        
        # max_pool
        self.max_pool = nn.MaxPool2d(kernel_size=(self.sent_q, self.sent_c))

        # cls
       
        self.fc = nn.Sequential(nn.Dropout(p=config.getfloat("train", "dropout")),  # p=dropout
                                        nn.Linear(2 * 2 * 2 * config.getint("model", "hidden_size"), config.getint("model", "hidden_size")),
                                        nn.Tanh(),
                                        nn.Dropout(p=config.getfloat("train", "dropout")),  # p=dropout
                                        nn.Linear(config.getint("model", "hidden_size"), 4))
  
    def forward(self, data, gpu_list, *args, **params):
        input_ids_q = data['input_ids_q']
        attn_mask_q = data['attention_mask_q']
        token_type_ids_q = data['token_type_ids_q']
        input_ids_c = data['input_ids_c']
        attn_mask_c = data['attention_mask_c']
        token_type_ids_c = data['token_type_ids_c']

        q = self.encoder(input_ids_q.view(-1, self.max_len), attn_mask_q.view(-1, self.max_len)).last_hidden_state
        c = self.encoder(input_ids_c.view(-1, self.max_len), attn_mask_c.view(-1, self.max_len)).last_hidden_state

        q = q.view(input_ids_q.shape[0], input_ids_q.shape[1], input_ids_q.shape[2], -1)
        c = c.view(input_ids_c.shape[0], input_ids_c.shape[1], input_ids_c.shape[2], -1)
       
        # q [b, sent_q, max_len, dim]
        # c [b, sent_c, max_len, dim]
        # print('0. ')
        # print(q.shape)
        # print(c.shape)
        

        q_c = self.siamese(q, c, attn_mask_q, attn_mask_c)
        
        return q_c

    def siamese(self, q, c, attn_mask_q, attn_mask_c):
        # q [b, sent_q, max_len, dim]
        # c [b, sent_c, max_len, dim]
        # attn_mask_q [b, sent_q, max_len]
        # attn_mask_c [b, sent_c, max_len]

        # attention
        # t = sent_q * sent_c
        # q [b, t, max_len, dim]
        # attn_mask_q [b, t, max_len]
        batch = q.shape[0]
        sent_q = q.shape[1]
        sent_c = c.shape[1]

        q = torch.repeat_interleave(q, sent_c,dim=1)
        attn_mask_q = torch.repeat_interleave(attn_mask_q, sent_c, dim=1)
        c = torch.repeat_interleave(c, sent_q,dim=1)
        attn_mask_c = torch.repeat_interleave(attn_mask_c, sent_q, dim=1)
        # print("1. ")
        # print(q.shape)
        # print(attn_mask_q.shape)
        # print(c.shape)
        # print(attn_mask_c.shape)

        # q [b*t, max_len, dim]
        # c [b*t, max_len, dim]
        # attn_mask_q [b*t, max_len]
        # attn_mask_c [b*t, max_len]
        q = q.view(-1, q.shape[2], q.shape[3])
        c = c.view(-1, c.shape[2], c.shape[3])
        attn_mask_q = attn_mask_q.view(-1, attn_mask_q.shape[2])
        attn_mask_c = attn_mask_c.view(-1, attn_mask_c.shape[2])
        # print("2. ")
        # print(q.shape)
        # print(attn_mask_q.shape)
        q_length = attn_mask_q.sum(dim=-1).long()
        c_length = attn_mask_c.sum(dim=-1).long()
        # attn_q [b*t, max_len, dim]
        # attn_c [b*t, max_len, dim]
        attn_q, attn_c = self.attention(q, attn_mask_q, c, attn_mask_c)

        # print("3. ")
        # print(attn_q.shape)

        # enhanced_q [b*t, max_len, 4 * dim]
        # enhanced_c [b*t, max_len, 4 * dim]
        enhanced_q = torch.cat([q,
                                attn_q,
                                q - attn_q,
                                q * attn_q],
                               dim=-1)

        enhanced_c = torch.cat([c,
                                attn_q,
                                c - attn_c,
                                c * attn_c],
                               dim=-1)
        # projected_q [b*t, max_len, 768]
        # projected_c [b*t, max_len, 768]
        projected_q = self.projection(enhanced_q)
        projected_c = self.projection(enhanced_c)

        # print(projected_q.shape)
        # print(projected_c.shape)

        # LSTM
        # v_qi [batch*t, max_len, bi*hidden]
        # v_ci [batch*t, max_len, bi*hidden]
        v_qi = self.composition(projected_q, q_length)
        v_ci = self.composition(projected_c, c_length)
        # print(v_qi.shape)
        # print(v_ci.shape)

        # v_qi [batch*t, bi*hidden]
        # v_ci [batch*t, bi*hidden]
        v_q_avg = torch.sum(v_qi * attn_mask_q.unsqueeze(1)
                            .transpose(2, 1), dim=1) / torch.sum(attn_mask_q, dim=1, keepdim=True)
        v_c_avg = torch.sum(v_ci * attn_mask_c.unsqueeze(1)
                            .transpose(2, 1), dim=1) / torch.sum(attn_mask_c, dim=1, keepdim=True)
        # print("5. ")
        # print(v_q_avg.shape)
        # print(v_c_avg.shape)

        # v_q_max [batch*t, bi*hidden]
        # v_c_max [batch*t, bi*hidden]
        v_q_max, _ = replace_masked(v_qi, attn_mask_q, -1e7).max(dim=1)
        v_c_max, _ = replace_masked(v_ci, attn_mask_c, -1e7).max(dim=1)
        
        # decomposition
        # v_q_avg [batch, sent_q, sent_c, bi*hidden]
        # v_c_avg [batch, sent_c, sent_q, bi*hidden]
        # v_q_max [batch, sent_q, sent_q, bi*hidden]
        # v_c_max [batch, sent_c, sent_q, bi*hidden]
        v_q_avg = v_q_avg.view(batch, sent_q, sent_c, -1)
        v_c_avg = v_c_avg.view(batch, sent_c, sent_q, -1)
        v_q_max = v_q_max.view(batch, sent_q, sent_c, -1)
        v_c_max = v_c_max.view(batch, sent_c, sent_q, -1)

        # v_q [batch, sent_q, sent_c, 2*bi*hidden]
        # v_c [batch, sent_c, sent_q, 2*bi*hidden]
        v_q = torch.cat([v_q_avg, v_q_max], dim=-1)
        v_c = torch.cat([v_c_avg, v_c_max], dim=-1)

        # max_pool
        # q_pooling [b, 2*bi*hidden]
        # c_pooling [b, 2*bi*hidden]
        v_q = v_q.view(batch, 2*2*self.hidden_size, sent_q, sent_c)
        v_c = v_c.view(batch, 2*2*self.hidden_size, sent_q, sent_c)
        # 注意batch_size为1
        q_pooling = self.max_pool(v_q).squeeze(2).squeeze(2)
        c_pooling = self.max_pool(v_c).squeeze(2).squeeze(2)
        # print('6. ')
        # print(q_pooling.shape)
        # print(c_pooling.shape)
        re = torch.cat([q_pooling, c_pooling], dim = 1)

        re = self.fc(re)

        return re
    
    def attention(self,
            premise_batch,
            premise_mask,
            hypothesis_batch,
            hypothesis_mask):
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)

        # sqrt_dim = np.sqrt(premise_batch.size()[2])
        #
        # self_premises_matrix = premise_batch.bmm(premise_batch.transpose(2, 1).contiguous()) / sqrt_dim
        # self_hypotheses_matrix = hypothesis_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous()) / sqrt_dim
        #
        # self_premises_attn = normal_softmax(self_premises_matrix)
        # self_hypotheses_attn = normal_softmax(self_hypotheses_matrix)
        # self_premises = self_premises_attn.bmm(premise_batch)
        # self_hypotheses = self_hypotheses_attn.bmm(hypothesis_batch)

        return attended_premises, attended_hypotheses  # , self_premises, self_hypotheses


class ELDMV2SVD(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ELDMV2SVD, self).__init__()
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

