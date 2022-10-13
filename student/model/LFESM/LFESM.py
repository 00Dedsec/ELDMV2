import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from model.LFESM.utils import masked_softmax, weighted_sum, sort_by_seq_lens, replace_masked

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
                                                      total_length=513)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs

class SoftmaxAttention(nn.Module):
    def forward(self,
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



class LFESMMoudle(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LFESMMoudle, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-chinese")
        feature_size = 28
        self._feature = nn.Linear(feature_size, config.getint("model", "hidden_size"))
        self._attention = SoftmaxAttention()
        self._projection = nn.Sequential(nn.Linear(4 * config.getint("model", "hidden_size"), config.getint("model", "hidden_size")),
                                    nn.ReLU())
        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           config.getint("model", "hidden_size"),
                                           config.getint("model", "hidden_size"),
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=config.getfloat("train", "dropout")),  # p=dropout
                                        nn.Linear(4 * 2 * config.getint("model", "hidden_size"), config.getint("model", "hidden_size")),
                                        nn.Tanh(),
                                        nn.Dropout(p=config.getfloat("train", "dropout")),  # p=dropout
                                        nn.Linear(config.getint("model", "hidden_size"), 4))
    
    def forward(self, data):
        q_input_ids = data['q_input_ids']
        c_input_ids = data['c_input_ids']
        q_attention_mask = data['q_attention_mask']
        c_attention_mask = data['c_attention_mask']
        q_features = data['q_features']
        c_features = data['c_features']

        q_output = self.bert(input_ids=q_input_ids, attention_mask=q_attention_mask).last_hidden_state
        c_output = self.bert(input_ids=c_input_ids, attention_mask=c_attention_mask).last_hidden_state

        #add features       
        q_feature = self._feature(q_features.unsqueeze(1))
        c_feature = self._feature(c_features.unsqueeze(1))
        q_extend = torch.cat([q_feature, q_output], dim=1)
        c_extend = torch.cat([c_feature, c_output], dim=1)
    
        q_mask = torch.cat([torch.tensor([[1]]*len(q_input_ids)).cuda(), q_attention_mask], dim=1)
        c_mask = torch.cat([torch.tensor([[1]]*len(c_input_ids)).cuda(), c_attention_mask], dim=1)

        v_qc = self.siamese(q_extend, c_extend, q_mask, c_mask)

        output = self._classification(v_qc) 
        return output

    def siamese(self, q_output, c_output, q_mask, c_mask):
        q_length = q_mask.sum(dim=-1).long()
        c_length = c_mask.sum(dim=-1).long()

        attended_q, attented_c = self._attention(q_output, q_mask, c_output, c_mask)

        enhanced_q = torch.cat([q_output,
                                attended_q,
                                q_output - attended_q,
                                q_output * attended_q],
                               dim=-1)

        enhanced_c = torch.cat([c_output,
                                attented_c,
                                c_output - attented_c,
                                c_output * attented_c],
                               dim=-1)

        projected_q = self._projection(enhanced_q)
        projected_c = self._projection(enhanced_c)

        v_ai = self._composition(projected_q, q_length)
        v_bj = self._composition(projected_c, c_length)

        v_a_avg = torch.sum(v_ai * q_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) / torch.sum(q_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * c_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) / torch.sum(c_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, q_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, c_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        return v

class LFESM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LFESM, self).__init__()
        self.LFESMMoudle = LFESMMoudle(config, gpu_list, *args, **params)
        self.loss = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config, *args, **params):
        self.LFESMMoudle = nn.DataParallel(self.LFESMMoudle, device_ids=device)
    
    def forward(self, data, config, gpu_list, mode, *args, **params):
        re = self.LFESMMoudle(data)
        if mode == 'train' or mode == 'valid':
            label = data['label']
            # label = label.to(torch.float)
            loss = self.loss(re, label)
            return re, loss
        else:
            return re
    