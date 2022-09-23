from numpy import sort
from utils.logger import Logger
import os
import torch
from torchmetrics import RetrievalNormalizedDCG
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import json
import numpy as np

logger = Logger(__name__)

def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if not (end is None):
        print(s, end=end)
    else:
        print(s)

def output_value_log(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)

    logger.get_log().info(s)        


def valid(model, dataset, epoch, writer, config, gpu_list, output_function, mode="valid"):
    model.eval()

    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""
    output_time = config.getint("output", "output_time")
    step = -1

    # 计算指标
    dcg_rank_dict = {}
    idcg_rank_dict = {}

    index = []
    preds = []
    targets = []

    ndcg = RetrievalNormalizedDCG(k=30)

    with torch.no_grad():
        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            logits, loss = model(data, config, gpu_list, "valid")

            total_loss += float(loss)
            cnt += 1
            
            # logits [batch_size, 4]
            def f(x):
                # if(max(x[3],x[2],x[1],x[0]) == x[3]):
                #     return 3.0
                # elif(max(x[3],x[2],x[1],x[0]) == x[2]):
                #     return 2.0
                # elif(max(x[3],x[2],x[1],x[0]) == x[1]):
                #     return 1.0
                # else:
                #     return 0.0
                return x[3]-x[2]-x[1]-x[0]

            results = map(f, logits)


            # 将结果添加到result_rank_dict中
            # result_rank_dict = {
            #   1(ridx): {
            #       11(candidate_id): rank, label   
            #       22(candidate_id): rank, label
            #   },
            #  2(ridx): {
            #      ...
            #   }
            # }
            result_rank_dict = {}
            for rank, item, label in zip(results, data['query_candidate_id'], data['label']):
                if item[0] not in result_rank_dict.keys():
                    result_rank_dict[item[0]] = {}
                result_rank_dict[item[0]][item[1]] = [rank,label]

            # result_rank_dict = {
            #   1(ridx): {
            #       (11(candidate_id), [rank, label] )
            #       (22(candidate_id), [rank, label] )
            #   },
            #  2(ridx): {
            #      ...
            #   }
            # }
            for item in result_rank_dict.keys():
                result_rank_dict[item] = sorted(result_rank_dict[item].items(), key = lambda candidate: candidate[1][0], reverse=True)

            for query_id in result_rank_dict.keys():
                dcg_rank_dict[query_id] = [rank[1][1] for rank in result_rank_dict[query_id]]
            
            for query_id in result_rank_dict.keys():
                idcg_rank_dict[query_id] = sorted([label[1][1] for label in result_rank_dict[query_id]], reverse=True)

            # 按照query_id排序
            # dcg_rank_dict = {query_id: [1,2,3...], query_id: [1,2,3...]}
            for query_id in sorted(dcg_rank_dict.keys()):
                for rank,iRank in zip(dcg_rank_dict[query_id], idcg_rank_dict[query_id]):
                    index.append(query_id)
                    preds.append(float(rank))
                    targets.append(iRank)

            ndcg_30 = ndcg(torch.tensor(preds), torch.tensor(targets), indexes=torch.tensor(index))
                
            if step % output_time == 0:
                delta_t = timer() - start_time
                output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                                "%.3lf" % (total_loss / (step + 1)), ndcg_30, '\r', config)


    delta_t = timer() - start_time
    output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), ndcg_30, '\r', config)

    

    output_value_log(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), ndcg_30, '\r', config)

    del ndcg
    del ndcg_30
    del result_rank_dict
    del dcg_rank_dict
    del idcg_rank_dict
    
    writer.add_scalar(config.get("output", "model_name") + "_eval_epoch", float(total_loss) / (step + 1),
                      epoch)

    model.train()