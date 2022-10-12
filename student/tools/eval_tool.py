from numpy import sort
from utils.logger import Logger
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import json
import numpy as np
import math

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

def ndcg(ranks, gt_ranks, K):
    dcg_value = 0.
    idcg_value = 0.
    # log_ki = []

    sranks = sorted(gt_ranks, reverse=True)

    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi

    return dcg_value/idcg_value

def metrics(prediction, config):
    label = json.load(open(config.get('data','valid_label_top30_data_path'),  'r'))
    preds = []
    target = []
    indexes = []
    count=0
    sndcg_10 = 0.0
    sndcg_20 = 0.0
    sndcg_30 = 0.0
    sp_5 = 0.0
    sp_10 = 0.0
    smap = 0.0
    for item_prediction in list(prediction.keys()):
        preds = []
        target = []
        indexes = []
        count = count+1
        item_prediction = str(item_prediction)
        temp = list(map(lambda a: label[item_prediction][str(a)] if str(a) in label[item_prediction].keys() else 0, prediction[item_prediction]))
        preds.extend(temp[0:30])
        target.extend(sorted(label[item_prediction].values(), reverse=True)[0:30])
        sndcg_10 += ndcg(preds, target, 10)
        sndcg_20 += ndcg(preds, target, 20)
        sndcg_30 += ndcg(preds, target, 30)
        
        label_sort = sorted(label[item_prediction].items(), key = lambda x: x[1], reverse = True)
        label_sort = list(map(lambda a: a[0], label_sort))
        topk = 10
        preds = [i for i in prediction[item_prediction] if str(i) in label_sort[:30]]
        sp_5 += float(len([j for j in preds[:topk] if label[item_prediction][str(j)] == 3])/topk)
        
        topk = 5
        preds = [i for i in prediction[item_prediction] if str(i) in list(label[item_prediction].keys())[:30]]
        sp_10 += float(len([j for j in preds[:topk] if label[item_prediction][str(j)] == 3])/topk)
        
        ranks = [i for i in prediction[item_prediction] if str(i) in label[item_prediction]] 
        rels = [ranks.index(i) for i in ranks if label[item_prediction][str(i)] == 3]
        tem_map = 0.0
        for rel_rank in rels:
            tem_map += float(len([j for j in ranks[:rel_rank+1] if label[item_prediction][str(j)] == 3])/(rel_rank+1))
        if len(rels) > 0:
            smap += tem_map / len(rels)
        
    precision_5 = sp_5/len(prediction.keys())   
    precision_10 = sp_10/len(prediction.keys())

    smap = smap/len(prediction.keys())
        
    ndcg_10 = sndcg_10/len(prediction.keys())
    ndcg_20 = sndcg_20/len(prediction.keys())
    ndcg_30 = sndcg_30/len(prediction.keys())
    return {
        'precision_5': round(precision_5, 4),
        'precision_10': round(precision_10, 4),
        'map': round(smap, 4) ,
        'ndcg_10': round(ndcg_10, 4),
        'ndcg_20': round(ndcg_20, 4),
        'ndcg_30': round(ndcg_30, 4)
    }


def valid(model, dataset, epoch, writer, config, gpu_list, output_function, mode="valid"):
    model.eval()

    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""
    output_time = config.getint("output", "output_time")
    step = -1
    result_rank_dict = {}
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
                return 1.5*x[3]+x[2]+x[1]-x[0]

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
            for rank, item, label in zip(results, data['query_candidate_id'], data['label']):
                if item[0] not in result_rank_dict.keys():
                    result_rank_dict[item[0]] = {}
                result_rank_dict[item[0]][item[1]] = [rank,label]
                
            if step % output_time == 0:
                delta_t = timer() - start_time
                output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                                "%.3lf" % (total_loss / (step + 1)), 'info', '\r', config)
    
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

        upload_dict = {}
        for query_id in result_rank_dict.keys():
            if query_id not in upload_dict.keys():
                upload_dict[str(query_id)] = []
            for rank in result_rank_dict[query_id]:
                upload_dict[str(query_id)].append(int(rank[0]))
        merics_result = metrics(upload_dict, config)


        delta_t = timer() - start_time
        output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)), merics_result, '\r', config)

        

        output_value_log(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)), merics_result, '\r', config)

        del result_rank_dict
        
        writer.add_scalar(config.get("output", "model_name") + "_eval_epoch", float(total_loss) / (step + 1),
                        epoch)

    model.train()