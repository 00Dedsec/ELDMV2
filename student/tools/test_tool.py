from distutils.command.upload import upload
import logging
import os
import torch
from torch.autograd import Variable
from timeit import default_timer as timer
from utils.logger import Logger
from tools.eval_tool import gen_time_str, output_value, output_value_log

logger = Logger(__name__)


def test(parameters, config, gpu_list):
    model = parameters["model"]
    dataset = parameters["test_dataset"]
    model.eval()

    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_time = config.getint("output", "output_time")
    step = -1

    # 需要上传的字典
    result_dict = {}

    with torch.no_grad():
        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            logits = model(data, config, gpu_list, "test")

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

            cnt += 1

            with torch.no_grad():
                for rank, item in zip(results, data['query_candidate_id']):
                    if item[0] not in result_dict.keys():
                        result_dict[item[0]] = {}
                    result_dict[item[0]][item[1]] = [float(rank)]
        
            if step % output_time == 0:
                delta_t = timer() - start_time
                output_value('0', 'test', "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                            "%.3lf" % (total_loss / (step + 1)), "info", '\r', config)

    if step == -1:
        logger.get_log().error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    delta_t = timer() - start_time
    output_value('0', 'test', "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), 'info', '\r', config)

    

    output_value_log('0', 'test', "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), 'info', '\r', config)


    for query_id in result_dict.keys():
        result_dict[query_id] = sorted(result_dict[query_id].items(), key = lambda candidate: candidate[1][0], reverse=True)
    
    upload_dict = {}
    for query_id in result_dict.keys():
        if query_id not in upload_dict.keys():
            upload_dict[query_id] = []
        for rank in result_dict[query_id]:
            upload_dict[query_id].append(int(rank[0]))
            # upload_dict[query_id].append((int(rank[0]), rank[1][0]))

    return upload_dict