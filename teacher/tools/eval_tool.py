from utils.logger import Logger
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import json
import torchmetrics
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


def valid(model, dataset, epoch, writer, config, gpu_list, output_function, mode="valid" ,*args, **params):
    model.eval()

    valid_accuracy = torchmetrics.Accuracy()
    valid_precision = torchmetrics.Precision()
    valid_recall = torchmetrics.Recall()
    valid_f1 = torchmetrics.F1Score()
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""
    output_time = config.getint("output", "output_time")
    step = -1

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, config, gpu_list, "valid")

        loss, logits = results["loss"], results["logits"]
        total_loss += float(loss)
        cnt += 1
        labels = data['labels']
        re = logits.max(dim=2)[1].reshape(-1).cpu()
        la = labels.reshape(-1).cpu()

        re_ = []
        la_ = []
        for i in range(0, la.shape[0]):
                if la[i] != -100 and la[i] != 0:
                    re_.append(re[i])
                    la_.append(la[i])

        # print(re_)
        # print(la_)
        re_ = torch.tensor(re_)
        la_ = torch.tensor(la_)
        acc = valid_accuracy(re_, la_)
        precision = valid_precision(re_, la_)
        recall = valid_recall(re_, la_)
        f1 = valid_f1(re_, la_)

        acc_result = {
            'acc: ': round(float(acc), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1': round(float(f1), 4)
        }

        if step % output_time == 0:
            delta_t = timer() - start_time
            # output_info = output_function(acc_result, config)

            output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                        "%.3lf" % (total_loss / (step + 1)), acc_result, '\r', config)

    if step == -1:
        logger.get_log().error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    delta_t = timer() - start_time
    # output_info = output_function(acc_result, config)

    acc_result = {'acc': round(float(valid_accuracy.compute()), 4),
                    'precision': round(float(valid_precision.compute()), 4),
                    'recall': round(float(valid_recall.compute()), 4),
                    'f1': round(float(valid_f1.compute()), 4)}
    
    valid_accuracy.reset()
    valid_precision.reset()
    valid_recall.reset()
    valid_f1.reset()
    
    output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                "%.3lf" % (total_loss / (step + 1)),acc_result, '\r', config)
 

    output_value_log(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)), acc_result, '\r', config)

    writer.add_scalar(config.get("output", "model_name") + "_eval_epoch", float(total_loss) / (step + 1),
                        epoch)

    model.train()