import torch
import os
from utils.logger import Logger
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.autograd import Variable
from reader.reader import init_formatter, init_test_dataset
from tools.eval_tool import output_value, output_value_log,gen_time_str,valid
import shutil
from timeit import default_timer as timer
import torchmetrics
from utils.bio_lables import bio_labels
import torch.nn as nn
logger = Logger(__name__)

def checkpoint(filename, model, optimizer, trained_epoch, config, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step
    }

    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.get_log().warning("Cannot save models with error %s, continue anyway" % str(e))


def train(parameters, config, gpu_list, do_test=False ,*args, **params):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")
    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path):
        logger.get_log().warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"] + 1
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]

    

    if do_test:
        init_formatter(config, ["test"])
        test_dataset = init_test_dataset(config)

    if trained_epoch == 0:
        shutil.rmtree(
            os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")), True)

    os.makedirs(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
            exist_ok=True)        

    writer = SummaryWriter(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                           config.get("output", "model_name"))

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    logger.get_log().info("Training start....")

    total_len = len(dataset)

    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num

        exp_lr_scheduler.step(current_epoch)

        train_accuracy = torchmetrics.Accuracy(num_classes=len(bio_labels)+2, average='macro')
        train_precision = torchmetrics.Precision(num_classes=len(bio_labels)+2, average='macro')
        train_recall = torchmetrics.Recall(num_classes=len(bio_labels)+2, average='macro')
        train_f1 = torchmetrics.F1Score(num_classes=len(bio_labels)+2, average='macro')

        # if len(gpu_list) > 0:
        #     train_accuracy = nn.DataParallel(train_accuracy, device_ids=gpu_list)
        #     train_precision = nn.DataParallel(train_precision, device_ids=gpu_list)
        #     train_recall = nn.DataParallel(train_recall, device_ids=gpu_list)
        #     train_f1 = nn.DataParallel(train_f1, device_ids=gpu_list)

        total_loss = 0

        output_info = ""
        step = -1
        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            optimizer.zero_grad()

            results = model(data, config, gpu_list, "train")

            loss, logits = results["loss"], results["logits"]
            total_loss += float(loss)
            
            loss.backward()
            optimizer.step()

            # logits [batch_size, max_len, num_class]
            # lables [batch_size, max_len]
            # logits.max(dim=2)[1] [batch_size, max_len]
            # logits.max(dim=2)[1].reshape(-1) [batch_size*max_len]
            # labels.reshape(-1) [batch_size*max_len]


            labels = data['labels'] 

            re = logits.max(dim=2)[1].reshape(-1).cpu() #[batch_size * max_len]
            la = labels.reshape(-1).cpu() #[batch_size * max_len]
            
            re_ = []
            la_ = []
            for i in range(0, la.shape[0]):
                    if la[i] != -100:
                        re_.append(re[i])
                        la_.append(la[i])

            # print(re_)
            # print(la_)
            re_ = torch.tensor(re_)
            la_ = torch.tensor(la_)
            acc = train_accuracy(re_, la_)
            precision = train_precision(re_, la_)
            recall = train_recall(re_, la_)
            f1 = train_f1(re_, la_)

            acc_result = {
                'acc: ': round(float(acc), 4),
                'precision': round(float(precision), 4),
                'recall': round(float(recall), 4),
                'f1': round(float(f1), 4)
            }

            if step % output_time == 0:
                # output_info = output_function(acc_result, config)

                delta_t = timer() - start_time


                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                            "%.3lf" % (total_loss / (step + 1)), acc_result, '\r', config)

            global_step += 1
            writer.add_scalar(config.get("output", "model_name") + "_train_iter", float(loss), global_step)
            
        
        
        exp_lr_scheduler.step(trained_epoch)
        if step == -1:
            logger.get_log().error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError

        checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config,
                   global_step)

        writer.add_scalar(config.get("output", "model_name") + "_train_epoch", float(total_loss) / (step + 1),
                        current_epoch)

        output_value_log(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)), 
                    {'acc': round(float(train_accuracy.compute()), 4),
                    'precision': round(float(train_precision.compute()), 4),
                    'recall': round(float(train_recall.compute()), 4),
                    'f1': round(float(train_f1.compute()), 4)}, '\r', config)

        train_accuracy.reset()
        train_precision.reset()
        train_recall.reset()
        train_f1.reset()

        if current_epoch % test_time == 0:
            with torch.no_grad():
                valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function ,*args, **params)
                if do_test:
                    valid(model, test_dataset, current_epoch, writer, config, gpu_list, output_function, mode="test" ,*args, **params)




