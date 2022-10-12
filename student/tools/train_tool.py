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


def train(parameters, config, gpu_list, do_test=True, *args, **params):
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
    logger.get_log().info("The current model is：" + config.get("model", "model_name") + "!")

    total_len = len(dataset)

    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num

        exp_lr_scheduler.step(current_epoch)

        total_loss = 0

        output_info = ""
        step = -1
        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor) and key != 'query_candidate_id':
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            optimizer.zero_grad()

            logits, loss = model(data, config, gpu_list, "train")
            
            # loss = (loss-0.1).abs() + 0.1

            total_loss += float(loss)

            loss.backward()
            optimizer.step()

            if step % output_time == 0:
                delta_t = timer() - start_time

                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                             "%.3lf" % (total_loss / (step + 1)), "info", '\r', config)

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
                    "%.3lf" % (total_loss / (step + 1)), "info", '\r', config)
                    
        if current_epoch % test_time == 0:
            with torch.no_grad():
                valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function)
                if do_test:
                    valid(model, test_dataset, current_epoch, writer, config, gpu_list, output_function, mode="test")




