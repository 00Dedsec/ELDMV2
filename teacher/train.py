import torch

import torch
import argparse
import os

from utils.logger import Logger
from utils.config_parser import create_config
from tools.init_tool import init_all
from tools.train_tool import train

logger = Logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', help="specific config file", required=True)
parser.add_argument('--gpu', '-g', help="gpu id list")
parser.add_argument('--checkpoint', help="checkpoint file path")
parser.add_argument('--do_test', help="do test while training or not", action="store_true")
parser.add_argument('--local_rank', help='local rank', default=-1, type=int)
args = parser.parse_args()

logger.get_log().info("link_start...")

use_gpu = True
gpu_list = []
if args.gpu is None:
    use_gpu = False
else:
    use_gpu = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device_list = args.gpu.split(",")
    for a in range(0, len(device_list)):
        gpu_list.append(int(a))

os.system("clear")

cuda = torch.cuda.is_available()

logger.get_log().info("CUDA available: %s" % str(cuda))
if not cuda and len(gpu_list) > 0:
    logger.get_log().error("CUDA is not available but specific gpu id")
    raise NotImplementedError

configFilePath = args.config
config = create_config(configFilePath)

if config.getboolean("distributed", "use"):
        torch.distributed.init_process_group(backend=config.get("distributed", "backend"))

parameters = init_all(config,gpu_list, args.checkpoint,"train")

"""
parameters:
    train_dataset,
    valid_dataset,
    optimizer,
    trained_epoch,
    output_function,
    global_step
"""

do_test = args.do_test

logger.get_log().info("the size of train_dataset: " + str(len(parameters["train_dataset"])))

for item in parameters["train_dataset"]:
    logger.get_log().info("an example of train_datatset:")
    logger.get_log().info(item)
    break

train(parameters,config, gpu_list, do_test)
