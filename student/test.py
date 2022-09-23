


import argparse
import os
import torch
import logging
import json

from tools.init_tool import init_all
from utils.logger import Logger
from utils.config_parser import create_config
from tools.test_tool import test

logger = Logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path", required=True)
    parser.add_argument('--result', help="result file path", required=True)
    args = parser.parse_args()

    configFilePath = args.config

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

    config = create_config(configFilePath)

    cuda = torch.cuda.is_available()
    logger.get_log().info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.get_log().error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    parameters = init_all(config, gpu_list, args.checkpoint, "test")

    for item in parameters["test_dataset"]:
        logger.get_log().info("an example of test_datatset:")
        logger.get_log().info(item)
        break

    json.dump(test(parameters, config, gpu_list), open(args.result, "w", encoding="utf8"), ensure_ascii=False,
              sort_keys=True, indent=2)