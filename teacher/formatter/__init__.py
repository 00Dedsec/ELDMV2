from utils.logger import Logger
from formatter.BertSCM import BertSCM
from formatter.SentSCM import SentSCM
from formatter.BertFormatter import BertFormatter
logger = Logger(__name__)

formatter_list = {
    "BertSCM": BertSCM,
    "SentSCM": SentSCM,
    "BertFormatter": BertFormatter
}


def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.get_log().warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)

    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)

        return formatter
    else:
        logger.get_log().error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError