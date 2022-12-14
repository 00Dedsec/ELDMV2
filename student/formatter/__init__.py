from utils.logger import Logger
from formatter.BertSent import BertSentFormatter
from formatter.Word import WordFormatter
from formatter.Sent import SentFormatter
from formatter.BertWord import BertWordFormatter
from formatter.LFESM import LFESMFormatter
from formatter.sLawformerWord import sLawformerWordFormatter
from formatter.cLawformerWord import cLawformerWordFormatter
from formatter.LFESM import LFESMFormatter
from formatter.PLIFormatter import PLIFormatter
from formatter.ELDMFormatter import ELDMFormatter

logger = Logger(__name__)

formatter_list = {
    "BertSentFormatter": BertSentFormatter,
    "WordFormatter": WordFormatter,
    "SentFormatter": SentFormatter,
    "BertWordFormatter": BertWordFormatter,
    "LFESMFormatter": LFESMFormatter,
    "sLawformerWordFormatter": sLawformerWordFormatter,
    "cLawformerWordFormatter": cLawformerWordFormatter,
    "LFESMFormatter": LFESMFormatter,
    "PLIFormatter": PLIFormatter,
    "ELDMFormatter": ELDMFormatter
}


def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)

    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)

        return formatter
    else:
        logger.get_log().error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError