from model.ELDM.ELDMV2 import ELDMV2
from model.SMASHRNN.smashRNN import SMASHRNN
from model.TextCNN.TexnCNN import TextCNN
from model.BertWord.BertWord import Bertword
from model.Lawformer.sLawformer import sLawformer
from model.Lawformer.cLawformer import cLawformer
from model.LFESM.LFESM import LFESM
from model.PLI.PLI import PLI

model_list = {
    "ELDMV2": ELDMV2,
    "SMASHRNN": SMASHRNN,
    "TextCNN": TextCNN,
    "Bertword": Bertword,
    "sLawformer": sLawformer,
    "cLawformer": cLawformer,
    "LFESM": LFESM,
    "PLI": PLI
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError