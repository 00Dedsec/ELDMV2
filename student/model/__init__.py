from model.ELDM.ELDMV2 import ELDMV2
from model.SMASHRNN.smashRNN import SMASHRNN
from model.TextCNN.TexnCNN import TextCNN

model_list = {
    "ELDMV2": ELDMV2,
    "SMASHRNN": SMASHRNN,
    "TextCNN": TextCNN
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError