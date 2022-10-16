import imp
from model.BertCRF import BertCRF

model_list = {
    "BertCRF": BertCRF
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError