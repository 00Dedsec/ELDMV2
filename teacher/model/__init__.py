import imp
from model.lawformer import teacher_moodel

model_list = {
    "teacher_moodel": teacher_moodel
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError