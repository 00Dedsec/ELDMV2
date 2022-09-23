import json

from .accuracy_tool import gen_micro_macro_result, get_accuracy


def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    
    temp = gen_micro_macro_result(data)
    acc = get_accuracy(data)
    result = {}
    result["acc"] = acc
    # for name in ["mip", "mir", "mif", "map", "mar", "maf"]:
    #     result[name] = temp[name]
    for name in ["mip", "mir", "mif"]:
        result[name] = temp[name]

    

    return json.dumps(result, sort_keys=True)