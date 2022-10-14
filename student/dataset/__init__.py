from dataset.nlp.JsonFromFiles import JsonFromFilesDataset
from dataset.nlp.MultiJsonFromFiles import MultiJsonFromFilesDataset
from dataset.nlp.PLIFromFIles import PLIFromFiles

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "MultiJsonFromFiles": MultiJsonFromFilesDataset,
    "PLIFromFiles": PLIFromFiles
}