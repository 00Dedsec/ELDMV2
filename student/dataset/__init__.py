from dataset.nlp.JsonFromFiles import JsonFromFilesDataset
from dataset.nlp.MultiJsonFromFiles import MultiJsonFromFilesDataset
from dataset.nlp.PLIFromFIles import PLIFromFiles
from dataset.nlp.ELDMFromFiles import ELDMFromFiles

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "MultiJsonFromFiles": MultiJsonFromFilesDataset,
    "PLIFromFiles": PLIFromFiles,
    "ELDMFromFiles": ELDMFromFiles
}