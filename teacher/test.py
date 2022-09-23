from struct import pack
from cv2 import randn
from transformers import AutoTokenizer
import torchmetrics
import torch
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

text = tokenizer.tokenize("啊？")
ids = tokenizer.convert_tokens_to_ids(text)
print(ids)
print(tokenizer.decode(ids))