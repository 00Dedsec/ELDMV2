import json
import os
from tqdm import tqdm
import numpy as np


def load_embedding_dict(path):
    lines = open(path, encoding='utf-8').readlines()
    embedding_dict = {}
    for i, line in enumerate(lines):
        if i == 0 and '\n' in line:
            continue

        if '\n' in line:
            line = line[:-2]    # remove the '[blank]\n' in the end of the string

        split = line.split(" ")
        embedding_dict[split[0]] = np.array(list(map(float, split[1:])))

    unk = sum(list(embedding_dict.values())) / len(embedding_dict.keys())
    embedding_dict['<UNK>'] = unk
    embedding_dict['<PAD>'] = np.random.randn(unk.shape[0])
    return embedding_dict

if __name__ == "__main__":
    print("start to load...")
    embedding_dict = load_embedding_dict('../data/sgns.wiki.bigram-char')
    word2id = {}
    word2vec_mat = []
    for (k, v) in tqdm(embedding_dict.items(), desc='reading pretrained word embeddings'):
        id = len(word2id)
        word2id[k] = id
        word2vec_mat.append(v)

    word2vec_mat = np.array(word2vec_mat, dtype=np.float32)
    if not os.path.exists('../data/word2vec.npy'):
        np.save('../data/word2vec.npy', word2vec_mat)
    
    json.dump(word2id, open("..\data\word2id.txt", "w", encoding="utf8"),
            ensure_ascii=False,
            indent=2)