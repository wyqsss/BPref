import numpy as np
import torch


def gen_embedding(glove_path, sentence):
    sentence = sentence.lower()
    dictionary = {}
    with open(glove_path) as f:
        for line in f.readlines():
            line = line.strip()
            parts = line.split()
            dictionary[parts[0]] = np.array(list(map(eval, parts[1:])))

    sent_emb = []
    for w in sentence.split():
        try:
            sent_emb.append(dictionary[w])
        except KeyError:
            print(f"do not exist key {w}")
            sent_emb.append(np.zeros(50))
    for i in range(len(sent_emb), 11):
        sent_emb.append(np.zeros(50))
    return torch.tensor(np.asarray(sent_emb), dtype=torch.float32)