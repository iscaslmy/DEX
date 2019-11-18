# -*- coding: utf-8 -*-
# @File    : generateCorpus.py
from samples_dao import read_all_data_fps

corpus = read_all_data_fps()
corpus = [list(s) for s in corpus]
with open("corpus.txt","w") as f:
    for line in corpus:
        f.writelines(' '.join(line) + '\n')