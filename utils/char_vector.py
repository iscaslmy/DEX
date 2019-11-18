# -*- coding: utf-8 -*-
import codecs
import os

from gensim.models import word2vec, Word2Vec, KeyedVectors

from samples_dao import read_all_labeled_samples_by_story, read_all_unlabeled_samples_by_story, read_stories

if not os.path.exists('corpus.txt'):
    labeled_data = read_all_labeled_samples_by_story()
    unlabeled_data = read_all_unlabeled_samples_by_story()

    data = [sample.sentence for samples in list(labeled_data.values()) for sample in samples]
    data.extend([sample.sentence for samples in list(unlabeled_data.values()) for sample in samples])

    with codecs.open('corpus.txt', 'w', encoding='utf-8') as f:
        for line in data:
            f.write(' '.join(list(line.strip())) + '\n')
    word_cnt = set()
    for line in data:
        for w in list(line):
            word_cnt.add(w)
    print(len(word_cnt))


if not os.path.exists('w2v.txt'):
    sentences = word2vec.LineSentence('corpus.txt')
    # for sentence in sentences:
    #     print(sentence)
    #
    # exit()
    model = word2vec.Word2Vec(sentences, hs=1, min_count=0, window=5, size=100)
    model.wv.save_word2vec_format('w2v.bin', binary=True)


model = KeyedVectors.load_word2vec_format('w2v.bin', binary=True)
model.get_vector()
if '我们' not in model.wv:
    print('no')
# print(model.wv['我们'])
