# encoding=utf-8
# 保存每个特征的对象
import json
import math
import os

import jieba.analyse
import jieba.posseg as pseg
import numpy as np
from gensim.models import word2vec

from DataSample import DataSample
from samples_dao import read_all_labeled_samples_by_story, read_all_unlabeled_samples_by_story, \
    read_all_iteration_samples_by_story


def serialize_instance(obj):
    d = {'__classname__': type(obj).__name__}
    d.update(vars(obj))
    return d


# Dictionary mapping names to known classes
classes = {
    'DataSample': DataSample
}


def unserialize_object(d):
    clsname = d.pop('__classname__', None)
    if clsname:
        cls = classes[clsname]
        obj = cls.__new__(cls)  # Make instance without calling __init__
        for key, value in d.items():
            setattr(obj, key, value)
        return obj
    else:
        return d


if os.path.exists('labeled_story.json'):
    with open('labeled_story.json', encoding='utf-8') as f:
        labeled_story = json.load(f, object_hook=unserialize_object)
    print('labeled data loaded')
else:
    labeled_story = read_all_labeled_samples_by_story()
    with open('labeled_story.json', 'wb') as f:
        data = json.dumps(labeled_story, default=serialize_instance, ensure_ascii=False)
        f.write(data.encode('utf8', 'replace'))
    print('labeled story dumped')

if os.path.exists('unlabeled_story.json'):
    with open('unlabeled_story.json', encoding='utf-8') as f:
        unlabeled_story = json.load(f, object_hook=unserialize_object)
    print('unlabeled data loaded')
else:
    unlabeled_story = read_all_unlabeled_samples_by_story()
    with open('unlabeled_story.json', 'wb') as f:
        data = json.dumps(unlabeled_story, default=serialize_instance, ensure_ascii=False)
        f.write(data.encode('utf8', 'replace'))
    print('unlabeled story dumped')

iteration_story = []


def update_iteration_story():
    global iteration_story
    iteration_story = read_all_iteration_samples_by_story()


class word_feature:

    # w2v_model = word2vec.Word2VecKeyedVectors.load_word2vec_format('ALL_MODEL.txt')

    def __init__(self, sample, window_idx, label_data):
        self.sample = sample
        # self.words = [sample.sen_words[widx] if widx > -1 else "" for widx in window_idx]
        # self.chars = [list(self.sample.sentence)[cidx] if cidx > -1 else "" for cidx in window_idx]
        self.chars = [list(self.sample.sentence)[cidx] for cidx in range(len(self.sample.sentence))]
        self.postags = self.char_pogs(self.sample.sentence)
        self.label_data = label_data

    def calWord_features(self, char_index):
        """
        calculate word features
        define private variables as word features
        """

        # get all story sentences by id
        if self.label_data:
            if self.sample.story_id in labeled_story:
                sentences = [s.sentence for s in labeled_story[self.sample.story_id]]
            else:
                sentences = [s.sentence for s in iteration_story[self.sample.story_id]]
        else:
            sentences = [s.sentence for s in unlabeled_story[self.sample.story_id]]

        # self.postag = [self.postag(self.sample.sentence, word) for word in self.words]

        # tf-idf
        self.tfidf = [self.char_tf_idf(sentences, char) for char in self.chars]

        # textrank
        # self.tr = [self.text_rank(sentences, word) for word in self.words]

        features = {}
        for index in range(17):
            feature = {}

            if index == 100:
                continue
            else:
                # feature['word'] = self.words[index]
                if len(self.postags) > char_index-8+index >= 0:
                    # print(char_index-2+index)
                    feature['postag'] = self.postags[char_index-8+index]
                    feature['char'] = self.chars[char_index-8+index]
                    feature['tf-idf'] = self.tf_idf[char_index-8+index]
                else:
                    feature['postag'] = 'NULL'
                    feature['char'] = 'NULL'
                    feature['tf-idf'] = -1

                # feature['tfidf'] = self.tfidf[index]
                # feature['tr'] = self.tr[index]

            features[str(index)] = feature

        for index in range(5):
            feature = {}

            if index == 2:
                continue
            else:
                if len(self.postags) > char_index - 2 + index >= 0:
                    feature['char'] = self.chars[char_index-2+index]
                else:
                    feature['char'] = ''
                # feature['tfidf'] = self.tfidf[index]
                # feature['tr'] = self.tr[index]

            features['second' + str(index)] = feature

        return features

    def vector(self, word):
        """
        :param word: word
        :return: list, vector of word
        """
        try:
            w2v = list(word_feature.w2v_model.get_vector(word))
        except KeyError:
            w2v = []
        return w2v

    def postag(self, sen, word):
        """
        :param sen:
        :param word:
        :return:
        """

        if word == '':
            return ''
        words = pseg.cut(sen)
        for w, p in words:
            if w == word:
                return p
        return ''

    def tf_idf(self, sentences, word):

        """

        :param sentences: text sentences
        :param word: word
        :return: tfidf value of word
        """
        keywords = jieba.analyse.extract_tags('。'.join(sentences), topK=None, withWeight=True)
        for w, tfidf in keywords:
            if w == word:
                return tfidf
        return 0.0

    def char_tf_idf(self, sentences, char):
        corpus = [list(sen) for sen in sentences]
        tf = 0.0
        for sen in corpus:
            tf += sen.count(char) / len(sen)
        idf = 0.0
        for sen in corpus:
            if sen.__contains__(char):
                idf += 1
                continue
        idf = math.log(len(corpus) / (idf + 1))

        return tf * idf


    def char_pogs(self, sentence):
        pogs = []
        words = pseg.cut(sentence)
        for w, p in words:
            for _ in range(len(w)):
                pogs.append(p)
        return pogs

    def text_rank(self, sentences, word):
        """
        :param sentences: text sentences
        :param word: word
        :return: textrank value of word
        """
        keywords = jieba.analyse.textrank('。'.join(sentences), topK=None, withWeight=True)
        for w, textrank in keywords:
            if w == word:
                return textrank
        return 0.0

