# -*- coding: utf-8 -*-
# @File    : classify_sentence.py
import jieba
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from samples_dao import read_all_labeled_samples


def nb_model(train, train_label, test, test_label):
    clf_model = MultinomialNB(alpha=0.01)
    clf_model.fit(train, train_label)
    predict_results = clf_model.predict(test)

    predict_list = predict_results.tolist()

    print(metrics.precision_score(test_label, predict_list, average='binary', pos_label=1))
    print(metrics.precision_score(test_label, predict_list, average='binary', pos_label=0))

    print('nb: ' + str(metrics.precision_score(test_label, predict_list, average='micro')))


def knn_model(train, train_label, test, test_label):
    knn_model = KNeighborsClassifier(n_neighbors=8)
    knn_model.fit(train, train_label)
    predict_results = knn_model.predict(test)

    predict_list = predict_results.tolist()
    print('knn: ' + str(metrics.precision_score(test_label, predict_list, average='micro')))



def svm_model(train, train_label, test, test_label):
    svm_clf = SVC(kernel="linear", verbose=False)
    svm_clf.fit(train, train_label)
    predict_results = svm_clf.predict(test)

    predict_list = predict_results.tolist()
    print('svm: ' + str(metrics.precision_score(test_label, predict_list, average='micro')))



def text_classification():

    all_samples = read_all_labeled_samples()
    corpus = [s.sentence for s in all_samples]
    label = [0 if s.fps[0] == 'null' else 1 for s in all_samples]

    X_train, X_test, y_train, y_test = train_test_split(corpus, label, test_size = 0.33)

    # print(len(y_train))
    # print(len(y_test))

    # 构建词典
    vec_total = CountVectorizer(tokenizer=lambda s : jieba.lcut(s))
    vec_total.fit_transform(corpus)

    # 基于构建的词典分别统计训练集/测试集词频
    vec_train = CountVectorizer(vocabulary=vec_total.vocabulary_, tokenizer=lambda s : jieba.lcut(s))
    tf_train = vec_train.fit_transform(X_train)

    # print(jieba.lcut(X_train[0]))
    # print(y_train[0])
    # print(np.sum(tf_train[0]))

    vec_test = CountVectorizer(vocabulary=vec_total.vocabulary_, tokenizer=lambda s : jieba.lcut(s))
    tf_test = vec_test.fit_transform(X_test)

    # 进一步计算词频-逆文档频率
    tfidftransformer = TfidfTransformer()
    tfidf_train = tfidftransformer.fit(tf_train).transform(tf_train)
    tfidf_test = tfidftransformer.fit(tf_test).transform(tf_test)

    # print(tfidf_train)

    nb_model(tfidf_train, y_train, tfidf_test, y_test)
    knn_model(tfidf_train, y_train, tfidf_test, y_test)
    svm_model(tfidf_train, y_train, tfidf_test, y_test)



if __name__ == '__main__':
    text_classification()