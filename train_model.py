import csv
import random
from functools import partial
from multiprocessing.pool import Pool

import distance
import time

import re

import os

import numpy as np
from numpy.linalg import norm
from pycrfsuite import Trainer, ItemSequence
from pycrfsuite import Tagger
from sklearn.feature_extraction.text import CountVectorizer

import data_preparation
import samples_dao
from sklearn import metrics
from word_feature import word_feature, update_iteration_story

from md_config import getConfig

################################################################################################################


def init_hyper_parameters():
    """
    初始化相关参数，只是当前类是主类运行时候会被调用
    :return:
    """
    """bootstrapping parameters"""
    global iteration_num
    iteration_num = int(getConfig('training', 'iteration_num'))

    # 使用语言模型过滤阈值  -1000为不过滤
    global lm_threshold
    lm_threshold = float(getConfig('training', 'lm_threshold'))

    # CRF阈值参数
    global unlabeled_threshold
    global iter_sample_rate
    global unlabeled_threshold1
    unlabeled_threshold = float(getConfig('training', 'unlabeled_threshold'))
    iter_sample_rate = float(getConfig('training', 'iter_sample_rate'))
    unlabeled_threshold1 =float( getConfig('training', 'unlabeled_threshold1'))

    # 动态采样初始值和变化值
    global sample_num
    global sample_delta
    sample_num = int(getConfig('training', 'sample_num'))
    sample_delta = int(getConfig('training', 'sample_delta'))

    # 线程数
    global thread_num
    thread_num = int(getConfig('training', 'thread_num'))

    # useless now
    global conf_factor
    conf_factor = getConfig('training', 'conf_factor')

    """空样本采样"""
    # 采样大小  非空样本的多少倍
    global null_sampling_num
    null_sampling_num = float(getConfig('training', 'null_sampling_num'))
    # 采样的CRF阈值
    global null_sampling_threshold
    null_sampling_threshold = float(getConfig('training', 'null_sampling_threshold'))

    """数据划分参数"""
    # 是否进行数据的重新划分
    global reset_data
    global training_percentage
    reset_data = bool(getConfig('training', 'reset_data'))
    # 数据划分时候 训练集的比例
    training_percentage = float(getConfig('training', 'training_percentage'))

    # useless now
    global lan_lambda
    lan_lambda = float(getConfig('training', 'lan_lambda'))

    """模型评估参数"""
    # 编辑距离相似度阈值
    global sim_t
    sim_t = float(getConfig('training', 'sim_t'))

# all samples grouped by story_id
# key: story_id       value: list:DataSample对象
# samples_dict = samples_dao.read_all_labeled_samples_by_story()

def build_model_features(sample, window_size, label_data):
    """
    :param sample:  一个DataSample对象
    :param window_size:
    :param label_data:
    :return: sample对应的ItemSequence
    """
    features = []

    chars = list(sample.sentence)
    for index, word in enumerate(chars):
        '''
        build features 
        '''
        windows_words = [i if i > 0 else -1 for i in range(index - window_size // 2, index)]
        windows_words.extend([i if i < len(chars) else -1 for i in range(index, index + window_size // 2 + 1)])

        feature = word_feature(sample, windows_words, label_data)

        features.append(feature.calWord_features(index))

    # print(features)
    item_sequence = ItemSequence(features)

    return item_sequence


def cross_evaluation_without_semi_supervision():
    '''
    10折评价模型，不进行半监督流程
    :return:
    '''
    # get all labeled samples divided into 10 fold
    all_labeled_samples = data_preparation.divide_samples_into_10()

    accuracy_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for test_index in range(10):
        test_samples = all_labeled_samples[test_index]
        del all_labeled_samples[test_index]

        training_samples = []
        for fold_samples in all_labeled_samples:
            training_samples.extend(fold_samples)

        model_name = 'model_' + str(test_index)
        train_model(training_samples, model_name)

        accuracy, recall, f1 = evaluate_model_by_story(model_name, test_samples)
        accuracy_sum += accuracy
        recall_sum += recall
        f1_sum += f1

    accuracy_sum = accuracy_sum / 10
    recall_sum = recall_sum / 10
    f1_sum = f1_sum / 10

    print("Total: \n\tAccuracy: %f\n\tRecall: %f\n\tF1: %f\n\n\n" % (accuracy_sum, recall_sum, f1_sum))


def cal_confidence_score(sample, model_name):
    '''
    给未标注数据打标签，并且计算得分，返回结果
    :param model: 模型
    :param sample: 对象
    :return: 预测的功能点名称  置信度
    '''
    model = Tagger()
    model.open(model_name)
    # unlabeled sample features
    feature_sequence = build_model_features(sample, 17, False)
    # words
    # words = sample.sen_words
    chars = list(sample.sentence)
    model.set(feature_sequence)
    predicted_labels = model.tag()

    # get predicted_fps
    fp_list = []
    fp = ''
    for index, label in enumerate(predicted_labels):
        if label == 'B' or label == 'I' or label == 'E':
            fp += chars[index]
        if label == 'N' and len(fp) > 0:
            fp = fp.replace('的', '')
            if '国税' in fp or '公积金' in fp or '个税' in fp:
                fp = ''
                continue
            fp_list.append(fp)
            fp = ''

    # calculate the probability of tagging
    crf_confidence = model.probability(predicted_labels)

    lan_confidence = 0
    global lm_threshold
    filtered_fp_list = []
    for fp_name in fp_list:
        if len(fp_name) == 0 or len(fp_name) > 12:
            continue
        data_fp_name = ' '.join(list(fp_name))
        lan_confidence_temp = -1
        if len(re.findall('[a-zA-Z0-9+]+', fp_name)) > 0:
            lan_confidence_temp += 5
        if lan_confidence_temp > lm_threshold:
            filtered_fp_list.append(fp_name)

    if len(filtered_fp_list) == 0:
        predicted_fps = 'null'
    else:
        predicted_fps = ' '.join(filtered_fp_list)

    # print(str(sample.story_id) +' '+ sample.sentence +' '+ fp +' '+ str(confidence))
    # 为防止多进程乱序执行导致结果跟sample不对应，因此同时返回sample信息
    return sample.story_id, sample.sentence, predicted_fps, crf_confidence


def max_sim(idx, ids, sim_matrix):
    if len(ids) == 0:
        return 0
    max_sim_score = 0
    for i in ids:
        sim_score = sim_matrix[idx][i]
        if sim_score == -1:
            sim_score = sim_matrix[i][idx]
        max_sim_score = max(sim_score, max_sim_score)
    return max_sim_score


def cal_sim_matrix(results):
    sim_matrix = []
    for row, result_1 in enumerate(results):
        sim_row = [-1 for _ in range(len(results))]
        for col in range(row + 1, len(results)):
            s1, s2 = ' '.join(list(result_1[1])), ' '.join(list(results[col][1]))
            cv = CountVectorizer(tokenizer=lambda s: s.split())
            corpus = [s1, s2]
            vectors = cv.fit_transform(corpus).toarray()
            sim_score = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
            sim_row[col] = sim_score
        sim_matrix.append(sim_row)
    return sim_matrix


def predicted_all_unlabeled_samples(model_name, samples_num=5000):
    '''
    获取所有的未标注样本，计算得分，将得分高的样本加入训练集（存入数据库）
    :param model_name: 模型名称
    :param samples_num: 迭代采样大小
    :return:
    '''
    # tagger = Tagger()
    # tagger.open(model_name)

    # data need to be inserted into database
    # [(story_id, sentence, fps, p)]
    database_data = dict()
    # extra_database_data = []

    unlabeled_samples = samples_dao.read_all_unlabeled_samples()
    all_unlabeled_samples = random.sample(unlabeled_samples, min(samples_num, len(unlabeled_samples)) )
    # all_unlabeled_samples = samples_dao.read_all_unlabeled_samples()

    start_time = time.time()
    # multi processing pool
    with Pool(thread_num) as pool:
        partial_work = partial(cal_confidence_score, model_name=model_name)
        results = pool.map(partial_work, all_unlabeled_samples)
    results = list(filter(lambda s: len(s[2]) != 0 and s[3] != 0.0, results))
    # sens = [res[1] for res in results]
    # max_len = max([len(s) for s in sens])
    print('get result')
    # sim_matrix = cal_sim_matrix(results)
    # print('get sim_matrix')
    global unlabeled_threshold
    print('threshold is ', unlabeled_threshold)
    # mmr_score = 1
    # while mmr_score > unlabeled_threshold:
    # best_sample = ()
    # best_idx = 0
    # max_mmr_score = -1

    null_samples = []
    not_null_sum = 0
    for idx, result in enumerate(results):
        # if probability is larger than the threshold,
        # add the unlabeled sample into database for further training
        # (story_id, sentence, fp, p)
        # if confidence > unlabele_threshold:
        #     database_data.append((story_id, sentence, fp, confidence))
        # if results[idx] is None:
        #     continue

        # 去掉功能点长度在句子中比例大的
        # if len(result[2]) * 1.0 / len(result[1]) > sen_fp_sim_filter_threshold:
        #     continue
        # mmr_factor = 0.8
        # mmr_score = mmr_factor * result[3] - (1 - mmr_factor) * max_sim(idx,list(database_data.keys()),sim_matrix)
        # mmr_score = result[3] + len(result[1]) / max_len
        mmr_score = result[3]
        fps = result[2]
        if unlabeled_threshold < mmr_score and fps != 'null':
            best_sample = (result[0], result[1], result[2], str(mmr_score))
            best_idx = idx
            database_data[best_idx] = best_sample
            not_null_sum += 1
        if fps == 'null':
            null_samples.append((result[0], result[1], 'null', str(mmr_score)) )

    null_samples.sort(key=lambda x:float(x[3]), reverse=True)
    global null_sampling_threshold
    null_samples_with_high_confidence = [ sample for sample in null_samples if float(sample[3]) >= null_sampling_threshold]

    global null_sampling_num
    null_sampling_size = int(not_null_sum*null_sampling_num)
    database_null_samples = random.sample(null_samples_with_high_confidence,
                                          min(null_sampling_size, len(null_samples_with_high_confidence))
                                          )

    # results[best_idx] = None
    print('Iteration ' + model_name.split('_')[2] + ' generating new samples time: ' + str(
        time.time() - start_time) + 's')
    # print('New samples: ' + str(len(database_data) + len(extra_database_data)) + '\n\n')
    print('New samples: ' + str(len(database_data)) + '\n\n')
    samples_dao.del_all_iteration_samples()
    # print(list(database_data.values()))
    samples_dao.write_iteration_samples(list(database_data.values()))
    samples_dao.write_iteration_samples(database_null_samples)
    # samples_dao.write_iteration_samples(extra_database_data)


# semi-supervised training crf
def semi_supervised_data_function_extract(iteration_number):
    init_hyper_parameters()
    # divide origin data into seed and test data
    global last_performance

    global reset_data
    global training_percentage
    seeds, test_data = data_preparation.divide_samples_into_seeds_and_test(training_percentage, reset_data)
    temp_test = test_data
    some_seeds = []
    temp_test.extend(seeds)
    # get first model using seeds
    train_model(seeds, "../Archive/date_performance/models/model_0")
    # evaluate_model('./Archive/date_performance/models/model_0', test_data)
    # evaluate_model_by_story('../Archive/date_performance/models/model_0', test_data)
    global unlabeled_threshold
    global sample_num

    # iterate traning model
    for iteration_count in range(iteration_number):
        predicted_all_unlabeled_samples('../Archive/date_performance/models/model_' + str(iteration_count), sample_num)
        # get new training samples for next iteration
        iteration_samples = samples_dao.read_all_iteration_samples()
        iteration_samples.extend(seeds)
        update_iteration_story()
        # training next model
        train_model(iteration_samples, "../Archive/date_performance/models/model_" + str(iteration_count + 1))
        # evaluate_model('../Archive/date_performance/models/model_' + str(iteration_count + 1), test_data.extend(seeds))
        # evaluate_model_by_story('../Archive/date_performance/models/model_' + str(iteration_count + 1),temp_test)
        global unlabeled_threshold1
        if unlabeled_threshold-iter_sample_rate > unlabeled_threshold1:
            unlabeled_threshold -= iter_sample_rate
        # samples _num
        sample_num += sample_delta

    # evaluate_model('../Archive/date_performance/models/model_' + str(iteration_number), test_data)
    # evaluate_model_by_story('../Archive/date_performance/models/model_' + str(iteration_number), temp_test)


def train_model(train_samples, model_name):
    """"
    训练模型---全部数据拿来训练
    :param train_samples:  [DataSample1, DataSample2, ...] 训练数据
    :param model_name:  保存模型的模型
    :return: None
    """
    train = Trainer()

    # append training samples into trainer
    for sample in train_samples:
        xseq = build_model_features(sample, 17, True)
        # yseq = sample.label
        yseq = sample.char_label
        train.append(xseq, yseq)

    train.train(model_name)


def evaluate_model(model_name, test_samples):
    '''
    最后一次迭代训练模型，并输出测试结果
    :param test_samples:
    :param model_name:
    :return:
    '''
    model = Tagger()
    model.open(model_name)

    accuracy = 0.0
    recall = 0.0
    f1 = 0.0
    # sample_accuracy = 0.0
    iteration_test_details = []
    for sample in test_samples:
        model.set(build_model_features(sample, 17, True))
        predicted_labels = model.tag()
        true_labels = sample.char_label

        predicted_label_index = []
        for predicted_label in predicted_labels:
            if predicted_label == 'N':
                predicted_label_index.append(0)
            else:
                predicted_label_index.append(1)

        true_label_index = []
        for true_label in true_labels:
            if true_label == 'N':
                true_label_index.append(0)
            else:
                true_label_index.append(1)

        iteration_test_details = []
        chars = list(sample.sentence)
        # sen_words = sample.sen_words
        iteration_test_details.append(sample.sentence)
        predicted_fps = ''
        actual_fps = ''
        for index, word in enumerate(predicted_labels):
            if word != 'N':
                predicted_fps += chars[index]
        if len(predicted_fps) == 0:
            predicted_fps = '-----'

        for index, word in enumerate(true_labels):
            if word != 'N':
                actual_fps += chars[index]

        iteration_test_details.append(actual_fps)
        iteration_test_details.append(predicted_fps)

        with open('../Archive/date_performance/results/Iteration_Test_Details.csv', 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(iteration_test_details)

        # print(sample.sen_words)
        # print(predicted_labels)
        # print(true_labels)

        accuracy += metrics.accuracy_score(true_label_index, predicted_label_index)
        recall += metrics.recall_score(true_label_index, predicted_label_index, average='binary', pos_label=1)
        f1 += 2*accuracy*recall/(accuracy+recall)
        # sample_accuracy += metrics.sequence_accuracy_score(true_labels, predicted_labels)

    print("Iteration: %s\n\tAccuracy: %f\n\tRecall: %f\n\tF1: %f\n\n\n"
          % (
              model_name.split('_')[2], accuracy / len(test_samples), recall / len(test_samples),
              f1 / len(test_samples)))

    data = ["Iteration " + model_name.split('_')[2], accuracy / len(test_samples), recall / len(test_samples),
            f1 / len(test_samples)]

    with open('../Archive/date_performance/results/IterRes.csv', 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(data)

    with open('../Archive/date_performance/results/Iteration_Test_Details.csv', 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(data)

    return accuracy / len(test_samples), recall / len(test_samples), f1 / len(test_samples)


def evaluate_model_by_story(model_name, test_samples):
    model = Tagger()
    model.open(model_name)

    story_fps = dict()
    for sample in test_samples:
        model.set(build_model_features(sample, 17, True))
        predicted_labels = model.tag()

        chars = list(sample.sentence)
        predicted_fps = []
        fp = ''
        for index, word in enumerate(predicted_labels):
            if word == 'E' or word == 'S':
                fp += chars[index]
                predicted_fps.append(fp)
                fp = ''
            if word == 'B' or word == 'I':
                fp += chars[index]

        actual_fps = [fp for fp in sample.fps if fp != '' and fp != 'null' and fp in sample.sentence]

        filtered_predicted_fps = predicted_fps
        # for predicted_fp in predicted_fps:
        #     lan_confidence_temp = lmmodel.score(predicted_fp, bos=True, eos=True) / len(predicted_fp)
        #     if len(re.findall('[a-zA-Z0-9+]+', predicted_fp)) > 0:
        #         lan_confidence_temp += 5
        #     if lan_confidence_temp > -2.4:
        #         filtered_predicted_fps.append(predicted_fp)

        if sample.story_id not in story_fps:
            story_fps[sample.story_id] = [set(actual_fps), set(filtered_predicted_fps)]
        else:
            story_fps[sample.story_id][0].update(actual_fps)
            story_fps[sample.story_id][1].update(filtered_predicted_fps)

    # print(len(story_fps))
    global sim_t
    sim_threshold = sim_t

    TP_precision = 0
    TP_recall = 0
    all_actual_fps = 0
    all_predicted_fps = 0
    for story_id, (actual_fps, predicted_fps) in story_fps.items():
        story_precision = 0.0
        story_recall = 0.0

        all_actual_fps += len(actual_fps)

        all_predicted_fps += len(predicted_fps)
        # for actual_fp in actual_fps:

        story = samples_dao.read_story_by_story_id(int(story_id))
        data = [story_id,
                story[0] if story is not None else '',
                story[1] if story is not None else '',
                story[2] if story is not None else '',
                story[3] if story is not None else '',
                story[4] if story is not None else '',
                actual_fps,
                predicted_fps]
        with open('../Archive/date_performance/resultsIterRes_by_story_details.csv', 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(data)
        for predicted_fp in predicted_fps:
            sim = []
            for actual_fp in actual_fps:
                similarity = 1-distance.nlevenshtein(actual_fp, predicted_fp, method=1)
                # if actual_fp in predicted_fp:
                #     similarity = 1
                sim.append(similarity)
            # print(sim)

            if len(sim) == 0:
                sim = [0]
            if max(sim) >= sim_threshold:
                TP_precision += 1
                story_precision += 1

        for actual_fp in actual_fps:
            sim = []
            for predicted_fp in predicted_fps:
                similarity = 1-distance.nlevenshtein(actual_fp, predicted_fp, method=1)
                sim.append(similarity)
            # print(sim)
            if len(sim) == 0:
                sim = [0]
            if max(sim) >= sim_threshold:
                TP_recall += 1
                story_recall += 1

        # 每个故事的详情
        story_precision = 0 if len(filtered_predicted_fps) == 0 else story_precision/len(filtered_predicted_fps)
        story_recall = 0 if len(actual_fps) == 0 else story_recall/len(actual_fps)
        data = ["STORY " + story_id, story_precision, story_recall]
        with open('../Archive/date_performance/results/story_details.csv', 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(data)
    with open('../Archive/date_performance/results/story_details.csv', 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["THE END!!!"])

    # 整体的详情
    precision = TP_precision/all_predicted_fps
    recall = TP_recall/all_actual_fps
    f1 = 2 * precision * recall / (precision + recall)

    print("By Story: Iteration: %s\n\tPrecision: %f\n\tRecall: %f\n\tF1: %f\n\n\n"
          % (model_name.split('_')[2], precision, recall, f1))

    data = ["BY STORY: Iteration " + model_name.split('_')[2], precision, recall, f1]

    with open('../Archive/date_performance/results/IterRes_by_story.csv', 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(data)

    return precision, recall, f1


def get_all_labeled_data_lm_scores():
    all_samples = samples_dao.read_all_labeled_samples_by_story()
    all_story_count = 0
    all_fps_count = 0
    for samples in all_samples.values():
        all_fps = set()
        all_story_count += 1
        for sample in samples:
            fps = sample.fps
            for fp_name in fps:
                all_fps.add(fp_name)
                if len(fp_name) == 0 or fp_name == 'null':
                    continue
                data_fp_name = ' '.join(list(fp_name))
                lan_confidence_temp = -1
                if len(re.findall('[a-zA-Z0-9+]+', fp_name)) > 0:
                    lan_confidence_temp += 5

                # print("%s: %f" % (fp_name, lan_confidence_temp))
        all_fps_count += len(all_fps)

    print(all_story_count)
    print(all_fps_count)

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)

if __name__ == '__main__':
    # get_all_labeled_data_lm_scores()
    # get_all_labeled_data_lm_scores()
    # global lmmodel
    # lmmodel = kenlm.Model('lm/datafp.arpa')

    # init_hyper_parameters()
    #
    # iteration_num = md_config.getConfig('training', 'iteration_num')
    # mkdir('../Archive')
    # mkdir('../Archive/date_performance/')
    # mkdir('../Archive/date_performance/models/')
    # mkdir('../Archive/date_performance/results/')
    # semi_supervised_data_function_extract(iteration_num)


    # predicted_all_unlabeled_samples('./Archive/date_performance/models/model_23', 100000)
    # iteration_number = 100
    # semi_supervised_data_function_extract(iteration_number)
    # training, test = data_preparation.divide_samples_into_seeds_and_test(0.9, True)
    # training.extend(test)
    # evaluate_model_by_story('./Archive/date_performance/models/model_23', training)
    print('???')
