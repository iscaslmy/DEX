import random
import re

import xlrd

from sql_helper import SQLHelper
from DataSample import DataSample

sqlHelper = SQLHelper()


# 将sample存入数据库
# param: [(story_id, sentence_id, word, label), (),]---- 一定要按照序列的顺序传入
def write_data_samples_into_db(params):
    print(params)
    sql = 'insert into data_samples(story_id, sentence, fps) ' \
          'values(%s, %s, %s)'
    sqlHelper.insert_records_into_db(sql, params)


def write_unlabeled_samples_into_db(params):
    print(len(params))
    sql = 'insert into unlabeled_data_sample(story_id, sentence) ' \
          'values(%s, %s, %s)'
    sqlHelper.insert_records_into_db(sql, params)


# 将sample存入数据库下一轮需要训练的表中
# param: [(sid_senid, sentence, word, label), (),]---- 一定要按照序列的顺序传入
def write_data_samples_into_iteration_db(params):
    sql = 'insert into data_samples(story_id, sentence) ' \
          'values(%s, %s, %s, %s)'
    sqlHelper.insert_records_into_db(sql, params)


def read_stories():
    sql = "select * from story"
    db_results = sqlHelper.select(sql)
    res = []
    for r in db_results:
        s = dict()
        s['sid'] = r['ID']
        s['sum'] = r['summary']
        if re.search(r"[作做身]为.+", str(r['summary'])) is None:
            continue
        s['des'] = r['description']
        s['acc'] = r['acceptance']
        res.append(s)
    return res


####################################################################


# 读取所有有标记的数据
def read_all_labeled_samples_by_story():
    # key-story_id   value-list[DataSample1, DataSample2,...]
    samples = {}

    sql = 'select * from data_samples  where  sentence != fps and fps != \'null \''
    db_results = sqlHelper.select(sql)

    for result in db_results:
        story_id = result['story_id']
        sentence = result['sentence']
        fps = str(result['fps']).split()

        sample = DataSample(story_id, sentence, fps)

        if story_id in samples:
            samples[story_id].append(sample)
        else:
            samples[story_id] = [sample]

    return samples


def read_all_unlabeled_samples_by_story():
    samples = dict()

    db_results = read_all_unlabeled_samples()

    for sample in db_results:
        story_id = sample.story_id
        if story_id in samples:
            samples[story_id].append(sample)
        else:
            samples[story_id] = [sample]

    return samples


def read_all_unlabeled_samples():
    '''
    获取所有未标注的样本，形成初始数据
    :return:
    '''
    # final results
    results = []

    sql = 'select * from unlabeled_data_samples_all ' \
          'where source =\'summary\' and team in ' \
          '(select distinct project_id from FP.data_samples_with_project) '
    db_results = sqlHelper.select(sql)

    if db_results is None:
        return results

    for result in db_results:
        story_id = result['story_id']
        sentence = result['sentence']

        sample = DataSample(story_id=story_id, sentence=sentence)
        results.append(sample)

    return results

def read_unlabeled_samples(story_id):
    '''
    获取所有未标注的样本，形成初始数据
    :param stody_id
    :return:
    '''
    # final results
    results = []

    sql = 'select * from unlabeled_data_samples where story_id = \''+ story_id +'\''
    db_results = sqlHelper.select(sql)

    for result in db_results:
        sentence = result['sentence']
        results.append(sentence)

    return results

def read_all_iteration_samples():
    results = []

    sql = 'select * from iteration_data_samples_2'
    db_results = sqlHelper.select(sql)

    for result in db_results:
        story_id = result['story_id']
        sentence = result['sentence']
        fps = result['fps'].split()

        sample = DataSample(story_id, sentence, fps)
        results.append(sample)

    return results


def read_all_iteration_samples_by_story():
    samples = dict()
    db_results = read_all_iteration_samples()
    for sample in db_results:
        if sample.story_id in samples:
            samples[sample.story_id].append(sample)
        else:
            samples[sample.story_id] = [sample]
    return samples


def write_iteration_samples(samples):
    sql = 'insert into iteration_data_samples_2(story_id, sentence, fps, p) ' \
          'values(%s, %s, %s, %s)'
    sqlHelper.insert_records_into_db(sql, samples)


def del_all_iteration_samples():
    sql = 'delete from iteration_data_samples_2'
    sqlHelper.delete(sql)


def read_all_labeled_samples():
    results = []

    sql = 'select * from data_samples'
    db_results = sqlHelper.select(sql)

    for result in db_results:
        story_id = result['story_id']
        sentence = result['sentence']
        fps = result['fps'].split()

        sample = DataSample(story_id, sentence, fps)
        results.append(sample)

    return results


def read_all_data_fps():
    results = []

    sql = 'select * from history_data_fp'
    db_results = sqlHelper.select(sql)

    for result in db_results:
        data_fp = result['data_fp']

        results.append(data_fp)

    return results


def read_story_by_story_id(story_id):
    data = xlrd.open_workbook("../DataFpMatchRatio_all.xls")
    table = data.sheets()[0]
    for i in range(1, table.nrows):
        if table.cell(i, -1).value == "" and len(table.cell(i, -2).value) > 0:
            sid = int(table.cell(i, 2).value)
            if sid == story_id:
                smr = str(table.cell(i, 3).value) if str(table.cell(i, 3).value) is not None else ''
                des = str(table.cell(i, 4).value) if str(table.cell(i, 4).value) is not None else ''
                acc = str(table.cell(i, 5).value) if str(table.cell(i, 5).value) is not None else ''
                test = str(table.cell(i, 6).value) if str(table.cell(i, 6).value) is not None else ''
                fps = str(table.cell(i, 7).value) if str(table.cell(i, 7).value) is not None else ''
                return [smr, des, acc, test, fps]





def write_predicted_samples(samples):
    sql = 'insert into fp_m_recommend(story_id, systemName, functionName, functionType, functionStyle, functionScore) ' \
          'values(%s, %s, %s, %s, %s, %s)'
    sqlHelper.insert_records_into_db(sql, samples)