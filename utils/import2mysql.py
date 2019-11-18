# -*- coding: utf-8 -*-
import re
from itertools import chain

import jieba
import xlrd

from samples_dao import write_data_samples_into_db, read_stories, write_unlabeled_samples_into_db


def cut_sent(para):
    if para is None:
        return []
    # para = re.sub('([,，;；。！？\?])([^”])', r"\1\n\2", para)  # 单字符断句符
    # para = re.sub('(\.{6})([^”])', r"\1\n\2", para)  # 英文省略号
    # para = re.sub('(\…{2})([^”])', r"\1\n\2", para)  # 中文省略号
    # para = re.sub(
    #     '((?<!\d)["（(“]{0,2}(?<!-|\+|/|\d)[\d１]{1,2}[\d\.\d]?[\s\.、，。,：．）)])|(((?<!\d)["0]?\d\-?<?!\d>))|([(（]?(?<!唯|之|统|第|景)[一二三四五][\s、．,，）)])|((?<=[)）,，：:。；？！;\?!\s])[abcdeABCDE][\s\.、，,：．）)])',
    #     r'\1\n\2', para)  # 序号
    # para = re.sub('(”)', '”\n', para)  # 把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = re.sub(r"\\s", "", para)
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    para = re.split(r",|，|。|：|:|；|;|\d[、\\.]|\n", para)
    para = [re.sub(r'【|】', "", s) for s in para]
    return [re.sub(r'[,.，。：:;；(\d、)(\d\\.)]$', '', s) for s in para]


def get_label(sid, smr, des, acc, fp):
    # print(smr, des,acc,fp)
    word_labels = []
    sens = set()
    sens.add(cut_sent(smr))
    sens.add(cut_sent(des))
    sens.add(cut_sent(acc))
    sens = list(chain(*sens))
    mask = [0 for _ in range(len(fp))]
    # print(sens, fp)
    for i in range(len(sens)):
        for j in range(len(fp)):
            if sens[i].__contains__(fp[j]):
                fp_words = jieba.lcut(fp[j])
                sens_words = jieba.lcut(sens[i])
                encode = ["N" for _ in range(len(sens_words))]
                start = sens[i].find(fp[j])
                start_idx = 0
                for s in sens_words:
                    if start >= len(s):
                        start -= len(s)
                        start_idx += 1
                    else:
                        break
                fp_len = len(fp_words)
                mask[j] = 1
                # print(len(encode), start_idx, fp_len)
                # print(sens_words)
                # print(fp_words)
                if len(fp_words) == 1:
                    encode[start_idx] = "S"
                elif len(fp_words) == 2:
                    encode[start_idx] = "B"
                    encode[start_idx + fp_len - 1] = "E"
                elif len(fp_words) > 2:
                    encode[start_idx] = "B"
                    encode[start_idx + fp_len - 1] = "E"
                    for k in range(1, len(fp_words) - 1):
                        encode[start_idx + k] = "I"
                for idx in range(len(encode)):
                    word_labels.append({'senid': i, 'sen': sens[i], 'word': sens_words[idx], 'label': encode[idx]})
            else:
                for w in jieba.lcut(sens[i]):
                    word_labels.append({'senid': i, 'sen': sens[i], 'word': w, 'label': 'N'})
    # if sum(mask) == 0:
    #     print(sid, sens, fp)
    return word_labels


def get_samples(sid, smr, des, acc, fp):
    res = []
    smr_sens = cut_sent(smr)
    sens = [smr_sens, cut_sent(des), cut_sent(acc)]
    sens = list(chain(*sens))
    sens = set(sens)
    for sen in sens:
        fp_str = [f for f in fp if sen.__contains__(f)]
        if len(fp_str) > 0:
            res.append((sid, sen, " ".join(fp_str)))
    # add sentences all labeled 'N' from summary
    for smr_sen in smr_sens:
        if smr_sen.__contains__("作为") or len(smr_sen) < 5:
            continue
        fp_str = [f for f in fp if smr_sen.__contains__(f)]
        if len(fp_str) == 0:
            res.append((sid, smr_sen, "null"))
            break
    return res


def run():
    data = xlrd.open_workbook("../DataFpMatchRatio_all_v3.xls")
    table = data.sheets()[0]
    res = []
    # print(table.row_values(0))
    for i in range(1, table.nrows):
        # print(i, table.row_values(i))
        if table.cell(i, -1).value == "" and len(table.cell(i, -2).value) > 0:
            sid = int(table.cell(i, 2).value)
            smr = str(table.cell(i, 3).value)
            des = str(table.cell(i, 4).value)
            acc = str(table.cell(i, 5).value)
            tcase = str(table.cell(i, 6).value)
            fp = str(table.cell(i, -2).value).split(" ")
            for n in range(len(fp)):
                if fp[n].strip() == "" or fp[n] is None:
                    del fp[n]
            res.append(get_samples(sid, smr, des, acc, fp))
    write_data_samples_into_db(list(chain(*res)))


def write_unlabeled_data():
    stories = read_stories()
    res = []
    for s in stories:
        sens = [cut_sent(s['sum']), cut_sent(s['des']), cut_sent(s['acc'])]
        sens = list(chain(*sens))
        sens = set(sens)
        for i in range(len(sens)):
            res.append((str(s['sid']), sens[i]))
    write_unlabeled_samples_into_db(res)


if __name__ == '__main__':
    # write_unlabeled_data()
    run()