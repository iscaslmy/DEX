"""
评价指标
"""

import distance


def cal_PRF1(ground_truths, predicted_fps, similar_threshold):
    """
    计算两个列表的准确率、召回率、F1
    :param ground_truths:
    :param predicted_fps:
    :param similar_threshold: 相似度阈值
    :return: precison, recall, f1
    """
    TP_precision = 0
    TP_recall = 0

    for predicted_fp in predicted_fps:
        sim = []
        for actual_fp in ground_truths:
            similarity = adjusted_similarity(actual_fp, predicted_fp)
            sim.append(similarity)

        if len(sim) == 0:
            sim = [0]
        if max(sim) >= similar_threshold:
            TP_precision += 1

    for actual_fp in ground_truths:
        sim = []
        for predicted_fp in predicted_fps:
            similarity = adjusted_similarity(actual_fp, predicted_fp)
            sim.append(similarity)
        # print(sim)
        if len(sim) == 0:
            sim = [0]
        if max(sim) >= similar_threshold:
            TP_recall += 1

    precision = 0.0 if len(predicted_fps) == 0 else TP_precision / len(predicted_fps)
    # precision = 0.0 if len(predicted_fps) == 0 else TP_precision / min(len(predicted_fps), len(ground_truths))
    # precision = 1 if precision > 1 else precision
    recall = 0.0 if len(ground_truths) == 0 else TP_recall / len(ground_truths)
    f1 = 0.0 if precision == 0 or recall == 0 \
        else 2*precision*recall/(recall+precision)
    return precision, recall, f1


def adjusted_similarity(dfp1, dfp2):
    """
    计算两个数据功能项的相似度
    :param dfp1:
    :param dfp2:
    :return:
    """
    if dfp1 in dfp2 or dfp2 in dfp1:
        return 1

    return 1 - distance.nlevenshtein(dfp1, dfp2, method=1)

