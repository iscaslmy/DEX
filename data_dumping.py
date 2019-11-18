import json
import os

from sql_helper import SQLHelper


sql_helper = SQLHelper(dbHost='192.168.15.208', dbUser='foo', dbPasswd='123', dbName='FP', dbPort=3306)
def dump_history_data_fps():
    """
    读取历史数据功能点的所有数据，用于过滤最后推荐的数据功能点
    :return: []
    """
    results = []
    global sql_helper
    sql = 'select data_fp from history_data_fp'
    db_results = sql_helper.select(sql)

    for result in db_results:
        fp = result['data_fp']
        results.append(fp)

    with open("data/M/historical_dfps.json", "w", encoding='utf-8') as f:
        json.dump(results, f)

    with open("data/PT/historical_dfps.json", "w", encoding='utf-8') as f:
        json.dump(results, f)
    print("加载入文件完成...")


dump_history_data_fps()
with open("data/M/historical_dfps.json", "r", encoding='utf-8') as f:
    dict = json.load(f)
    print(dict)
