import requests
import json
from sql_helper import SQLHelper

################### 数据准备 #######################
url_path = 'http://localhost:5002/m_data_preparation'

data = {'transaction_id': '7',
        'db_params': {'host':'localhost', 'name':'FP', 'user':'root', 'password':'656893aA',
                      'port':'3306', 'charset':'utf8'}
        }

requests.post(url_path.strip(),  data=json.dumps(data))


################### 交叉验证 #######################
url_path = 'http://localhost:5002/m_data_crossvalidation'
data = {'transaction_id': '0630_8',
        'db_params': {'host':'localhost', 'name':'FP', 'user':'root', 'password':'656893aA',
                      'port':'3306', 'charset':'utf8'},
        'model_infos' : {'dir': '0630'},
        'model_params': {'fold': '10',
                         'dict_threshold': '0.6',
                         'initial_confidence': '0.8', 'delta_confidence':'0.02', 'iteration_num':'5',
                         'null_sampling_rate': '1',
                         'evaluation_threshold': '0.6'}
        }

# requests.post(url_path.strip(),  data=json.dumps(data))


################### 模型训练 #######################
url_path = 'http://localhost:5002/m_data_train_model'

data = {'transaction_id': '0630_10',
        'db_params': {'host':'localhost', 'name':'FP', 'user':'root', 'password':'656893aA',
                      'port':'3306', 'charset':'utf8'},

        'model_infos': {'dir': '0630', 'name': 'crf_model_0630'},
        'model_params': {'fold': '10',
                         'dict_threshold': '0.6',
                         'initial_confidence': '0.8', 'delta_confidence':'0.02', 'iteration_num':'5',
                         'null_sampling_rate': '1',
                         'evaluation_threshold': '0.6'}
        }

# requests.post(url_path.strip(),  data=json.dumps(data))

verified_samples_ids = []
with open('/Users/limingyang/Downloads/temp_ids.txt', 'r', encoding='utf-8') as f:
    verified_samples_ids = [line.strip() for line in f.readlines()]

print(len(verified_samples_ids))

demands = set()
sql_helper111 = SQLHelper(dbHost='192.168.15.208', dbUser='foo', dbPasswd='123')
for id in verified_samples_ids:
    sql = 'select testRequirementCode from FP.fp_testdemand where id = %s' %id
    result = sql_helper111.select(sql)
    demands.add(result[0]['testRequirementCode'])

for demand in demands:
    print(demand)

