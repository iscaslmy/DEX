#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
from flask import Flask, request

from PT_Interface.model_prediction import predict_fps_pt

app = Flask(__name__)

from M_Interface.model_prediction import predict_fps_in_json
from M_Interface.model_preparation import data_preparation_interface
from M_Interface.model_training import crossvalidation_interface
from M_Interface.model_training import train_interface
from M_Interface.model_prediction import predict_fps_in_json_pt


@app.route('/crf_recommendations', methods=['GET', 'POST'])
def crf_recommendations():
    """
    :return: recommendations
    """
    if request.method == 'POST':
       # 获取Json数据
       source_data = request.get_data()
       # UTF-8解码
       decode_data = source_data.decode(encoding="utf-8")
       print(decode_data)
       # Json序列化
       story_ids = json.loads(decode_data)
       # 调用方法
       res = predict_fps_in_json(story_ids)
       # print(res)
       # 返回fpTosysCode方法处理结果
       return res


@app.route('/crf_recommendations_pt_story', methods=['GET', 'POST'])
def crf_recommendations_for_pt_stories():
    """
    :return: recommendations
    """
    if request.method == 'POST':
       # 获取Json数据
       source_data = request.get_data()
       # UTF-8解码
       decode_data = source_data.decode(encoding="utf-8")
       print(decode_data)
       # Json序列化
       story_ids = json.loads(decode_data)
       # 调用方法
       res = predict_fps_in_json_pt(story_ids)
       # print(res)
       # 返回fpTosysCode方法处理结果
       return res


# CRF算法服务调用地址http://服务地址:5002/crf_recommendations
@app.route('/crf_recommendations_pt', methods=['GET', 'POST'])
def crf_recommendations_pt():
    """
    crf推荐功能项
    :return: recommendations
    """
    if request.method == 'POST':
       # 获取Json数据
       source_data = request.get_data()
       # UTF-8解码
       decode_data = source_data.decode(encoding="utf-8")
       # Json序列化
       story_ids = json.loads(decode_data)
       # 调用方法
       res = predict_fps_pt(story_ids)
       # 返回fpTosysCode方法处理结果
       return res


"""
SERVICES FOR M_CRF
"""
# CRF算法服务调用地址http://服务地址:5002/crf_recommendations
@app.route('/m_data_preparation', methods=['GET', 'POST'])
def crf_m_data_preparation():
    """
    crf推荐功能项
    :return: recommendations
    """
    if request.method == 'POST':
       # 获取Json数据
       source_data = request.get_data()
       # UTF-8解码
       decode_data = source_data.decode(encoding="utf-8")
       # Json序列化
       dict = json.loads(decode_data)
       # 调用方法
       res = data_preparation_interface(dict)
       # 返回fpTosysCode方法处理结果
       return res


@app.route('/m_data_crossvalidation', methods=['GET', 'POST'])
def crf_m_crossvalidation():
    """
    crf交叉验证
    :return: recommendations
    """
    if request.method == 'POST':
       # 获取Json数据
       source_data = request.get_data()
       # UTF-8解码
       decode_data = source_data.decode(encoding="utf-8")
       # Json序列化
       dict = json.loads(decode_data)
       # 调用方法
       res = crossvalidation_interface(dict)
       # 返回fpTosysCode方法处理结果
       return res


@app.route('/m_data_train_model', methods=['GET', 'POST'])
def crf_m_train():
    """
    crf交叉验证
    :return: recommendations
    """
    if request.method == 'POST':
       # 获取Json数据
       source_data = request.get_data()
       # UTF-8解码
       decode_data = source_data.decode(encoding="utf-8")
       print(decode_data)
       # Json序列化
       dict = json.loads(decode_data)
       print(dict)
       # 调用方法
       res = train_interface(dict)
       # 返回fpTosysCode方法处理结果
       return res
"""
END SERVICES FOR M_CRF
"""

if __name__ == '__main__':
    # app.run(debug=True)
    # 服务端口
    port = int(os.environ.get("PORT", "5089"))
    # 服务地址
    app.run(host='127.0.0.1', port=port, debug=True)
