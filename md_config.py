import configparser
import os

# 获取配置配置文件中的所有属性
def getConfig(section, key):
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + '/config.conf'
    config.read(path, encoding='utf-8')
    return config.get(section, key)