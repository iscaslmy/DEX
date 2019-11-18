import pymysql
import md_config


class SQLHelper:

    def __init__(self,
                 dbHost=md_config.getConfig('db', 'dbhost'),
                 dbName=md_config.getConfig('db', 'dbname'),
                 dbUser=md_config.getConfig('db', 'dbuser'),
                 dbPasswd=md_config.getConfig('db', 'dbpasswd'),
                 dbPort=int(md_config.getConfig('db', 'dbport')),
                 dbCharset=md_config.getConfig('db', 'dbcharset')):
        SQLHelper.dbHost = dbHost
        SQLHelper.dbName = dbName
        SQLHelper.dbUser = dbUser
        SQLHelper.dbPasswd = dbPasswd
        SQLHelper.dbPort = dbPort
        SQLHelper.dbCharset = dbCharset

    # 获取数据库连接
    @staticmethod
    def getconn():
        try:
            conn = pymysql.connect(host=SQLHelper.dbHost,
                                   user=SQLHelper.dbUser,
                                   passwd=SQLHelper.dbPasswd,
                                   db=SQLHelper.dbName,
                                   port=SQLHelper.dbPort,
                                   charset=SQLHelper.dbCharset)

        except pymysql.Error as e:
            print("MySQLDBError: %s" % e)
            conn = None

        return conn

    # 查询结果,返回结果是一个字典
    def select(self, sql):
        conn = self.getconn()
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)

        try:
            cursor.execute(sql)
            results = cursor.fetchall()
        except pymysql.Error:
            print("查询数据库出现错误")
            results = None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        return results

    # 数据库删除方法
    def delete(self, sql):
        conn = self.getconn()
        cursor = conn.cursor()

        try:
            cursor.execute(sql)
            conn.commit()
        except pymysql.Error:
            conn.rollback()
            print("删除数据库出现错误")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    # 将多条结果插入数据库
    # sql: 插入语句基本模版
    # params: 插入参数  列表[(),()]
    def insert_records_into_db(self, sql, params):
        conn = self.getconn()

        try:
            cursor = conn.cursor()
            cursor.executemany(sql, params)
            conn.commit()
        except Exception as e:
            print(e)
            conn.rollback()
        finally:
            if (cursor):
                cursor.close()
            if (conn):
                conn.close()


# 创建表,返回结果是一个字典
    def create(self, sql):
        conn = self.getconn()
        cursor = conn.cursor()

        try:
            cursor.execute(sql)
        except pymysql.Error:
            print("数据库创建表出现错误")
            results = None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

