import os
import pandas as pd
import numpy as np
import datetime
import json

import sqlalchemy.engine
from pyhive import hive
import getpass
import pymongo
from sqlalchemy import create_engine

from pyhive import hive
import pyhdfs

class HiveEngine:
    @classmethod
    def build_hive_conn(cls, ip, port, db_name, user_name, passwd):
        """
        获取hive 连接
        """
        if user_name is None:
            hive_con = hive.Connection(host=ip, port=port)
        else:
            hive_con = hive.Connection(host=ip, port=port,
                                       auth="LDAP", username=user_name, database=db_name, password=passwd)
        instan = HiveEngine()
        instan.__hive_con = hive_con
        return instan

    def query_sql(self, sql):
        cursor = self.__hive_con.cursor()
        # 执行Hive查询
        cursor.execute(sql)
        # 获取查询结果
        results = cursor.fetchall()
        cols = np.array(cursor.description)[:, 0].tolist()
        df = pd.DataFrame.from_records(results, columns=cols)
        # 关闭连接
        cursor.close()
        return df

    def close_hive(self):
        self.__hive_conn.close()


def get_hive_engine(ip, port, db_name, user_name, passwd):
    """
    获取hive 连接
    密码：通过远程获取；或者使用加密的代码中再进行解密使用
    """
    # hive_params = {
    #     'host': 'hive://10.96.23.44:25005/edw_tinyv',
    #     'auth_mechanism': 'LDAP',
    #     'user': username,
    #     'password': getpass.getpass(prompt=f"""Please input user {username}'s password:"""),
    # }
    # parsed_url = urlparse(hive_params['host'])

    hive_engine = hive.Connection(host=ip,
                                  port=port,
                                  auth="LDAP",
                                  username=user_name,
                                  password=passwd,
                                  database=db_name)
    return hive_engine


def get_impala_engine(ip, port, db_name, user_name, passwd):
    """
    获取impala 连接
    """
    # impala_engine = create_engine(
    #     f'impala://{ip}:{port}/{db_name}',
    #     connect_args={
    #         'auth_mechanism': 'LDAP',
    #         'user': user_name,
    #         'password': getpass.getpass(prompt=f"""Please input user {user_name}'s password:""")
    # })
    impala_engine = create_engine(
        f'impala://{ip}:{port}/{db_name}',
        connect_args={
            'auth_mechanism': 'LDAP',
            'user': user_name,
            'password': passwd
        })
    return impala_engine


class MySqlEngine:
    """
    sqlalchemy: 支持pandas 写入数据库
    pymysql :   不支持pandas 写入数据库
    """

    def __init__(self, ip, port, db_name, user_name, passwd, echo=False):
        """
        给出数据库名字，创建数据库连接
        config_file:数据库配置文件地址
        echo: 是否打印sql
        """
        self.__engine = create_engine(
            'mysql+mysqldb://{user_name}:{passwd}@{ip}:{port}/{db_name}?charset=utf8'.format(
                ip=ip, port=port, user_name=user_name, passwd=passwd, db_name=db_name),
            echo=echo)

    def get_engine(self):
        return self.__engine

    def get_sql_in(self, sql: str, ids: list, step=5000) -> pd.DataFrame:
        """
        分段查询sql in 操作
        sql: select  * from test where id in {}
        """
        res = []
        for i in range(0, len(ids), step):
            tmp = pd.read_sql(sql=sql.format(str(tuple(ids[i:i + step])).replace(',)', ')')), con=self.__engine)
            res.append(tmp)
        return pd.concat(res)


class MongoClient(object):
    def __init__(self, ip, port, db_name, user_name, passwd):
        self.__mongo_uri = "mongodb://{}:{}@{}:{}/?authSource={}".format(
            user_name, passwd, ip, port, db_name)
        self.__client = pymongo.MongoClient(self.__mongo_uri)
        self.__db = self.__client[db_name]

    def query_db(self, collecton_name, query_con: dict, show_cols: list) -> pd.DataFrame:
        """
        查询mongo
        collecton_name: mongodb 中的表名
        :param query_con: dict，查询条件
        :param show_cols:  dict，返回的字段
        """
        collecton = self.__db[collecton_name]
        # 转为dict
        show_cols = {x: 1 for x in show_cols}
        show_cols['_id'] = 0
        cursor = collecton.find(filter=query_con, projection=show_cols)
        data = list(cursor)
        return pd.DataFrame(data)

    def query_in(self, collecton_name, show_cols: list, in_name: str, in_list,
                 step=5000) -> pd.DataFrames:
        """
        对应in查询
        collecton_name:mongodb 表名
        :param show_cols: dict，返回的字段
        :param in_name: 查询的 in 的列名
        :param in_list: 查询的 in 的 ids
        :param step:
        show_cols = {
                'A': 1,
                'B': 1,
        }
        example:
        query_in(show_cols=show_cols,
                in_list=('a', 'b', 'c'), in_name='_id')
        """
        res = []
        for i in range(0, len(in_list), step):
            query = {
                in_name: {'$in': in_list[i:i + step]}
            }
            res.append(self.query_db(collecton_name, query, show_cols))
        return pd.concat(res)

    def mongo_export(self, collecton_name, query, fields, mongo_path='/home/public/mongo_data'):
        """
        根据筛选条件从mongo导出数据
        :param query: json/json_str
        :param fields: list/str
        :param mongo_path:
        返回导出文件名
        """
        today = datetime.date.today().strftime('%Y-%m-%d')
        today_path = os.path.join(mongo_path, today)
        if not os.path.exists(today_path):
            os.makedirs(name=today_path)
        file_name = os.path.join(today_path, '{}_{}.csv'.format(today, os.getpid()))
        mongo_export = """\
        mongoexport --uri {uri} --collection {collection} --fields {fields} --query '{query}' --type {type} --out {out_file}
        """.format(
            uri=self.__mongo_uri,
            collection=collecton_name,
            fields=','.join(fields),
            query=json.dumps(query),
            type='csv',
            out_file=file_name)
        os.system(mongo_export)
        return file_name


def save_data_to_hive(spark, df: pd.DataFrame, view_name):
    """
    利用spark 将数据存储到 hive 临时表
    view_name:临时表名
    """
    if not spark:
        # spark = SparkSession.builder.appName('wlf').enableHiveSupport().config("hive.exec.dynamic.partition", "true").config('spark.default.parallelism','5000').config('spark.sql.shuffle.partitions','5000').config("hive.exec.dynamic.partition.mode", "nonstrict").getOrCreate()
        # if 临时视图存在，则返回True；如果不存在，则返回False
        spark.catalog.dropTempView(view_name)
        spark_df = spark.createDataFrame(df).repartition(1)
        spark_df.createOrReplaceTempView(view_name)


def mysql_query(sql, engine_mysql: sqlalchemy.engine.Engine) -> pd.DataFrame:
    """
    查询大量数据
    :param sql:sql 语句
    :param engine_mysql:查询器
    """
    res = []
    # == palo 每次查询不超过10000
    tmp = pd.read_sql(sql, engine_mysql, chunksize=5001)
    for tt in tmp:
        res.append(tt)
    return pd.concat(res)

# spark
# spark_df.withColumn('label_fst7',F.UserDefinedFunction(lambda obj: 1 if obj >= 7 else 0)(spark_df.fst_overdue_day))

def write_df_to_hdfs(hdfs_path,df):
    """
    利用hive 将dataframe 写入 hdfs_path
    """
    pass
