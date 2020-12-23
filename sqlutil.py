from urllib.parse import urlparse
from pyhive import hive
import getpass
import pymongo
from sqlalchemy import create_engine

def get_hive()

username = 'wanglinfang'
hive_params = {
    'host': 'hive://10.96.23.44:25005/edw_tinyv',
    'auth_mechanism': 'LDAP',
    'user': username,
    'password': getpass.getpass(prompt=f"""Please input user {username}'s password:"""),
}
parsed_url = urlparse(hive_params['host'])

hive_engine = hive.Connection(host=parsed_url.hostname,
                                  port=parsed_url.port,
                                  auth="LDAP",
                                  username=hive_params['user'],
                                  password=hive_params['password'],
                                  database=parsed_url.path.strip('/'))

def get_impala():
	username='wanglinfang'
	impala_engine = sa.create_engine(
		'impala://10.96.23.44:25005/edw_tinyv',
		connect_args={
			'auth_mechanism': 'LDAP',
			'user': username,
			'password': getpass.getpass(prompt=f"""Please input user {username}'s password:""")
	})
	return impala_engine



def get_mongodb(usr_name,pwd,ip,port,db_name):
	myclient = pymongo.MongoClient("mongodb://{}:{}@{}:{}/?authSource={}".format(usr_name,pwd,ip,port,db_name))
	mydb=myclient[db_name]
	return mydb

def get_mysql_read(usr_name,pwd,ip,port,db_name):
	'''
	sqlalchemy: 支持pandas 写入数据库
	pymysql :   不支持pandas 写入数据库
	'''
	return create_engine("mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8".format(usr_name,pwd,ip,port,db_name))


