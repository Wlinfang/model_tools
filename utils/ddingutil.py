"""
钉钉帮助文档 https://ding-doc.dingtalk.com/doc#/serverapi2/krgddi
"""
import requests
import time
import hmac
import hashlib
import urllib
import base64
import logging
import collections
logger = logging.getLogger()

ding_url = 'https://oapi.dingtalk.com/robot/send?access_token=0f07ca0875fe7199786cecbe4420912a94b88f1328a112a8e06ffc477dde8888'

class DingDing:
    def __init__(self,token,secret):
        self.__token = token
        self.__secret = secret
    def __update_sign(self):
        '''
        计算签名
        '''
        # 计算签名
        self.__timestamp = int(round(time.time()*1000))
        secret_enc = self.__secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(self.__timestamp, self.__secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        self.__sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

    def send_text(self,content,phones=[]):
        '''
        发送文本
        content:文本内容
        phones:指定发送的用户的手机号列表；['133','156']
        '''
        params = dict()
        params['msgtype'] = 'text'
        params['text'] = {'content':content}
        if not phones:
            params['at']={'atMobiles': phones}
        else:
            params['at'] = {'isAtAll': False}
        headers={
            'Content-Type':'application/json; charset=utf-8',
        }
        self.__update_sign()
        durl='https://oapi.dingtalk.com/robot/send?access_token={}&timestamp={}&sign={}'.format(
            self.__token,self.__timestamp,self.__sign)
        r = requests.post(url=durl,json=params,headers=headers)
        res = r.json()
        if res.get('errcode') == 0:
            return True
        else :
            logger.error('dingding send msg error %s',res)
            return False

    def send_picture(self,title,pic_url):
        '''
        发送图片
        pic_url:图片地址
        title:标题
        '''
        params = {
            'msgtype':'markdown',
            'markdown':{
                'title':title,
                'text':"#### {} \n ![fig]({})\n".format(title,pic_url),
            }
        }
        headers={
            'Content-Type':'application/json; charset=utf-8',
        }
        # 计算签名
        self.__update_sign()
        durl='https://oapi.dingtalk.com/robot/send?access_token={}&timestamp={}&sign={}'.format(
            self.__token,self.__timestamp,self.__sign)
        r = requests.post(url=durl,json=params,headers=headers)
        res = r.json()
        if res.get('errcode') == 0:
            return True
        else :
            logger.error('dingding send msg error %s',res)
            return False
