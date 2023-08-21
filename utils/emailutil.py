import os

import smtplib
import mimetypes

from email import encoders
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage


class EmailUtil:
    def __init__(self, host: str, port: int, from_user: str, from_passwd: str):
        """
        host:邮件服务器 mail.quantgroup.cn
        port:服务器端口
        from_user：发送人账户 datamonitor@quantgroup.cn
        from_passwd：发送人密码
        """
        self.__host = host
        self.__port = port
        self.__from_user = from_user
        self.__from_passwd = from_passwd

    def send_email(self, to: list, subject:str, content:str, cc=[], file_paths=[]):
        """
        to:接收邮件账户
        subject：邮件title
        content:邮件内容
        cc:抄送邮件账户
        file_paths：附件地址
        """
        mail_part = MIMEMultipart()
        mail_part['Subject'] = subject
        mail_part['From'] = self.__from_user
        mail_part['To'] = ';'.join(to)
        if not cc:
            mail_part['Cc'] = ';'.join(cc)
        # 邮件内容
        puretext = MIMEText(content, 'plain', 'utf-8')
        mail_part.attach(puretext)
        # 附件
        if not file_paths:
            for file_path in file_paths:
                if not os.path.isfile(file_path):
                    continue
                ctype, encoding = mimetypes.guess_type(file_path) or 'application/octet-stream'
                maintype, subtype = ctype.split('/', 1)
                if maintype == 'text':
                    with open(file_path,'rb') as fp:
                        msg = MIMEText(fp.read(), _subtype=subtype)
                elif maintype == 'image':
                    with open(file_path,'rb') as fp:
                        msg = MIMEImage(fp.read(), _subtype=subtype)
                elif maintype == 'audio':
                    with open(file_path,'rb') as fp:
                        msg = MIMEAudio(fp.read(), _subtype=subtype)
                else:
                    with open(file_path,'rb') as fp:
                        msg = MIMEBase(maintype, subtype)
                        msg.set_payload(fp.read())
                    # Encode the payload using Base64
                    encoders.encode_base64(msg)
                file_name=os.path.basename(file_path)
                msg.add_header('Content-Disposition', 'attachment', filename=file_name)
                mail_part.attach(msg)
        smtp = smtplib.SMTP(self.__host)
        try:
            smtp.connect(host=self.__host, port=self.__port)
            smtp.starttls()
            smtp.login(self.__from_user, self.__from_passwd)
            smtp.sendmail(self.__from_user, to, mail_part.as_string())
            smtp.quit()
        except Exception as e:
            raise Exception('---', str(e))