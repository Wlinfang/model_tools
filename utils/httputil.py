import os
import mimetypes
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder


def http_post(url, file_paths=[], **kwargs):
    """
    post：提交数据，form-data 形式
    kwargs：表单字段
    """
    fields = {}
    if not kwargs:
        for key, value in kwargs.items():
            fields['key'] = value

    if not file_paths:
        file_list = []
        for file_path in file_paths:
            file_type, encoding = mimetypes.guess_type(file_path) or 'application/octet-stream'
            file_name = os.path.basename(file_path)
            file_list.append((file_name, open(file_name, 'rb'), file_type))
        fields['file'] = file_list
    m = MultipartEncoder(fields)
    response = requests.post(url, data=m, headers={'Content-Type': m.content_type})
    return response.text
