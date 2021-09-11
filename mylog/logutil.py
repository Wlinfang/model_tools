import logging.config
import json
import os

class MyLogging:

    def __init__(self):
        path='config.json'
        if os.path.exists(path):
            with open(path,'r') as f:
                cf=json.load(f)
                logging.config.dictConfig(cf)
        else:
            logging.basicConfig(level=logging.INFO)

    def get_logger(self,name):
        return logging.getLogger(name)