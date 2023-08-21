'''
新浪数科 wejoy_analysis
'''
import logging
import pathlib

# 提供了暴露当前模块对外暴露的 变量、函数、类
# from model import *  引用中只能引用 __all__ 中的
__all__ = ["get_simple_logger"]


DEFAULT_FORMAT = logging.Formatter(
    "[%(asctime)s, %(name)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
)


def get_simple_logger(name, path=None, flevel=logging.WARNING, clevel=logging.INFO):
    logger = logging.getLogger(name)
    formatter = DEFAULT_FORMAT

    ch = logging.StreamHandler()
    ch.setLevel(clevel)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if path is not None:
        fh = logging.FileHandler(path)
        fh.setLevel(flevel)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    return logger
