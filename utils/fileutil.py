import os
import json
from ruamel import yaml
from docx import Document
import configparser
import pickle
# PMML方式保存和读取模型
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from pypmml import Model


def save_model_as_pkl(model, path):
    """
    # Pickle方式保存和读取模型
    保存模型到路径path
    :param model: 训练完成的模型
    :param path: 保存的目标路径
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=4)


def load_model_from_pkl(path):
    """
    # Pickle方式保存和读取模型
    从路径path加载模型
    :param path: 保存的目标路径
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def save_model_as_pmml(alg, save_file_path):
    """
    # 以xgb模型为例，方式1：
    # sklearn接口的xgboost，可使用sklearn2pmml生成pmml文件
    保存模型到路径save_file_path
    :param x: 训练数据特征
    :param y: 训练数据标签
    :param save_file_path: 保存的目标路径
    """
    # # 设置pmml的pipeline
    # xgb = XGBClassifier(random_state=88)
    # mapper = DataFrameMapper([([i], None) for i in x.columns])
    # pipeline = PMMLPipeline([('mapper', mapper), ('classifier', xgb)])
    # # 模型训练
    # pipeline.fit(x, y)
    # 模型结果保存
    sklearn2pmml(alg, pmml=save_file_path, with_repr=True)


def load_model_from_pmml(load_file_path):
    """
    # PMML格式读取
    从路径load_file_path加载模型
    :param load_file_path: pmml文件路径
    """
    model = Model.fromFile(load_file_path)
    return model


def read_dict_conf(file_path: str) -> dict:
    """
    读取配置文件
    注意：python 2 版本的不兼容；
    file_path :配置文件路径
    配置文件格式：'''
    [section]
    key=value
    '''
    返回格式：{
    "section":{"key":value},
    "section2":{"key":value}
    }
    """
    if not os.path.isfile(file_path):
        raise ValueError('%s is not a file' % file_path)
    if not os.path.exists(file_path):
        raise ValueError('%s not exists ' % file_path)
    cp = configparser.ConfigParser()
    cp.read(file_path, encoding="utf-8")
    # 配置信息存入字典
    conf_dict = {}
    for part in cp.sections():
        conf_dict[part] = {}
        for item in cp.options(part):
            # item ->key
            try:
                # 如果是 dict 等形式，则进行转换
                conf_dict[part][item] = eval(cp.get(part, item))
            except Exception:
                # 如果是 float int str  list 形式
                conf_dict[part][item] = cp.get(part, item)
    return conf_dict


def generate_yaml_doc_ruamel(data: (dict, json), yaml_file: str):
    """
    将 data 转为 yaml 文件形式
    data：json or dict；data =
    yaml_file:输出的文件名
    """
    if isinstance(data, str):
        data = json.loads(data)
    with open(yaml_file, 'w', encoding='utf-8') as fp:
        yaml.dump(data, fp, Dumper=yaml.RoundTripDumper)


def get_yaml_data_ruamel(yaml_file):
    """
    读取yaml 文件
    yaml_file: 文件名
    """
    with open(yaml_file, 'r', encoding='utf-8') as fp:
        data = yaml.load(fp.read(), Loader=yaml.Loader)
    return data


class DocumentUtil:
    def __init__(self, file_path):
        self.__file_path = file_path
        file_name = os.path.basename(file_path)
        # 文件后缀名s
        file_prefex = file_name.rsplit('.', 1)[-1]
        if file_prefex not in ['doc', 'docx']:
            raise ValueError('{} is not a word file'.format(file_path))
        self.__document = Document(file_path)

    def get_document(self):
        return self.__document

    def save_document(self):
        self.__document.save(self.__file_path)

    def insert_table(self, cols, values):
        """
        cols: 列名
        values：值
        """
        table = self.__document.add_table(rows=1, cols=len(cols), style='Medium Grid 1 Accent 1')
        hdr_cells = table.rows[0].cells
        for i in range(len(cols)):
            hdr_cells[i].text = cols[i]
        for value in values:
            row_cells = table.add_row().cells
            for i in range(len(cols)):
                row_cells[i].text = str(value[i])
