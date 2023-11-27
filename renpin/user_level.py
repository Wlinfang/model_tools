import numpy as np


class UserLevel:
    @classmethod
    def build_ul(cls):
        return cls()

    def get_user_level(self, *args):
        if len(args) == 3:
            os_type, flag_type, user_type = args
        else:
            os_type, flag_type, user_type, _ = args
        _user_l = {
            '0': {
                '0': {'A': 'BA-0', 'B': 'BA-0', 'C': 'BA-0', 'D': 'B-0', 'E': 'B-0', 'F': 'B-0'},
                '1': {'A': 'CA-0', 'B': 'CA-0', 'C': 'CA-0', 'D': 'C-0', 'E': 'C-0', 'F': 'C-0'},
                '2': {'A': 'AA-0', 'B': 'AA-0', 'C': 'AA-0', 'D': 'A-0', 'E': 'A-0', 'F': 'A-0'},
            },
            '1': {
                '0': {'A': 'BA-1', 'B': 'BA-1', 'C': 'BA-1', 'D': 'B-1', 'E': 'B-1', 'F': 'B-1'},
                '1': {'A': 'CA-1', 'B': 'CA-1', 'C': 'CA-1', 'D': 'C-1', 'E': 'C-1', 'F': 'C-1'},
                '2': {'A': 'AA-1', 'B': 'AA-1', 'C': 'AA-1', 'D': 'A-1', 'E': 'A-1', 'F': 'A-1'},
            }
        }
        return _user_l.get(str(os_type), {}).get(str(user_type), {}).get(str(flag_type), 'KA')


class UserLevelAlg:
    def predict_proba(self, X):
        try:
            ul = UserLevel.build_ul()
            args = tuple(np.array(X)[0])
            v = ul.get_user_level(*args)
            return {'code': 200, 'value': v}
        except ValueError:
            return {'code': 500, 'value': -9999998}


ula = UserLevelAlg()
from model_tools.utils import fileutil

# 保存模型
fileutil.save_model_as_pkl(ula, 'user_level_v1_str.pkl')

#=====test 加载模型 ====================================
ula = fileutil.load_by_joblib('user_level_v1_str.pkl')
input_cols=[
    'os_type',
    'flag_type',
    'user_type',
    'age',
]
values=['0','A','0',23]
import pandas as pd
X = pd.DataFrame([values], columns=input_cols)
ula.predict_proba(X)
