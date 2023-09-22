"""
百融特征工程
"""
from model_tools.mymdl.feature_engineer import FeatureEngineer
from model_tools.mymdl import mdlutil


class ApplyLoanStr:
    """
    借贷意向验证接口
    """

    def __init__(self, feature_cols, feature_dict):
        self.feature_cols = feature_cols
        self.feature_dict = feature_dict
        self.feature_engineer = FeatureEngineer.build_cls(feature_cols, feature_dict)

    def add_features(self, df):
        # ================== 申请机构数-手机号==================#
        # 近7日
        df = self.feature_engineer.add_feature_sum(df, 'als_d7_cell_orgnum', '按手机号查询，近7天申请机构数',
                                                   ['als_d7_cell_bank_orgnum', 'als_d7_cell_nbank_orgnum'])

        # 近15日
        df = self.feature_engineer.add_feature_sum(df, 'als_d15_cell_orgnum', '按手机号查询，近15天申请机构数',
                                                   ['als_d15_cell_bank_orgnum', 'als_d15_cell_nbank_orgnum'])

        # 近1月
        df = self.feature_engineer.add_feature_sum(df, 'als_m1_cell_orgnum', '按手机号查询，近1月申请机构数',
                                                   ['als_m1_cell_bank_orgnum', 'als_m1_cell_nbank_orgnum'])

        # 近3月
        df = self.feature_engineer.add_feature_sum(df, 'als_m3_cell_orgnum', '按手机号查询，近3月申请机构数',
                                                   ['als_m3_cell_bank_orgnum', 'als_m3_cell_nbank_orgnum'])

        # 近6月
        df = self.feature_engineer.add_feature_sum(df, 'als_m6_cell_orgnum', '按手机号查询，近6月申请机构数',
                                                   ['als_m6_cell_bank_orgnum', 'als_m6_cell_nbank_orgnum'])

        # 近12月
        df = self.feature_engineer.add_feature_sum(df, 'als_m12_cell_orgnum', '按手机号查询，近12月申请机构数',
                                                   ['als_m12_cell_bank_orgnum', 'als_m12_cell_nbank_orgnum'])

        # 小贷机构数-新版统计方式
        df = self.feature_engineer.add_feature_sum(df, 'als_m1_cell_nbank_sloan_nsloan_orgnum',
                                                   '按手机号查询，近1月小贷机构数',
                                                   ['als_m1_cell_nbank_nsloan_orgnum',
                                                    'als_m1_cell_nbank_sloan_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m3_cell_nbank_sloan_nsloan_orgnum',
                                                   '按手机号查询，近3月小贷机构数',
                                                   ['als_m3_cell_nbank_nsloan_orgnum',
                                                    'als_m3_cell_nbank_sloan_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m6_cell_nbank_sloan_nsloan_orgnum',
                                                   '按手机号查询，近6月小贷机构数',
                                                   ['als_m6_cell_nbank_nsloan_orgnum',
                                                    'als_m6_cell_nbank_sloan_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m12_cell_nbank_sloan_nsloan_orgnum',
                                                   '按手机号查询，近12月小贷机构数',
                                                   ['als_m12_cell_nbank_nsloan_orgnum',
                                                    'als_m12_cell_nbank_sloan_orgnum'])

        # 近7天/近1月
        df = self.feature_engineer.add_feature_divide(df, 'als_d7_over_m1_cell_orgrate', '近7天申请机构占近1月比率',
                                                      'als_d7_cell_orgnum', 'als_m1_cell_orgnum')
        # 近3月/近6月
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_over_m6_cell_orgrate', '近3月申请机构占近6月比率',
                                                      'als_m3_cell_orgnum', 'als_m6_cell_orgnum')
        # 近6月/近12月
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_over_m12_cell_orgrate', '近6月申请机构占近12月比率',
                                                      'als_m6_cell_orgnum', 'als_m12_cell_orgnum')

        # 近3月非银机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_cell_nbank_orgrate', '近3月非银机构占比',
                                                      'als_m3_cell_nbank_orgnum', 'als_m3_cell_orgnum')

        # 近6个月非银机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_cell_nbank_orgrate', '近6月非银机构占比',
                                                      'als_m6_cell_nbank_orgnum', 'als_m6_cell_orgnum')

        # 近12月非银机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_cell_nbank_orgrate', '近12月非银机构占比',
                                                      'als_m12_cell_nbank_orgnum', 'als_m12_cell_orgnum')

        # 近3个月非银其他机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_cell_nbank_else_orgrate', '近3月非银其他机构占比',
                                                      'als_m3_cell_nbank_else_orgnum', 'als_m3_cell_orgnum')

        # 近6个月非银其他机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_cell_nbank_else_orgrate', '近6月非银其他机构占比',
                                                      'als_m6_cell_nbank_else_orgnum', 'als_m6_cell_orgnum')

        # 近12个月非银其他机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_cell_nbank_else_orgrate', '近12月非银其他机构占比',
                                                      'als_m12_cell_nbank_else_orgnum', 'als_m12_cell_orgnum')

        # 近3个月小贷机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_cell_nbank_sloan_nsloan_orgrate',
                                                      '近3月非银小贷机构占比',
                                                      'als_m3_cell_nbank_sloan_nsloan_orgnum', 'als_m3_cell_orgnum')
        # 近6个月小贷机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_cell_nbank_sloan_nsloan_orgrate',
                                                      '近6月非银小贷机构占比',
                                                      'als_m6_cell_nbank_sloan_nsloan_orgnum', 'als_m6_cell_orgnum')

        # 近12个月小贷机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_cell_nbank_sloan_nsloan_orgrate',
                                                      '近12月非银小贷机构占比',
                                                      'als_m12_cell_nbank_sloan_nsloan_orgnum', 'als_m12_cell_orgnum')

        # 近7日小额现金贷机构占比--老版
        df = self.feature_engineer.add_feature_divide(df, 'als_d7_cell_pdl_orgrate', '近7日小额现金机构占比',
                                                      'als_d7_cell_pdl_orgnum', 'als_d7_cell_orgnum')
        # 近3月小额现金贷机构占比--老版
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_cell_pdl_orgrate', '近3月小额现金机构占比',
                                                      'als_m3_cell_pdl_orgnum', 'als_m3_cell_orgnum')

        # 近3月非银小贷机构对于12月比率
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_over_m12_cell_nbank_sloan_nsloan_orgrate',
                                                      '近3月非银小贷机构对于12月比率',
                                                      'als_m3_cell_nbank_sloan_nsloan_orgnum',
                                                      'als_m12_cell_nbank_sloan_nsloan_orgnum')

        # 近7日其他机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_d7_cell_nbank_oth_orgrate', '近7日其他机构占比',
                                                      'als_d7_cell_nbank_oth_orgnum', 'als_d7_cell_orgnum')

        # 近3月其他机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_cell_nbank_oth_orgrate', '近3月其他机构占比',
                                                      'als_m3_cell_nbank_oth_orgnum', 'als_m3_cell_orgnum')

        # ================== 手机号查询，申请次数 ==================#

        # 近7日
        df = self.feature_engineer.add_feature_sum(df, 'als_d7_cell_allnum', '按手机号查询，近7天申请次数',
                                                   ['als_d7_cell_bank_allnum', 'als_d7_cell_nbank_allnum'])

        # 近15日
        df = self.feature_engineer.add_feature_sum(df, 'als_d15_cell_allnum', '按手机号查询，近15天申请次数',
                                                   ['als_d15_cell_bank_allnum', 'als_d15_cell_nbank_allnum'])

        # 近1月
        df = self.feature_engineer.add_feature_sum(df, 'als_m1_cell_allnum', '按手机号查询，近1月申请次数',
                                                   ['als_m1_cell_bank_allnum', 'als_m1_cell_nbank_allnum'])

        # 近3月
        df = self.feature_engineer.add_feature_sum(df, 'als_m3_cell_allnum', '按手机号查询，近3月申请次数',
                                                   ['als_m3_cell_bank_allnum', 'als_m3_cell_nbank_allnum'])

        # 近6月
        df = self.feature_engineer.add_feature_sum(df, 'als_m6_cell_allnum', '按手机号查询，近6月申请次数',
                                                   ['als_m6_cell_bank_allnum', 'als_m6_cell_nbank_allnum'])

        # 近12月
        df = self.feature_engineer.add_feature_sum(df, 'als_m12_cell_allnum', '按手机号查询，近12月申请次数',
                                                   ['als_m12_cell_bank_allnum', 'als_m12_cell_nbank_allnum'])

        # 其他申请次数占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_cell_oth_allrate',
                                                      '按手机号查询，近3月其他申请次数占比',
                                                      'als_m3_cell_oth_allnum', 'als_m3_cell_allnum')
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_cell_oth_allrate',
                                                      '按手机号查询，近6月其他申请次数占比',
                                                      'als_m6_cell_oth_allnum', 'als_m6_cell_allnum')
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_cell_oth_allrate',
                                                      '按手机号查询，近12月其他申请次数占比',
                                                      'als_m12_cell_oth_allnum', 'als_m12_cell_allnum')

        # 非银其他-老版
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_cell_nbank_oth_allrate',
                                                      '按手机号查询，近3月非银其他申请次数占比',
                                                      'als_m3_cell_nbank_oth_allnum', 'als_m3_cell_allnum')
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_cell_nbank_oth_allrate',
                                                      '按手机号查询，近6月非银其他申请次数占比',
                                                      'als_m6_cell_nbank_oth_allnum', 'als_m6_cell_allnum')
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_cell_nbank_oth_allrate',
                                                      '按手机号查询，近12月非银其他申请次数占比',
                                                      'als_m12_cell_nbank_oth_allnum', 'als_m12_cell_allnum')
        # 非银其他-新版
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_cell_nbank_else_allrate',
                                                      '新版—按手机号查询，近3月非银其他申请次数占比',
                                                      'als_m3_cell_nbank_else_allnum', 'als_m3_cell_allnum')
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_cell_nbank_else_allrate',
                                                      '新版—按手机号查询，近6月非银其他申请次数占比',
                                                      'als_m6_cell_nbank_else_allnum', 'als_m6_cell_allnum')
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_cell_nbank_else_allrate',
                                                      '新版—按手机号查询，近12月非银其他申请次数占比',
                                                      'als_m12_cell_nbank_else_allnum', 'als_m12_cell_allnum')

        # 近7天/近1月
        df = self.feature_engineer.add_feature_divide(df, 'als_d7_over_m1_cell_allrate',
                                                      '按手机号查询，近7天申请次数占近1月比率',
                                                      'als_d7_cell_allnum', 'als_m1_cell_allnum')
        # 近3月/近6月
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_over_m6_cell_allrate',
                                                      '按手机号查询，近3月申请次数占近6月比率',
                                                      'als_m3_cell_allnum', 'als_m6_cell_allnum')
        # 6/12
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_over_m12_cell_allrate',
                                                      '按手机号查询，近6月申请次数占近12月比率',
                                                      'als_m6_cell_allnum', 'als_m12_cell_allnum')

        # 新版小贷申请次数
        df = self.feature_engineer.add_feature_sum(df, 'als_d7_cell_nbank_sloan_nsloan_allnum',
                                                   '新版-按手机号查询，近7天非银-小贷申请次数',
                                                   ['als_d7_cell_nbank_nsloan_allnum',
                                                    'als_d7_cell_nbank_sloan_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_m1_cell_nbank_sloan_nsloan_allnum',
                                                   '新版-按手机号查询，近1月非银-小贷申请次数',
                                                   ['als_m1_cell_nbank_nsloan_allnum',
                                                    'als_m1_cell_nbank_sloan_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_m3_cell_nbank_sloan_nsloan_allnum',
                                                   '新版-按手机号查询，近3月非银-小贷申请次数',
                                                   ['als_m3_cell_nbank_nsloan_allnum',
                                                    'als_m3_cell_nbank_sloan_allnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m6_cell_nbank_sloan_nsloan_allnum',
                                                   '新版-按手机号查询，近6月非银-小贷申请次数',
                                                   ['als_m6_cell_nbank_nsloan_allnum',
                                                    'als_m6_cell_nbank_sloan_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_m12_cell_nbank_sloan_nsloan_allnum',
                                                   '新版-按手机号查询，近12月非银-小贷申请次数',
                                                   ['als_m12_cell_nbank_nsloan_allnum',
                                                    'als_m12_cell_nbank_sloan_allnum'])

        # 非银小贷占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_cell_nbank_sloan_nsloan_allrate',
                                                      '新版-按手机号查询，近3月非银-小贷申请占比',
                                                      'als_m3_cell_nbank_sloan_nsloan_allnum', 'als_m3_cell_allnum')
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_cell_nbank_sloan_nsloan_allrate',
                                                      '新版-按手机号查询，近6月非银-小贷申请占比',
                                                      'als_m6_cell_nbank_sloan_nsloan_allnum', 'als_m6_cell_allnum')
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_cell_nbank_sloan_nsloan_allrate',
                                                      '新版-按手机号查询，近12月非银-小贷申请占比',
                                                      'als_m12_cell_nbank_sloan_nsloan_allnum', 'als_m12_cell_allnum')

        # 消费分期申请次数占比-老版
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_cell_nbank_cf_allrate',
                                                      '按手机号查询，近3月消费分期申请次数占比',
                                                      'als_m3_cell_nbank_cf_allnum', 'als_m12_cell_allnum')
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_cell_nbank_cf_allrate',
                                                      '按手机号查询，近6月消费分期申请次数占比',
                                                      'als_m6_cell_nbank_cf_allnum', 'als_m12_cell_allnum')
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_cell_nbank_cf_allrate',
                                                      '按手机号查询，近12月消费分期申请次数占比',
                                                      'als_m12_cell_nbank_cf_allnum', 'als_m12_cell_allnum')

    def add_features_v2(self, df):
        """
        第二版：由于身份证号&手机号查询返回数字可能不一样，提取最大值
        :param df:
        :return:
        """
        # ================== 申请机构数==================#

        # 近7日
        df = self.feature_engineer.add_feature_sum(df, 'als_d7_cell_orgnum', '按手机号查询，近7天申请机构数',
                                                   ['als_d7_cell_bank_orgnum', 'als_d7_cell_nbank_orgnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_d7_id_orgnum', '按身份证号查询，近7天申请机构数',
                                                   ['als_d7_id_bank_orgnum', 'als_d7_id_nbank_orgnum'])
        # 提取身份证号&手机号的最大值
        df = self.feature_engineer.add_feature_max(df, 'als_d7_orgnum', '近7天申请机构数',
                                                   ['als_d7_cell_orgnum', 'als_d7_id_orgnum'])

        # 近15日
        df = self.feature_engineer.add_feature_sum(df, 'als_d15_cell_orgnum', '按手机号查询，近15天申请机构数',
                                                   ['als_d15_cell_bank_orgnum', 'als_d15_cell_nbank_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_d15_id_orgnum', '按身份证号查询，近15天申请机构数',
                                                   ['als_d15_id_bank_orgnum', 'als_d15_id_nbank_orgnum'])

        df = self.feature_engineer.add_feature_max(df, 'als_d15_orgnum', '近15天申请机构数',
                                                   ['als_d15_cell_orgnum', 'als_d15_id_orgnum'])

        # 近1月
        df = self.feature_engineer.add_feature_sum(df, 'als_m1_cell_orgnum', '按手机号查询，近1月申请机构数',
                                                   ['als_m1_cell_bank_orgnum', 'als_m1_cell_nbank_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m1_id_orgnum', '按身份证号查询，近1月申请机构数',
                                                   ['als_m1_id_bank_orgnum', 'als_m1_id_nbank_orgnum'])

        df = self.feature_engineer.add_feature_max(df, 'als_m1_orgnum', '近1月申请机构数',
                                                   ['als_m1_cell_orgnum', 'als_m1_id_orgnum'])

        # 近3月
        df = self.feature_engineer.add_feature_sum(df, 'als_m3_cell_orgnum', '按手机号查询，近3月申请机构数',
                                                   ['als_m3_cell_bank_orgnum', 'als_m3_cell_nbank_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m3_id_orgnum', '按身份证号查询，近3月申请机构数',
                                                   ['als_m3_id_bank_orgnum', 'als_m3_id_nbank_orgnum'])

        df = self.feature_engineer.add_feature_max(df, 'als_m3_orgnum', '近3月申请机构数',
                                                   ['als_m3_cell_orgnum', 'als_m3_id_orgnum'])

        # 近6月
        df = self.feature_engineer.add_feature_sum(df, 'als_m6_cell_orgnum', '按手机号查询，近6月申请机构数',
                                                   ['als_m6_cell_bank_orgnum', 'als_m6_cell_nbank_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m6_id_orgnum', '按身份证号查询，近6月申请机构数',
                                                   ['als_m6_id_bank_orgnum', 'als_m6_id_nbank_orgnum'])

        df = self.feature_engineer.add_feature_max(df, 'als_m6_orgnum', '近6月申请机构数',
                                                   ['als_m6_cell_orgnum', 'als_m6_id_orgnum'])

        # 近12月
        df = self.feature_engineer.add_feature_sum(df, 'als_m12_cell_orgnum', '按手机号查询，近12月申请机构数',
                                                   ['als_m12_cell_bank_orgnum', 'als_m12_cell_nbank_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m12_id_orgnum', '按身份证号查询，近12月申请机构数',
                                                   ['als_m12_id_bank_orgnum', 'als_m12_id_nbank_orgnum'])

        df = self.feature_engineer.add_feature_max(df, 'als_m12_orgnum', '近12月申请机构数',
                                                   ['als_m12_cell_orgnum', 'als_m12_id_orgnum'])

        # 小贷机构数-新版统计方式
        df = self.feature_engineer.add_feature_sum(df, 'als_m1_cell_nbank_sloan_nsloan_orgnum',
                                                   '按手机号查询，近1月小贷机构数',
                                                   ['als_m1_cell_nbank_nsloan_orgnum',
                                                    'als_m1_cell_nbank_sloan_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m1_id_nbank_sloan_nsloan_orgnum',
                                                   '按身份证号查询，近1月小贷机构数',
                                                   ['als_m1_id_nbank_nsloan_orgnum',
                                                    'als_m1_id_nbank_sloan_orgnum'])
        df = self.feature_engineer.add_feature_max(df, 'als_m1_nbank_sloan_nsloan_orgnum',
                                                   '近1月小贷机构数',
                                                   ['als_m1_cell_nbank_sloan_nsloan_orgnum',
                                                    'als_m1_id_nbank_sloan_nsloan_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m3_cell_nbank_sloan_nsloan_orgnum',
                                                   '按手机号查询，近3月小贷机构数',
                                                   ['als_m3_cell_nbank_nsloan_orgnum',
                                                    'als_m3_cell_nbank_sloan_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m3_id_nbank_sloan_nsloan_orgnum',
                                                   '按身份证号查询，近3月小贷机构数',
                                                   ['als_m3_id_nbank_nsloan_orgnum',
                                                    'als_m3_id_nbank_sloan_orgnum'])

        df = self.feature_engineer.add_feature_max(df, 'als_m3_nbank_sloan_nsloan_orgnum',
                                                   '近3月小贷机构数',
                                                   ['als_m3_cell_nbank_sloan_nsloan_orgnum',
                                                    'als_m3_id_nbank_sloan_nsloan_orgnum'])
        # 6
        df = self.feature_engineer.add_feature_sum(df, 'als_m6_cell_nbank_sloan_nsloan_orgnum',
                                                   '按手机号查询，近6月小贷机构数',
                                                   ['als_m6_cell_nbank_nsloan_orgnum',
                                                    'als_m6_cell_nbank_sloan_orgnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_m6_id_nbank_sloan_nsloan_orgnum',
                                                   '按身份证号查询，近3月小贷机构数',
                                                   ['als_m6_id_nbank_nsloan_orgnum',
                                                    'als_m6_id_nbank_sloan_orgnum'])

        df = self.feature_engineer.add_feature_max(df, 'als_m6_nbank_sloan_nsloan_orgnum',
                                                   '近6月小贷机构数',
                                                   ['als_m6_cell_nbank_sloan_nsloan_orgnum',
                                                    'als_m6_id_nbank_sloan_nsloan_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m12_cell_nbank_sloan_nsloan_orgnum',
                                                   '按手机号查询，近12月小贷机构数',
                                                   ['als_m12_cell_nbank_nsloan_orgnum',
                                                    'als_m12_cell_nbank_sloan_orgnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m12_id_nbank_sloan_nsloan_orgnum',
                                                   '按身份证号查询，近12月小贷机构数',
                                                   ['als_m12_id_nbank_nsloan_orgnum',
                                                    'als_m12_id_nbank_sloan_orgnum'])

        df = self.feature_engineer.add_feature_max(df, 'als_m12_nbank_sloan_nsloan_orgnum',
                                                   '近12月小贷机构数',
                                                   ['als_m12_cell_nbank_sloan_nsloan_orgnum',
                                                    'als_m12_id_nbank_sloan_nsloan_orgnum'])

        # 近7天/近1月
        df = self.feature_engineer.add_feature_divide(df, 'als_d7_over_m1_orgrate', '近7天申请机构占近1月比率',
                                                      'als_d7_orgnum', 'als_m1_orgnum')
        # 近3月/近6月
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_over_m6_orgrate', '近3月申请机构占近6月比率',
                                                      'als_m3_orgnum', 'als_m6_orgnum')
        # 近6月/近12月
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_over_m12_orgrate', '近6月申请机构占近12月比率',
                                                      'als_m6_orgnum', 'als_m12_orgnum')

        # 近3月非银机构占比
        df = self.feature_engineer.add_feature_max(df, 'als_m3_nbank_orgnum', '近3月非银机构数',
                                                   ['als_m3_cell_nbank_orgnum', 'als_m3_id_nbank_orgnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_nbank_orgrate', '近3月非银机构占比',
                                                      'als_m3_nbank_orgnum', 'als_m3_orgnum')

        # 近6个月非银机构占比
        df = self.feature_engineer.add_feature_max(df, 'als_m6_nbank_orgnum', '近6月非银机构数',
                                                   ['als_m6_cell_nbank_orgnum', 'als_m6_id_nbank_orgnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_cell_nbank_orgrate', '近6月非银机构占比',
                                                      'als_m6_nbank_orgnum', 'als_m6_orgnum')

        # 近12月非银机构占比
        df = self.feature_engineer.add_feature_max(df, 'als_m12_nbank_orgnum', '近12月非银机构数',
                                                   ['als_m12_cell_nbank_orgnum', 'als_m12_id_nbank_orgnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_nbank_orgrate', '近12月非银机构占比',
                                                      'als_m12_nbank_orgnum', 'als_m12_orgnum')

        # 近3个月非银其他机构占比
        df = self.feature_engineer.add_feature_max(df, 'als_m3_nbank_else_orgnum', '近3月非银其他机构数',
                                                   ['als_m3_cell_nbank_else_orgnum', 'als_m3_id_nbank_else_orgnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_nbank_else_orgrate', '近3月非银其他机构占比',
                                                      'als_m3_nbank_else_orgnum', 'als_m3_orgnum')

        # 近6个月非银其他机构占比
        df = self.feature_engineer.add_feature_max(df, 'als_m6_nbank_else_orgnum', '近6月非银其他机构数',
                                                   ['als_m6_cell_nbank_else_orgnum', 'als_m6_id_nbank_else_orgnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_nbank_else_orgrate', '近6月非银其他机构占比',
                                                      'als_m6_nbank_else_orgnum', 'als_m6_orgnum')

        # 近12个月非银其他机构占比
        df = self.feature_engineer.add_feature_max(df, 'als_m12_nbank_else_orgnum', '近12月非银其他机构数',
                                                   ['als_m12_cell_nbank_else_orgnum', 'als_m12_id_nbank_else_orgnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_nbank_else_orgrate', '近12月非银其他机构占比',
                                                      'als_m12_nbank_else_orgnum', 'als_m12_orgnum')

        # 近3个月小贷机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_nbank_sloan_nsloan_orgrate',
                                                      '近3月非银小贷机构占比',
                                                      'als_m3_nbank_sloan_nsloan_orgnum', 'als_m3_orgnum')
        # 近6个月小贷机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_nbank_sloan_nsloan_orgrate',
                                                      '近6月非银小贷机构占比',
                                                      'als_m6_nbank_sloan_nsloan_orgnum', 'als_m6_orgnum')

        # 近12个月小贷机构占比
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_nbank_sloan_nsloan_orgrate',
                                                      '近12月非银小贷机构占比',
                                                      'als_m12_nbank_sloan_nsloan_orgnum', 'als_m12_orgnum')

        # 小贷申请机构数
        df = self.feature_engineer.add_feature_max(df, 'als_m3_pdl_orgnum', '近3个月申请线上小额现金贷的机构数',
                                                   ['als_m3_cell_pdl_orgnum', 'als_m3_id_pdl_orgnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_pdl_orgrate', '近3个月申请线上小额现金贷的机构占比',
                                                      'als_m3_pdl_orgnum', 'als_m3_orgnum')

        df = self.feature_engineer.add_feature_max(df, 'als_m6_pdl_orgnum', '近6个月申请线上小额现金贷的机构数',
                                                   ['als_m6_cell_pdl_orgnum', 'als_m6_id_pdl_orgnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_pdl_orgrate', '近6个月申请线上小额现金贷的机构占比',
                                                      'als_m6_pdl_orgnum', 'als_m6_orgnum')

        df = self.feature_engineer.add_feature_max(df, 'als_m12_pdl_orgnum', '近12个月申请线上小额现金贷的机构数',
                                                   ['als_m12_cell_pdl_orgnum', 'als_m12_id_pdl_orgnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_pdl_orgrate', '近12个月申请线上小额现金贷的机构占比',
                                                      'als_m12_pdl_orgnum', 'als_m12_orgnum')




        # ===================申请次数===================#
        # 近7日
        df = self.feature_engineer.add_feature_sum(df, 'als_d7_cell_allnum', '按手机号查询，近7天申请次数',
                                                   ['als_d7_cell_bank_allnum', 'als_d7_cell_nbank_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_d7_id_allnum', '按身份证号查询，近7天申请次数',
                                                   ['als_d7_id_bank_allnum', 'als_d7_id_nbank_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_d7_allnum', '近7天申请次数',
                                                   ['als_d7_cell_allnum', 'als_d7_id_allnum'])

        # 近15日
        df = self.feature_engineer.add_feature_sum(df, 'als_d15_cell_allnum', '按手机号查询，近15天申请次数',
                                                   ['als_d15_cell_bank_allnum', 'als_d15_cell_nbank_allnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_d15_id_allnum', '按身份证号查询，近15天申请次数',
                                                   ['als_d15_id_bank_allnum', 'als_d15_id_nbank_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_d15_allnum', '近15天申请次数',
                                                   ['als_d15_cell_allnum', 'als_d15_id_allnum'])

        # 近1月
        df = self.feature_engineer.add_feature_sum(df, 'als_m1_cell_allnum', '按手机号查询，近1月申请次数',
                                                   ['als_m1_cell_bank_allnum', 'als_m1_cell_nbank_allnum'])

        df = self.feature_engineer.add_feature_sum(df, 'als_m1_id_allnum', '按身份证号查询，近1月申请次数',
                                                   ['als_m1_id_bank_allnum', 'als_m1_id_nbank_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_m1_allnum', '近1月申请次数',
                                                   ['als_m1_cell_allnum', 'als_m1_id_allnum'])

        # 近3月
        df = self.feature_engineer.add_feature_sum(df, 'als_m3_cell_allnum', '按手机号查询，近3月申请次数',
                                                   ['als_m3_cell_bank_allnum', 'als_m3_cell_nbank_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_m3_id_allnum', '按身份证号查询，近3月申请次数',
                                                   ['als_m3_id_bank_allnum', 'als_m3_id_nbank_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_m3_allnum', '近3月申请次数',
                                                   ['als_m3_cell_allnum', 'als_m3_id_allnum'])

        # 近6月
        df = self.feature_engineer.add_feature_sum(df, 'als_m6_cell_allnum', '按手机号查询，近6月申请次数',
                                                   ['als_m6_cell_bank_allnum', 'als_m6_cell_nbank_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_m6_id_allnum', '按身份证号查询，近6月申请次数',
                                                   ['als_m6_id_bank_allnum', 'als_m6_id_nbank_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_m6_allnum', '近6月申请次数',
                                                   ['als_m6_cell_allnum', 'als_m6_id_allnum'])

        # 近12月
        df = self.feature_engineer.add_feature_sum(df, 'als_m12_cell_allnum', '按手机号查询，近12月申请次数',
                                                   ['als_m12_cell_bank_allnum', 'als_m12_cell_nbank_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_m12_id_allnum', '按身份证号查询，近12月申请次数',
                                                   ['als_m12_id_bank_allnum', 'als_m12_id_nbank_allnum'])
        df = self.feature_engineer.add_feature_sum(df, 'als_m12_allnum', '近12月申请次数',
                                                   ['als_m12_cell_allnum', 'als_m12_id_allnum'])

        # 近7天/近1月
        df = self.feature_engineer.add_feature_divide(df, 'als_d7_over_m1_allrate',
                                                      '近7天申请次数占近1月比率',
                                                      'als_d7_allnum', 'als_m1_allnum')
        # 近3月/近6月
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_over_m6_allrate',
                                                      '近3月申请次数占近6月比率',
                                                      'als_m3_allnum', 'als_m6_allnum')
        # 6/12
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_over_m12_allrate',
                                                      '近6月申请次数占近12月比率',
                                                      'als_m6_allnum', 'als_m12_allnum')

        # 非银申请次数占比
        df = self.feature_engineer.add_feature_max(df, 'als_m3_nbank_allnum', '近3月在非银申请次数',
                                                   ['als_m3_cell_nbank_allnum', 'als_m3_id_nbank_allnum'])

        df = self.feature_engineer.add_feature_divide(df, 'als_m3_nbank_allrate',
                                                      '近3月非银申请次数占比',
                                                      'als_m3_nbank_allnum', 'als_m3_allnum')

        df = self.feature_engineer.add_feature_max(df, 'als_m6_nbank_allnum', '近6月在非银申请次数',
                                                   ['als_m6_cell_nbank_allnum', 'als_m6_id_nbank_allnum'])

        df = self.feature_engineer.add_feature_divide(df, 'als_m6_nbank_allrate',
                                                      '近6月非银申请次数占比',
                                                      'als_m6_nbank_allnum', 'als_m6_allnum')

        df = self.feature_engineer.add_feature_max(df, 'als_m12_nbank_allnum', '近12月在非银申请次数',
                                                   ['als_m12_cell_nbank_allnum', 'als_m12_id_nbank_allnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_nbank_allrate',
                                                      '近12月非银申请次数占比',
                                                      'als_m12_nbank_allnum', 'als_m12_allnum')

        # 非银其他-新版
        df = self.feature_engineer.add_feature_max(df, 'als_m3_nbank_else_allnum', '新版-近3月在非银其他申请次数',
                                                   ['als_m3_cell_nbank_else_allnum', 'als_m3_id_nbank_else_allnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_nbank_else_allrate',
                                                      '新版—近3月非银其他申请次数占比',
                                                      'als_m3_nbank_else_allnum', 'als_m3_allnum')

        df = self.feature_engineer.add_feature_max(df, 'als_m6_nbank_else_allnum', '新版-近6月在非银其他申请次数',
                                                   ['als_m6_cell_nbank_else_allnum', 'als_m6_id_nbank_else_allnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_nbank_else_allrate',
                                                      '新版—近6月非银其他申请次数占比',
                                                      'als_m6_nbank_else_allnum', 'als_m6_allnum')

        df = self.feature_engineer.add_feature_max(df, 'als_m12_nbank_else_allnum', '新版-近12月在非银其他申请次数',
                                                   ['als_m12_cell_nbank_else_allnum', 'als_m12_id_nbank_else_allnum'])

        df = self.feature_engineer.add_feature_divide(df, 'als_m12_nbank_else_allrate',
                                                      '新版—近12月非银其他申请次数占比',
                                                      'als_m12_nbank_else_allnum', 'als_m12_allnum')

        # 非银其他 3/6
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_over_m6_nbank_else_allrate',
                                                      '新版—近3月非银其他申请次数占近6个月比率',
                                                      'als_m3_nbank_else_allnum', 'als_m6_nbank_else_allnum')
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_over_m12_nbank_else_allrate',
                                                      '新版—近6月非银其他申请次数占近12个月比率',
                                                      'als_m6_nbank_else_allnum', 'als_m12_nbank_else_allnum')

        # 业务分类：小额现金贷 申请次数
        df = self.feature_engineer.add_feature_max(df, 'als_m3_pdl_allnum', '近3个月申请线上小额现金贷的申请次数',
                                                   ['als_m3_cell_pdl_allnum', 'als_m3_id_pdl_allnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m3_pdl_orgrate', '近3个月申请线上小额现金贷的机构占比',
                                                      'als_m3_pdl_orgnum', 'als_m3_orgnum')

        df = self.feature_engineer.add_feature_max(df, 'als_m6_pdl_orgnum', '近6个月申请线上小额现金贷的机构数',
                                                   ['als_m6_cell_pdl_orgnum', 'als_m6_id_pdl_orgnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m6_pdl_orgrate', '近6个月申请线上小额现金贷的机构占比',
                                                      'als_m6_pdl_orgnum', 'als_m6_orgnum')

        df = self.feature_engineer.add_feature_max(df, 'als_m12_pdl_orgnum', '近12个月申请线上小额现金贷的机构数',
                                                   ['als_m12_cell_pdl_orgnum', 'als_m12_id_pdl_orgnum'])
        df = self.feature_engineer.add_feature_divide(df, 'als_m12_pdl_orgrate', '近12个月申请线上小额现金贷的机构占比',
                                                      'als_m12_pdl_orgnum', 'als_m12_orgnum')

        return df


