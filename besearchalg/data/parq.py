import os
import glob
import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
from wejoy_analysis.utils.log import get_simple_logger
import tqdm

logger = get_simple_logger('data.parq')

PARQ_ROOT = '/data/tinyv/parq'


CONFIG = {
    'dim_channel_d': {
        'path_dir': f'{PARQ_ROOT}/dim_channel_d',
        'is_overall': 1},
    'dim_dss_fields_d': {
        'path_dir': f'{PARQ_ROOT}/dim_dss_fields_d',
        'is_overall': 1},
    'dim_dss_facts_d': {
        'path_dir': f'{PARQ_ROOT}/dim_dss_facts_d',
        'is_overall': 1},
    'dim_decision_strategies_d': {
        'path_dir': f'{PARQ_ROOT}/dim_decision_strategies_d',
        'time_col': 'dt'
        },
    'e_flow_platform_daily_d': {
        'path_dir': f'{PARQ_ROOT}/e_flow_platform_daily_d',
        'time_col': 'dt'
        },
    'f_flow_daily_d': {
        'path_dir': f'{PARQ_ROOT}/f_flow_daily_d',
        'time_col': 'dt'
        },
    'e_user_weiliang_features_v2_i_api_credit': {
        'path_dir': f'{PARQ_ROOT}/e_user_weiliang_features_v2_i/api_credit',
        'time_col': 'credit_date'
        },
    'b_fuse_quota_amount_d_credit': {
        'path_dir': f'{PARQ_ROOT}/b_fuse_quota_amount_d/credit',
        },
    'e_user_base_credit_info_d': {
        'path_dir': f'{PARQ_ROOT}/e_user_base_credit_info_d',
        },
    'e_user_base_credit_info_d_register': {
        'path_dir': f'{PARQ_ROOT}/e_user_base_credit_info_d_register',
        'time_col': 'first_bind_mobile_succ_time'
        },
    'e_credit_apply_sku_core_d': {
        'path_dir': f'{PARQ_ROOT}/e_credit_apply_sku_core_d',
        },
    'e_credit_apply_core_d': {
        'path_dir': f'{PARQ_ROOT}/e_credit_apply_core_d',
        },
    'e_loan_info_core_d': {
        'path_dir': f'{PARQ_ROOT}/e_loan_info_core_d',
        },
    'e_repay_plan_core_d': {
        'path_dir': f'{PARQ_ROOT}/e_repay_plan_core_d',
        },
    'e_risk_feature_level_user_d': {
        'path_dir': f'{PARQ_ROOT}/e_risk_feature_level_user_d',
        },
    # 'e_user_base_loan_info_d': {
    #     'path_dir': f'{PARQ_ROOT}/e_user_base_loan_info_d',
    #     },
    'e_user_base_loan_info_d_credit': {
        'path_dir': f'{PARQ_ROOT}/e_user_base_loan_info_d/credit',
        },
    'e_user_base_loan_info_d_loan': {
        'path_dir': f'{PARQ_ROOT}/e_user_base_loan_info_d/loan',
        },
    'e_user_base_market_info_d_loan': {
        'path_dir': f'{PARQ_ROOT}/e_user_base_market_info_d/loan',
        },
    'e_user_base_market_info_d_credit': {
        'path_dir': f'{PARQ_ROOT}/e_user_base_market_info_d/credit',
        },
    'e_user_base_market_info_d_register': {
        'path_dir': f'{PARQ_ROOT}/e_user_base_market_info_d/register',
        },
    'b_fuse_decision_detail_logs_h_loan': {
        'path_dir': f'{PARQ_ROOT}/b_fuse_decision_detail_logs_h/loan',
        'dh': 1,
        },
    'b_fuse_decision_detail_logs_h_credit': {
        'path_dir': f'{PARQ_ROOT}/b_fuse_decision_detail_logs_h/credit',
        'dh': 1,
        },
    'b_fuse_decision_logs_h': {
        'path_dir': f'{PARQ_ROOT}/b_fuse_decision_logs_h',
        'dh': 1,
        },
    'b_fuse_decision_logs_d': {
        'path_dir': f'{PARQ_ROOT}/b_fuse_decision_logs_d',
        'time_col': 'create_time'},
    'e_risk_feature_d_resultdata': {
        'path_dir': f'{PARQ_ROOT}/e_risk_feature_d/resultdata_float',
        'dh': 1,
        'has_apply_month': 1},
    'e_risk_feature_d_resultdata_add': {
        'path_dir': f'{PARQ_ROOT}/e_risk_feature_d/resultdata_add_float',
        'dh': 1,
        'has_apply_month': 1},
}


def stat(tname):
    if tname not in CONFIG:
        logger.warning(f"CONFIG has not {tname}.")
    config = CONFIG[tname]
    is_overall = config.get('is_overall', 0)
    is_dh = config.get('dh', 0)
    path_dir = config['path_dir']
    files = glob.glob(os.path.join(path_dir, '*'))
    dts = pd.to_datetime([os.path.basename(x).strip('.parquet')
                         for x in files])
    ddf = dd.read_parquet(files, engine='pyarrow')
    return {
        'directory': path_dir,
        'start': min(dts),
        'end': max(dts),
        'files': len(files),
        # 'length': len(ddf),  # some table cost too many seconds
        'columns': [str(x) for x in ddf.columns.tolist()],
        'types': [str(x) for x in ddf.dtypes.values.tolist()]
    }


def wc(tname):
    if tname not in CONFIG:
        logger.warning(f"CONFIG has not {tname}.")
    config = CONFIG[tname]
    path_dir = config['path_dir']
    files = glob.glob(os.path.join(path_dir, '*'))
    if len(files) > 1000:
        logger.warning(
            f'There are too many files ({len(files)}), may cost more seconds...')
    ddf = dd.read_parquet(files, engine='pyarrow')
    return len(ddf)


def ls():
    return list(CONFIG.keys())


def read(tname, dt_st=None, dt_ed=None, filters=None, columns=None, renames=None, transforms=None, compute=True):
    """Short summary.

    Parameters
    ----------
    tname : type
        Description of parameter `tname`.
    dt_st : type
        Description of parameter `dt_st`.
    dt_ed : type
        Description of parameter `dt_ed`.
    filters : type
        Description of parameter `filters`.
    columns : type
        Description of parameter `columns`.
    renames : type
        Description of parameter `renames`.
    transforms : type
        Description of parameter `transforms`.
    compute : type
        Description of parameter `compute`.

    Returns
    -------
    type
        Description of returned object.

    """

    config = CONFIG[tname]
    is_overall = config.get('is_overall', 0)
    is_dh = config.get('dh', 0)
    path_dir = config['path_dir']
    if dt_st and dt_ed:
        dt_l = pd.date_range(dt_st, dt_ed)
        dts = [x.strftime('%Y-%m-%d') for x in dt_l]
    else:
        assert is_overall, "This table need dt_st and dt_ed."
    if is_overall:
        path_ = os.path.join(path_dir, f'*.parquet')
    elif is_dh:
        path_ = [os.path.join(path_dir, f'{x}-*.parquet') for x in dts]
    else:
        path_ = [os.path.join(path_dir, f'{x}.parquet') for x in dts]
    ddf = dd.read_parquet(path_, engine='pyarrow')
    if filters is not None:
        for k, v in filters.items():
            if isinstance(v, (list, np.ndarray, pd.Series)):
                ddf = ddf[ddf[k].isin(v)]
            elif isinstance(v, (str, int, float)):
                ddf = ddf[ddf[k] == v]
            else:
                ddf = ddf[ddf[k] == v]
    else:
        logger.warning(
            'Because filters is None, dask.dataframe.compute may take a long time.')
    if columns is not None:
        ddf = ddf[columns]
    if renames is not None:
        ddf = ddf.rename(columns=renames)
    if transforms is not None:
        for k, v in transforms.items():
            ddf[k] = v(ddf)
    if compute:
        return ddf.compute()
    else:
        return ddf


def get_wide_resultdata(dt_st, dt_ed, reg_brand_name, features):
    def0 = read('e_risk_feature_d_resultdata', dt_st, dt_ed)
    def1 = def0[def0['reg_brand_name'] == reg_brand_name]
    def1 = def1[['key', 'value', 'biz_type', 'biz_id']]
    def1 = def1[def1['key'].isin(features)]
    t0 = time.time()
    ef1 = def1.compute()
    t1 = time.time()
    logger.info(f'dask elapsed: {t1 - t0}')
    # t0 = time.time()
    # ef2 = ef1.pivot_table(index=['biz_type', 'biz_id'], columns='key', values='value', aggfunc='mean')
    # t1 = time.time()
    # logger.info(f'pivot elapsed: {t1 - t0}')
    # return ef2.reset_index(drop=True)
    t0 = time.time()
    v_d = {name: g.values for name, g in ef1.groupby('key')['value']}
    t1 = time.time()
    logger.info(f'to key: value elapsed: {t1 - t0}')
    return v_d
