import numpy as np
import pandas as pd
from wejoy_analysis.component.data.parq import read, ls
from wejoy_analysis.component.analysis.find import join_df, simple_hash, any_to_num


def get_rn_unbalance_strategy(
    biz_ids: np.ndarray,
    biz_type,
    dt_st,
    dt_ed,
    dt_mid,
    strategy_stage_type=None
):
    """Short summary.

    Parameters
    ----------
    biz_ids : str
        '0': 不包含分流决策流, None: 取所有，不做过滤
    Returns
    -------
    type
        Description of returned object.

    """
    filters = {
        'biz_id': biz_ids,
        'biz_type': biz_type,
    }
    if strategy_stage_type is not None:
        filters['strategy_stage_type'] = strategy_stage_type
    df_log1 = read(
        'b_fuse_decision_logs_d',
        dt_st,
        dt_ed,
        filters=filters,
        columns=['decision_log_id', 'user_id', 'biz_id', 'biz_type', 'result_code', 'strategy_id', 'strategy_version', 'strategy_stage_type', 'create_time'],
        renames={
            'biz_id': f'{biz_type}_id',
        }
    )
    df_log1 = df_log1.drop_duplicates(['decision_log_id', f'{biz_type}_id'])
    df_log1['result_code'] = df_log1['result_code'].apply(any_to_num)
    df_log1['create_time'] = pd.to_datetime(df_log1['create_time'])
    dflg1a = df_log1[df_log1['create_time'] < pd.to_datetime(dt_mid)]
    dflg1b = df_log1[df_log1['create_time'] > pd.to_datetime(dt_mid)]
    dflg1g1 = dflg1a.groupby('strategy_id')['result_code'].mean().reset_index()
    r1a = dflg1a.groupby('strategy_id')['decision_log_id'].count() / dflg1a[f'{biz_type}_id'].nunique()
    dflg1g1['ratio'] = dflg1g1['strategy_id'].map(r1a)
    dflg1g2 = dflg1b.groupby('strategy_id')['result_code'].mean().reset_index()
    r1b = dflg1b.groupby('strategy_id')['decision_log_id'].count() / dflg1b[f'{biz_type}_id'].nunique()
    dflg1g2['ratio'] = dflg1g2['strategy_id'].map(r1b)
    dflg1x = pd.merge(dflg1g1, dflg1g2, on='strategy_id', how='outer')
    dflg1x['abs_delta_code'] = (dflg1x['result_code_y'] - dflg1x['result_code_x']).abs()
    dflg1x['abs_delta_ratio'] = (dflg1x['ratio_y'] - dflg1x['ratio_x']).abs()
    dflg1x = dflg1x.sort_values(['ratio_x', 'abs_delta_code', 'abs_delta_ratio'], ascending=False)
    return dflg1x[['strategy_id', 'result_code_x', 'result_code_y', 'ratio_x', 'ratio_y', 'abs_delta_code', 'abs_delta_ratio']]


def get_r3_unbalance_strategy(
    loan_ids: np.ndarray,
    dt_st,
    dt_ed,
    dt_mid
):
    df_log1 = read(
        'b_fuse_decision_logs_d',
        dt_st,
        dt_ed,
        filters={
            'biz_id': loan_ids,
            'biz_type': 'loan',
            'strategy_stage_type': '0'
        },
        columns=['decision_log_id', 'user_id', 'biz_id', 'biz_type', 'result_code', 'strategy_id', 'strategy_version', 'create_time'],
        renames={
            'biz_id': 'loan_id',
        }
    )
    df_log1 = df_log1.drop_duplicates(['decision_log_id', 'loan_id'])
    df_log1['result_code'] = df_log1['result_code'].apply(any_to_num)
    df_log1['create_time'] = pd.to_datetime(df_log1['create_time'])
    dflg1a = df_log1[df_log1['create_time'] < pd.to_datetime(dt_mid)]
    dflg1b = df_log1[df_log1['create_time'] > pd.to_datetime(dt_mid)]
    dflg1g1 = dflg1a.groupby('strategy_id')['result_code'].mean().reset_index()
    r1a = dflg1a.groupby('strategy_id')['decision_log_id'].count() / dflg1a['loan_id'].nunique()
    dflg1g1['ratio'] = dflg1g1['strategy_id'].map(r1a)
    dflg1g2 = dflg1b.groupby('strategy_id')['result_code'].mean().reset_index()
    r1b = dflg1b.groupby('strategy_id')['decision_log_id'].count() / dflg1b['loan_id'].nunique()
    dflg1g2['ratio'] = dflg1g2['strategy_id'].map(r1b)
    dflg1x = pd.merge(dflg1g1, dflg1g2, on='strategy_id', how='outer')
    dflg1x['abs_delta_code'] = (dflg1x['result_code_y'] - dflg1x['result_code_x']).abs()
    dflg1x['abs_delta_ratio'] = (dflg1x['ratio_y'] - dflg1x['ratio_x']).abs()
    dflg1x = dflg1x.sort_values(['ratio_x', 'abs_delta_code', 'abs_delta_ratio'], ascending=False)
    return dflg1x[['strategy_id', 'result_code_x', 'result_code_y', 'ratio_x', 'ratio_y', 'abs_delta_code', 'abs_delta_ratio']]


def get_strategy_unbalance_node(
    dt_st,
    dt_ed,
    dt_mid,
    biz_ids: np.ndarray,
    biz_type: str,
    strategy_id: int,
):
    dflog1 = read(
        'b_fuse_decision_logs_h',
        dt_st,
        dt_ed,
        filters={
            'biz_id': biz_ids,
            'biz_type': biz_type,
            'strategy_id': strategy_id
        },
        columns=['decision_log_id', 'user_id', 'biz_id', 'biz_type', 'strategy_id', 'strategy_version', 'create_time'],
    )

    dflog2 = read(
        f'b_fuse_decision_detail_logs_h_{biz_type}',
        dt_st,
        dt_ed,
        filters={
            'decision_log_id': dflog1['decision_log_id'].values,
            'category': 'node',
        },
        columns=['decision_log_id', f'{biz_type}_id', 'user_id', 'node_id', 'create_time', 'code']
    )
    dflog2 = dflog2.drop_duplicates(['decision_log_id', 'node_id'])
    dflog2['create_time'] = pd.to_datetime(dflog2['create_time'])
    dflg1a = dflog2[dflog2['create_time'] < pd.to_datetime(dt_mid)]
    dflg1b = dflog2[dflog2['create_time'] > pd.to_datetime(dt_mid)]
    dflg1g1 = dflg1a.groupby('node_id')['code'].mean().reset_index()
    r1a = dflg1a.groupby('node_id')['decision_log_id'].count() / dflg1a['decision_log_id'].nunique()
    dflg1g1['ratio'] = dflg1g1['node_id'].map(r1a)
    dflg1g2 = dflg1b.groupby('node_id')['code'].mean().reset_index()
    r1b = dflg1b.groupby('node_id')['decision_log_id'].count() / dflg1b['decision_log_id'].nunique()
    dflg1g2['ratio'] = dflg1g2['node_id'].map(r1b)
    dflg1x = pd.merge(dflg1g1, dflg1g2, on='node_id', how='outer')
    dflg1x['abs_delta_code'] = (dflg1x['code_y'] - dflg1x['code_x']).abs()
    dflg1x['abs_delta_ratio'] = (dflg1x['ratio_y'] - dflg1x['ratio_x']).abs()
    dflg1x = dflg1x.sort_values(['ratio_x', 'abs_delta_code', 'abs_delta_ratio'], ascending=False)
    return dflg1x[['node_id', 'code_x', 'code_y', 'ratio_x', 'ratio_y', 'abs_delta_code', 'abs_delta_ratio']]


def get_node_unbalance_fact(
    dt_st,
    dt_ed,
    dt_mid,
    biz_ids: np.ndarray,
    biz_type: str,
    strategy_id: int,
    node_id: int,
):
    dflog1 = read(
        'b_fuse_decision_logs_h',
        dt_st,
        dt_ed,
        filters={
            'biz_id': biz_ids,
            'biz_type': biz_type,
            'strategy_id': strategy_id
        },
        columns=['decision_log_id', 'user_id', 'biz_id', 'biz_type', 'strategy_id', 'strategy_version', 'create_time'],
    )
    dflog2 = read(
        f'b_fuse_decision_detail_logs_h_{biz_type}',
        dt_st,
        dt_ed,
        filters={
            'decision_log_id': dflog1['decision_log_id'].values,
            'category': 'fact',
            'node_id': node_id,
        },
        columns=['decision_log_id', f'{biz_type}_id', 'user_id', 'node_id', 'create_time', 'code', 'fact_id', 'value']
    )
    dflog2 = dflog2.drop_duplicates(['decision_log_id', 'node_id', 'fact_id'])
    dflog2['value'] = dflog2['value'].apply(any_to_num)
    dflog2['create_time'] = pd.to_datetime(dflog2['create_time'])
    dflg1a = dflog2[dflog2['create_time'] < pd.to_datetime(dt_mid)]
    dflg1b = dflog2[dflog2['create_time'] > pd.to_datetime(dt_mid)]
    dflg1g1 = dflg1a.groupby('fact_id')['value'].mean().reset_index()
    r1a = dflg1a.groupby('fact_id')['decision_log_id'].count() / dflg1a['decision_log_id'].nunique()
    dflg1g1['ratio'] = dflg1g1['fact_id'].map(r1a)
    dflg1g2 = dflg1b.groupby('fact_id')['value'].mean().reset_index()
    r1b = dflg1b.groupby('fact_id')['decision_log_id'].count() / dflg1b['decision_log_id'].nunique()
    dflg1g2['ratio'] = dflg1g2['fact_id'].map(r1b)
    dflg1x = pd.merge(dflg1g1, dflg1g2, on='fact_id', how='outer')
    dflg1x['abs_delta_value'] = (dflg1x['value_y'] - dflg1x['value_x']).abs()
    dflg1x['change_rate'] = (dflg1x['value_y'] - dflg1x['value_x']) / dflg1x['value_x']
    dflg1x['abs_delta_ratio'] = (dflg1x['ratio_y'] - dflg1x['ratio_x']).abs()
    dflg1x = dflg1x.sort_values(['ratio_x', 'change_rate', 'abs_delta_ratio'], ascending=False)
    dss_facts = read('dim_dss_facts_d', filters={})
    dflg1x['name'] = dflg1x['fact_id'].map(pd.Series(index=dss_facts['id'].values, data=dss_facts['name'].values))
    return dflg1x[['fact_id', 'name', 'value_x', 'value_y', 'ratio_x', 'ratio_y', 'change_rate', 'abs_delta_ratio']]
