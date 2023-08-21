import pathlib
import hashlib
import numbers
import numpy as np
import pandas as pd
import talib
import bottleneck as bn
import matplotlib.pyplot as plt
from wejoy_analysis.component.algorithm.beamsearch import BeamSearchLift, BeamSearchMask, get_top_lift_chain, get_stat_of_chain_end
import dask.dataframe as dd
import tqdm


def simple_hash(s, base=10000):
    return int(hashlib.md5(s.encode('utf8')).hexdigest(),16) % base

def any_to_num(x):
    if x is None:
        ret = float('nan')
    elif isinstance(x, numbers.Number):
        ret = float(x)
    elif len(x) == 0:
        ret = float('nan')
    else:
        try:
            ret = float(x)
        except:
            ret = float(simple_hash(x.lower()))
    return ret


def join_df(df1, df2, how='inner', index=['date', 'id']):
    cols_to_use = df2.columns.difference(df1.columns)
    cols_to_use = list(set(cols_to_use).union(set(index)))
    return pd.merge(df1, df2[cols_to_use], how=how, on=index)


def to_frame(ddf, num_workers=16):
    if isinstance(ddf, dd.DataFrame):
        return ddf.compute()
    elif isinstance(ddf, pd.DataFrame):
        return ddf
    else:
        raise Exception("Unkown")


def flatmap_has_or_nan(df, index_cols, key):
    l = []
    for name, g in tqdm.tqdm(df[index_cols+[key]].groupby(index_cols)):
        d = {str(x): 1.0 for x in g[key].to_list()}
        d = {**{k: name[i] for i, k in enumerate(index_cols)}, **d}
        l.append(d)
    return pd.DataFrame(l)


def get_days_in_month(dt_str_l):
    from collections import defaultdict
    d_in_m = defaultdict(list)
    for dt_str in dt_str_l:
        y, m, d = dt_str.split('-')
        d_in_m[f'{y}_{m}'].append(d)
    d_in_m = {k: ','.join(v) for k, v in d_in_m.items()}
    return d_in_m


def text_days_in_month(dt_str_l):
    d_in_m = get_days_in_month(dt_str_l)
    return '\n'.join([f'{k}: {v}' for k, v in d_in_m.items()])


def flatnan_drill_down(df_y, ddf_x, y_ids, x_ids, key, out_ids, dt1l, dt2l, df_hint=None, selected=3):
    return flatmap_drill_down(df_y, ddf_x, y_ids, x_ids, key, out_ids, dt1l, dt2l, df_hint=df_hint, selected=selected, mode='nan')


def flatmap_fact_to_num(df, index_cols):
    l = []
    for name, g in tqdm.tqdm(df[index_cols+['fact_id', 'value']].groupby(index_cols)):
        d = {str(fid): any_to_num(v) for fid, v in zip(g['fact_id'].values.tolist(), g['value'].values.tolist())}
        d = {**{k: name[i] for i, k in enumerate(index_cols)}, **d}
        l.append(d)
    return pd.DataFrame(l)


def flatmap_drill_down(df_y, ddf_x, y_ids, x_ids, key, out_ids, dt1l, dt2l, df_hint=None, selected=3, mode='nan'):
    # OUTPUT: chain_l_d, df_hint_next
    # y_ids = ['loan_id', 'user_id']
    # x_ids = ['loan_id', 'user_id']
    # out_ids = ['loan_id', 'user_id', 'decision_log_id']
    # key = 'strategy_id'
    # df_hint = None
    logger.info(f'dask pre filter start.')
    if df_hint is not None:
        logger.info(f'counting len_before_filter_by_hint...')
        len_before_filter_by_hint = len(ddf_x)
        df_x_raw = ddf_x
        for id in x_ids:
            df_x_raw = df_x_raw[df_x_raw[id].isin(df_hint[id].to_list())]
        df_x_raw = to_frame(df_x_raw)
        logger.info(f'counting len_after_filter_by_hint...')
        len_after_filter_by_hint = len(df_x_raw)
        logger.info(f'len_before_filter_by_hint: {len_before_filter_by_hint}, len_after_filter_by_hint: {len_after_filter_by_hint}')
    else:
        df_x_raw = to_frame(ddf_x)
    if mode == 'nan':
        logger.info(f'flatmap_has_or_nan start.')
        df_x = flatmap_has_or_nan(df_x_raw, x_ids, key)
    elif mode == 'fact':
        logger.info(f'flatmap_fact_to_num start.')
        df_x = flatmap_fact_to_num(df_x_raw, x_ids)
    logger.info(f'join_df: df_y, df_x start.')
    df = join_df(df_y, df_x, index=y_ids)
    logger.info(f'fillna start.')
    nan_cols = []
    for col in df.columns.to_list():
        if (~df[col].isnull()).sum() == 0:
            nan_cols.append(col)
    has_cols = [x for x in df.columns.to_list() if x not in nan_cols]

    df = df[has_cols]
    if mode != 'fact':
        df = df.fillna(0.0)
    features = [x for x in df.columns if x not in df_y.columns]
    features = [x for x in features if x not in x_ids]
    features = [x for x in features if x not in y_ids]
    fea_d = {k: df[k].values for k in features}
    y = df['y'].values
    logger.info(f'len(features): {len(features)}, len(has_cols): {len(has_cols)}, len(nan_cols): {len(nan_cols)}')

    dt_mask1 = df['datetime'].dt.strftime("%Y-%m-%d").isin(dt1l)
    dt_mask2 = df['datetime'].dt.strftime("%Y-%m-%d").isin(dt2l)

    y[dt_mask1].mean()
    y[dt_mask2].mean()

    logger.info(f'get_top_lift_chain start.')
    BeamSearchLift.set_dt_mask(dt_mask1, dt_mask2)
    top_contrib_cols = features
    min_population = 100
    beam_width = len(features)
    MAX_level = 1
    if mode == 'fact':
        split_mode = 'pd_prop_diff'
    else:
        split_mode = 'right_w_pd_lift_diff'
    chain_l_d = get_top_lift_chain(fea_d, y, top_contrib_cols, min_population, beam_width, MAX_level, split_mode=split_mode)
    level = 0
    out_l = list()
    real_selected = min(selected, len(chain_l_d[level]))
    logger.info(f'real_selected: {real_selected}, selected: {selected}, len(chain_l_d[level]): {len(chain_l_d[level])}')
    for i in range(real_selected):
        key_x = chain_l_d[level][i].name
        out_l.append({
            'key_x': key_x,
            'chain_end': chain_l_d[level][i],
            'df_hint_next': df_x_raw[df_x_raw[key].astype(str) == str(key_x)][out_ids].reset_index(drop=True),
            'fea_d': fea_d,
            'y': y
        })
    detail = {
        'df': df,
        'dt_mask1': dt_mask1,
        'dt_mask2': dt_mask2,
    }
    return out_l, detail


def get_loan_stat(dt1l, dt2l, group_code, category, df_hint_former, e_loan_info_core_d_dir, decision_logs_dir):
    dts = sorted(list(set(dt1l).union(set(dt2l))))
    files_y = [e_loan_info_core_d_dir.format(x) for x in dts]
    ddf = dd.read_parquet(files_y, engine='pyarrow')
    ddf = ddf[ddf['group_code'] == group_code]  # quyong_general_dae
    df = ddf.compute(num_workers=16)
    df_y = df.reset_index(drop=True)
    df_y['audit_status'] = df_y['audit_status'].astype('int64')
    df_y['y'] = df_y['audit_status'].map({2002005: 1, 2002006: 0})
    df_y['y'] = 1 - df_y['y']
    df_y['datetime'] = pd.to_datetime(df_y['apply_time'])

    if category == 'strategy':
        files_x = [decision_logs_dir.format(x) for x in dts]
        ddf_x = dd.read_parquet(files_x, engine='pyarrow')
        ddf_x = ddf_x[ddf_x['biz_type'] == 'loan']
        ddf_x = ddf_x.rename(columns={'biz_id': 'loan_id'})
        ddf_x = ddf_x[ddf_x['strategy_version_type'] == 'N']
        ddf_x = ddf_x[ddf_x['strategy_stage_type'].astype(int) == 0] # 决策strategy, 非分流

        y_ids = ['loan_id', 'user_id']
        x_ids = ['loan_id', 'user_id']
        key = 'strategy_id'
        out_ids = ['loan_id', 'user_id', 'decision_log_id']
        df_hint = df_y[['loan_id', 'user_id']]
        out1_l, detail1 = flatnan_drill_down(df_y, ddf_x, y_ids, x_ids, key, out_ids, dt1l, dt2l, df_hint=df_hint, selected=100)
    else:
        file_l = [decision_logs_dir.format(x) for x in dts]
        ddf2_x = dd.read_parquet(file_l, engine='pyarrow')
        ddf2_x = ddf2_x[ddf2_x['category'] == category]
        y2_ids = ['loan_id', 'user_id']
        x2_ids = ['loan_id', 'user_id', 'decision_log_id']
        key2 = f'{category}_id'
        out2_ids = ['loan_id', 'user_id', 'decision_log_id', key2]
        df_hint_out1 = df_hint_former
        if category == 'fact':
            mode = 'fact'
        else:
            mode = 'nan'
        out1_l, detail1 = flatmap_drill_down(df_y, ddf2_x, y2_ids, x2_ids, key2, out2_ids, dt1l, dt2l, df_hint=df_hint_out1, selected=100, mode=mode)
    tb_stat_l = []
    for i in range(len(out1_l)):
        tb_stat_l.append(get_stat_of_chain_end(out1_l[i]['chain_end'], out1_l[i]['fea_d'], out1_l[i]['y']))
    return out1_l, detail1, tb_stat_l


def plot_loan_stat_per_day(dt1l, dt2l, title_name, tb_stat_l, show_len):
    stat_l = [{'name': tb['test'][0]['name'], **stat} for tb, stat in tb_stat_l[:show_len]]
    df_stat = pd.DataFrame(stat_l)
    df_stat1 = df_stat[['name', 'bad_rate_test', 'bad_rate_base', 'sample_ratio_test', 'sample_ratio_base']]
    df_stat2 = df_stat[['name', 'sample_number_test', 'sample_number_base']]
    df_stat2['sample_number_test_per_day'] = df_stat2['sample_number_test'] / len(dt1l)
    df_stat2['sample_number_base_per_day'] = df_stat2['sample_number_base'] / len(dt2l)

    df_stat1.plot.bar(x='name', y=['bad_rate_test', 'bad_rate_base', 'sample_ratio_test', 'sample_ratio_base'])
    title_str = f'{title_name}\n test: {text_days_in_month(dt1l)}\n base:  {text_days_in_month(dt2l)}'
    plt.title(title_str)
    plt.show()

    df_stat2.plot.bar(x='name', y=['sample_number_test_per_day', 'sample_number_base_per_day'])
    title_str = f'{title_name}\n test: {text_days_in_month(dt1l)}\n base:  {text_days_in_month(dt2l)}'
    plt.title(title_str)

def get_resultdata_small_pivot_table(keys, path, biz_type, group_code):
    if biz_type == 'loan':
        biz_id_name = 'loan_id'
    elif biz_type == 'credit':
        biz_id_name = 'credit_id'
    logger.info(f'get_loan_small_pivot_table start: {path}')
    NOT_FEATURE_COLS = ['user_id', biz_id_name, 'biz_type', 'reg_brand_name', 'biz_code', 'group_code', 'apply_time']
    ddf_x = dd.read_parquet(path, engine='pyarrow')
    ddf_x = ddf_x[ddf_x['biz_type'] == biz_type]
    ddf_x = ddf_x[ddf_x['group_code'] == group_code]
    ddf_x = ddf_x[ddf_x['key'].isin(keys)]
    ddf_x = ddf_x.rename(columns={'biz_id': biz_id_name})
    df_x = ddf_x.compute()
    df_x_w = df_x.pivot_table(index=NOT_FEATURE_COLS, columns='key', values='value', aggfunc='mean')
    df_x_w = df_x_w.reset_index()
    df_x_w['user_id'] = df_x_w['user_id'].astype('int64')
    df_x_w[biz_id_name] = df_x_w[biz_id_name].astype('int64')
    logger.info(f'get_loan_small_pivot_table done: {path}')
    return df_x_w

def get_loan_feature_stat(dt1l, dt2l, group_code, business_table_dir, attribute_table_dir, reg_brand_name=None, used_keys=None):
    NOT_FEATURE_COLS = ['user_id', 'loan_id', 'biz_type', 'reg_brand_name', 'biz_code', 'group_code', 'apply_time']
    dts = sorted(list(set(dt1l).union(set(dt2l))))
    files_y = [business_table_dir.format(x) for x in dts]
    ddf = dd.read_parquet(files_y, engine='pyarrow')
    ddf = ddf[ddf['group_code'] == group_code]
    if reg_brand_name:
        ddf = ddf[ddf['reg_brand_name'] == reg_brand_name]
    df = ddf.compute()
    df_y = df.reset_index(drop=True)
    df_y['audit_status'] = df_y['audit_status'].astype('int64')
    df_y['y'] = df_y['audit_status'].map({2002005: 1, 2002006: 0})
    df_y['y'] = 1 - df_y['y']
    df_y['datetime'] = pd.to_datetime(df_y['apply_time'])
    files_x = [attribute_table_dir.format(x) for x in dts]
    ddf_x = dd.read_parquet(files_x, engine='pyarrow')
    ddf_x = ddf_x[ddf_x['biz_type'] == 'loan']
    ddf_x = ddf_x[ddf_x['group_code'] == group_code]
    if reg_brand_name:
        ddf_x = ddf_x[ddf_x['reg_brand_name'] == reg_brand_name]
    ddf_x = ddf_x.rename(columns={'biz_id': 'loan_id'})
    if not used_keys:
        logger.info(f'counting used features...')
        key_count = ddf_x.groupby('key').count().compute()
        keys = key_count[key_count['value'] > key_count.max().max() // 10].index.tolist()
        keys = [x for x in keys if x not in NOT_FEATURE_COLS]
        logger.info(f'all features: {len(key_count)}, choosed {len(keys)} features most used.')
    else:
        keys = used_keys
    # ddf_x = ddf_x[ddf_x['key'].isin(keys)]
    # ddf_x = ddf_x.rename(columns={'biz_id': 'loan_id'})
    # logger.info(f'loading {len(keys)} features...')
    # df_x = ddf_x.compute()
    # logger.info(f'df_x.shape: {df_x.shape}, pivot_table start...')
    # df_x_w = df_x.pivot_table(index=NOT_FEATURE_COLS, columns='key', values='value', aggfunc='mean')
    # df_x_w = df_x_w.reset_index()
    # df_x_w['user_id'] = df_x_w['user_id'].astype('int64')
    # df_x_w['loan_id'] = df_x_w['loan_id'].astype('int64')
    # df_x = df_x_w
    df_l = []
    for p in files_x:
        df_l.append(get_resultdata_small_pivot_table(keys, p, 'loan', group_code))
    logger.info(f'concat df_l start...')
    df_x_w = pd.concat(df_l, axis=0).reset_index(drop=True)
    logger.info(f'concat df_l done.')
    df_x = df_x_w
    logger.info(f'before join_df, df_y.shape: {df_y.shape}, df_x_w.shape: {df_x_w.shape}')
    df = join_df(df_y, df_x, index=['user_id', 'loan_id', 'apply_time'])
    logger.info(f'after join_df, df.shape: {df.shape}')

    features = keys
    fea_d = {k: df[k].values for k in features}
    y = df['y'].values
    dt_mask1 = df['datetime'].dt.strftime("%Y-%m-%d").isin(dt1l)
    dt_mask2 = df['datetime'].dt.strftime("%Y-%m-%d").isin(dt2l)
    BeamSearchLift.set_dt_mask(dt_mask1, dt_mask2)
    top_contrib_cols = features
    min_population = 100
    beam_width = len(features)
    MAX_level = 1
    split_mode = 'pd_prop_diff'
    logger.info(f'get_top_lift_chain start...')
    chain_l_d = get_top_lift_chain(fea_d, y, top_contrib_cols, min_population, beam_width, MAX_level, split_mode=split_mode)
    logger.info(f'done.')
    detail = {
        'fea_d': fea_d,
        'y': y,
        'dt_mask1': dt_mask1,
        'dt_mask2': dt_mask2,
    }
    return chain_l_d, detail


def plot_chain_contrib_values(chain_l, topk=20, title='feature strange rank'):
    _topk = min(topk, len(chain_l))
    df = pd.DataFrame(index=list(range(_topk)))
    df['feature_name'] = [x.name for x in chain_l[:_topk]]
    df['contrib_value'] = [x.abs_contrib for x in chain_l[:_topk]]
    df.plot.bar(x='feature_name', y='contrib_value', title=title)


def get3df2part(x, y, m_left, m_right, dt_mask1, dt_mask2):
    m_nan = np.isnan(x)
    df_count = pd.DataFrame(columns=['left', 'right', 'nan'])
    df_bad = pd.DataFrame(columns=['left', 'right', 'nan'])
    df_total_time_part = pd.DataFrame(columns=['left', 'right', 'nan'])
    df_count.loc['base', :] = [(dt_mask2 & m_left).sum(), (dt_mask2 & m_right).sum(), (dt_mask2 & m_nan).sum()]
    df_count.loc['test', :] = [(dt_mask1 & m_left).sum(), (dt_mask1 & m_right).sum(), (dt_mask1 & m_nan).sum()]
    df_bad.loc['base', :] = [np.nansum(y[dt_mask2 & m_left]), np.nansum(y[dt_mask2 & m_right]), np.nansum(y[dt_mask2 & m_nan])]
    df_bad.loc['test', :] = [np.nansum(y[dt_mask1 & m_left]), np.nansum(y[dt_mask1 & m_right]), np.nansum(y[dt_mask1 & m_nan])]
    df_total_time_part.loc['base', :] = [dt_mask2.sum(), dt_mask2.sum(), dt_mask2.sum()]
    df_total_time_part.loc['test', :] = [dt_mask1.sum(), dt_mask1.sum(), dt_mask1.sum()]
    df_count_on_time_part = df_count / df_total_time_part
    df_bad_on_count = df_bad / df_count
    df_bad_on_time_part = df_bad / df_total_time_part
    return {'df_count_on_time_part': df_count_on_time_part,
            'df_bad_on_count': df_bad_on_count,
            'df_bad_on_time_part': df_bad_on_time_part,
            'df_bad': df_bad,
            'df_count': df_count,
            'df_total_time_part': df_total_time_part}


def plot_y_x_bin(y, x, dt_mask1, dt_mask2, bins=10, fact_id_str='fact', mode='qcut'):
    box_num = bins
    title = fact_id_str
    df = pd.DataFrame({
        'y': np.array(y),
        fact_id_str: np.array(x),
    })
    if mode == 'qcut':
        box_label = pd.qcut(df[fact_id_str], box_num, labels=[i for i in range(box_num)])
        box_box = pd.qcut(df[fact_id_str], box_num)
    elif mode == 'cut':
        box_label = pd.cut(df[fact_id_str], box_num, labels=[i for i in range(box_num)])
        box_box = pd.cut(df[fact_id_str], box_num)
    box_cats = box_box.cat.categories.tolist()

    df['bin'] = box_label

    df_show = pd.DataFrame(index=[i for i in range(box_num + 1)])

    y_mean_mask1 = df[dt_mask1].groupby('bin')['y'].mean().tolist()
    y_mean_mask1.append(df[dt_mask1 & df['bin'].isnull()]['y'].mean())
    y_sum_mask1 = (df[dt_mask1].groupby('bin')['y'].sum()).tolist()
    y_sum_mask1.append(df[dt_mask1 & df['bin'].isnull()]['y'].sum())
    bin_ratio_mask1 = (df[dt_mask1].groupby('bin')['y'].count() / len(df[dt_mask1])).tolist()
    bin_ratio_mask1.append(df[dt_mask1 & df['bin'].isnull()]['y'].count() / len(df[dt_mask1]))
    bin_count_mask1 = (df[dt_mask1].groupby('bin')['y'].count()).tolist()
    bin_count_mask1.append(df[dt_mask1 & df['bin'].isnull()]['y'].count())
    df_show['y_mean_test'] = y_mean_mask1
    # df_show['y_ratio_test'] = y_ratio_mask1
    df_show['bin_ratio_test'] = bin_ratio_mask1
    df_show['bin_count_test'] = bin_count_mask1

    y_mean_mask2 = df[dt_mask2].groupby('bin')['y'].mean().tolist()
    y_mean_mask2.append(df[dt_mask2 & df['bin'].isnull()]['y'].mean())
    y_sum_mask2 = (df[dt_mask2].groupby('bin')['y'].sum()).tolist()
    y_sum_mask2.append(df[dt_mask2 & df['bin'].isnull()]['y'].sum())
    bin_ratio_mask2 = (df[dt_mask2].groupby('bin')['y'].count() / len(df[dt_mask2])).tolist()
    bin_ratio_mask2.append(df[dt_mask2 & df['bin'].isnull()]['y'].count() / len(df[dt_mask2]))
    bin_count_mask2 = (df[dt_mask2].groupby('bin')['y'].count()).tolist()
    bin_count_mask2.append(df[dt_mask2 & df['bin'].isnull()]['y'].count())
    df_show['y_mean_base'] = y_mean_mask2
    # df_show['y_ratio_base'] = y_ratio_mask2
    df_show['bin_ratio_base'] = bin_ratio_mask2
    df_show['bin_count_base'] = bin_count_mask2

    len1 = len(df[dt_mask1])
    len2 = len(df[dt_mask2])
    df_show['y_contrib'] = [(y_sum_mask1[i]*len2/len1 - y_sum_mask2[i]) / (sum(y_sum_mask1)*len2/len1 - sum(y_sum_mask2))
                            for i in range(len(y_sum_mask1))]

    box_cats_str = [str(box_cats[i]) for i in range(box_num)]
    box_cats_str.append('NaN')
    df_show['box_cats_name'] = box_cats_str

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(5, 15))
    xticks_rotation = -30
    ha = 'left'
    df_show.plot.bar(x='box_cats_name', y=['y_mean_test', 'y_mean_base'], rot=-30, ax=axes[0], title=title)
    axes[0].set_xticklabels(df_show['box_cats_name'], rotation=xticks_rotation, ha=ha)
    df_show.plot.bar(x='box_cats_name', y=['y_contrib'], rot=-30, ax=axes[1], title=title)
    axes[1].set_xticklabels(df_show['box_cats_name'], rotation=xticks_rotation, ha=ha)
    df_show.plot.bar(x='box_cats_name', y=['bin_ratio_test', 'bin_ratio_base'], rot=-30, ax=axes[2])
    axes[2].set_xticklabels(df_show['box_cats_name'], rotation=xticks_rotation, ha=ha)
    df_show.plot.bar(x='box_cats_name', y=['bin_count_test', 'bin_count_base'], rot=-30, ax=axes[3])
    axes[3].set_xticklabels(df_show['box_cats_name'], rotation=xticks_rotation, ha=ha)


def plot_y_x_autobin(y, x, dt_mask1, dt_mask2, init_n_bins=100, strategy='chi-merge', title='fact'):
    from mekong.ml.preprocessing import AutoNBinsDiscretizer
    df_nan = pd.DataFrame({
        'y': np.array(y),
        'bin': np.array(x),
    })
    binr = AutoNBinsDiscretizer(strategy=strategy)
    x_v = x.copy()
    v_mask = ~np.isnan(x_v)
    x_v = x_v[v_mask]
    x_v = x_v.reshape(-1, 1)
    y_v = y[v_mask]
    df = pd.DataFrame({
        'y': y_v.reshape(-1),
        'bin': x_v.reshape(-1),
    })
    dt_mask1_v = np.array(dt_mask1)[v_mask]
    dt_mask2_v = np.array(dt_mask2)[v_mask]
    binr.fit(x_v, y_v, init_n_bins=init_n_bins)
    _bin_edges = binr._bin_edges[0]
    box_cats = [(_bin_edges[i - 1], _bin_edges[i]) for i in range(2, len(_bin_edges))]
    box_label = binr.transform(x_v).astype(int)
    box_num = len(_bin_edges) - 1

    df['bin'] = box_label

    df_show = pd.DataFrame(index=[i+2 for i in range(box_num)])

    y_mean_mask1 = df[dt_mask1_v].groupby('bin')['y'].mean().tolist()
    y_mean_mask1.append(df_nan[dt_mask1 & df_nan['bin'].isnull()]['y'].mean())
    bin_ratio_mask1 = (df[dt_mask1_v].groupby('bin')['y'].count() / len(df[dt_mask1_v])).tolist()
    bin_ratio_mask1.append(df_nan[dt_mask1 & df_nan['bin'].isnull()]['y'].count() / len(df_nan[dt_mask1]))
    bin_count_mask1 = (df[dt_mask1_v].groupby('bin')['y'].count()).tolist()
    bin_count_mask1.append(df_nan[dt_mask1 & df_nan['bin'].isnull()]['y'].count())
    df_show['y_mean_test'] = y_mean_mask1
    df_show['bin_ratio_test'] = bin_ratio_mask1
    df_show['bin_count_test'] = bin_count_mask1

    y_mean_mask2 = df[dt_mask2_v].groupby('bin')['y'].mean().tolist()
    y_mean_mask2.append(df_nan[dt_mask2 & df_nan['bin'].isnull()]['y'].mean())
    bin_ratio_mask2 = (df[dt_mask2_v].groupby('bin')['y'].count() / len(df[dt_mask2_v])).tolist()
    bin_ratio_mask2.append(df_nan[dt_mask2 & df_nan['bin'].isnull()]['y'].count() / len(df_nan[dt_mask2]))
    bin_count_mask2 = (df[dt_mask2_v].groupby('bin')['y'].count()).tolist()
    bin_count_mask2.append(df_nan[dt_mask2 & df_nan['bin'].isnull()]['y'].count())
    df_show['y_mean_base'] = y_mean_mask2
    df_show['bin_ratio_base'] = bin_ratio_mask2
    df_show['bin_count_base'] = bin_count_mask2

    box_cats_str = [f'[{box_cats[i][0]}, {box_cats[i][1]})' for i in range(box_num-1)] + ['NaN']
    df_show['box_cats_name'] = box_cats_str

    # https://stackoverflow.com/questions/14852821/aligning-rotated-xticklabels-with-their-respective-xticks
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 15))
    xticks_rotation = -30
    ha = 'left'
    df_show.plot.bar(x='box_cats_name', y=['y_mean_test', 'y_mean_base'], rot=-30, ax=axes[0], title=title)
    axes[0].set_xticklabels(df_show['box_cats_name'], rotation=xticks_rotation, ha=ha)
    df_show.plot.bar(x='box_cats_name', y=['bin_ratio_test', 'bin_ratio_base'], rot=-30, ax=axes[1])
    axes[1].set_xticklabels(df_show['box_cats_name'], rotation=xticks_rotation, ha=ha)
    df_show.plot.bar(x='box_cats_name', y=['bin_count_test', 'bin_count_base'], rot=-30, ax=axes[2])
    axes[2].set_xticklabels(df_show['box_cats_name'], rotation=xticks_rotation, ha=ha)


def calc_contrib_table(fea_d, y, dt_mask_base, dt_mask_test, mode='contrib_II', order_keys=None, **kwargs):
    _Searcher = type(
        '_Searcher',
        (BeamSearchMask,),
        {
            'MASK_MAPPING': dict(),
            'fea_d': dict(),
            'y': None
        }
    )
    fea_keys = list(fea_d.keys())
    y = y if y is not None else np.ones(len(fea_d[fea_keys[0]]))
    _Searcher.set_X_y(fea_d, y)
    _Searcher.set_dt_mask(dt_mask_test, dt_mask_base)
    if mode == 'contrib_I':
        assert len(fea_keys) == 1, f'len(fea_keys) = {len(fea_keys)}, if mode = {mode}, there must be only one key in fea_d'
        bsm = _Searcher(fea_keys[0], split_mode=mode, bins_mode='unique')
        bsm.select_mask()
        return pd.DataFrame({
            '环比': bsm.ringr,
            '总环比贡献值': bsm.ringr_contrib_v,
            '总环比贡献率': bsm.ringr_contrib_r,
        })
    elif mode == 'contrib_II':
        assert len(fea_keys) == 1, f'len(fea_keys) = {len(fea_keys)}, if mode = {mode}, there must be only one key in fea_d'
        bsm = _Searcher(fea_keys[0], split_mode=mode, bins_mode='unique')
        bsm.select_mask()
        return pd.DataFrame({
            '环比': bsm.ringr,
            '总环比贡献值': bsm.ringr_contrib_v,
            '总环比贡献率': bsm.ringr_contrib_r,
        })
    elif mode == 'ab_contrib_II':
        mask_a = kwargs['mask_a']
        mask_b = kwargs['mask_b']
        setattr(_Searcher, 'mask_a', mask_a)
        setattr(_Searcher, 'mask_b', mask_b)
        # bsm = _Searcher('channel', split_mode=mode, bins_mode='unique')
        assert len(fea_keys) == 1, f'len(fea_keys) = {len(fea_keys)}, if mode = {mode}, there must be only one key in fea_d'
        bsm = _Searcher(fea_keys[0], split_mode=mode, bins_mode='unique')
        bsm.select_mask()
        return pd.DataFrame({
            '贡献值': bsm.ringr_contrib_v,
            'base时段贡献值': bsm.ringr_contrib_v_base,
            'test时段贡献值': bsm.ringr_contrib_v_test,
        })
    elif mode == 'contrib_III':
        order_keys = order_keys if order_keys is not None else list(fea_d.keys())
        end_nodes = _Searcher.layer_traverse(order_keys, split_mode=mode, bins_mode='bool')
        node = end_nodes[-1]
        chain_l = node.get_chain_list()
        log_E_F = chain_l[-1].log_E_F
        return pd.DataFrame([{x.name: x.log_Ei_Fi / log_E_F for x in chain_l}])
    elif mode == 'contrib_IV':
        order_keys = order_keys if order_keys is not None else list(fea_d.keys())
        end_nodes = _Searcher.layer_traverse(order_keys, split_mode=mode, bins_mode='unique')
        return pd.DataFrame([{
            '分类': x.chain_secondary_name,
            '环比': x.ringr,
            "总环比贡献值": x.ringr_contrib_v,
            "总环比贡献率": x.ringr_contrib_r,
            "分布变化贡献值": x.distribution_change_v,
            "分布变化贡献率": x.distribution_change_r,
            "数值变化贡献值": x.numeric_change_v,
            "数值变化贡献率": x.numeric_change_r
        } for x in end_nodes])


def get_LINEARREG_X(df_g_r2c, group_name, y_name, how, timeperiod=None):
    '''
    how = ANGLE, INTERCEPT, SLOPE
    '''
    tmp_d = {}
    _func = getattr(talib, f'LINEARREG_{how}')
    for name, g in df_g_r2c.groupby([group_name]):
        _ = g[y_name].values
        if len(_) < 2:
            tmp_d[name] = 0.0
            continue
        _timeperiod = len(_) if timeperiod is None else timeperiod
        tmp_d[name] = _func(_, _timeperiod)[-1]
    return tmp_d


def get_biz_mean_groupby_dt_and_group(dt, group, y):
    df = pd.DataFrame({
        'date': dt,
        'group': group,
        'y': y
    })
    df['count'] = 1
    df_g_r2 = df.groupby(['date', 'group'])['y'].mean()
    df_g_r2b = df_g_r2.reset_index()
    df_g_r2_2 = df.groupby(['date', 'group'])['count'].count()
    df_g_r2_2 = df_g_r2_2.reset_index()
    df_g_r2b = df_g_r2b.merge(df_g_r2_2, on=['date', 'group'], how='left')
    return df_g_r2b


def get_linearreg_rank(dt, group, y, weight=None, topk=20, timeperiod=None):
    """检测不同 group 的业务指标 y 随 dt 变化的异常情况 (y 直接给单个样本的表现即可)

    Parameters
    ----------
    dt : numpy vector
        时间列, 检测趋势变化的 x 轴.
    group : numpy vector
        分组信息, 检测同一组内连续 dt 的 y 的变化.
    y : numpy vector
        业务指标值.
    weight: dict or pd.Series
        分组和权重的映射关系: topk 按这个选
    topk : int
        只看样本数最大的 topk 的组.
    timeperiod : type
        检测多长时间周期的 y, timeperiod 代表 dt 最细力度的个数.

    Returns
    -------
    pd.DataFrame
        不同组异常度排序的表.

    """
    df = pd.DataFrame({
        'group': group,
        'y': y
    })
    df_g_cnt = df.groupby(['group'])['y'].count()
    df_g_cnt = pd.DataFrame(df_g_cnt)
    df_g_cnt = df_g_cnt.reset_index()
    df_g_cnt = df_g_cnt.rename(columns={'y': 'count'})
    if weight is not None:
        df_g_cnt['weight'] = df_g_cnt['group'].map(weight)
    else:
        df_g_cnt['weight'] = df_g_cnt['count']
    df_g_cnt['sample_ratio'] = df_g_cnt['count'] / df_g_cnt['count'].sum()
    df_g_r2b = get_biz_mean_groupby_dt_and_group(dt, group, y)
    df_g_r2c = df_g_r2b[['group', 'y']]
    df_g_cnt['ANGLE'] = df_g_cnt['group'].map(get_LINEARREG_X(df_g_r2c, 'group', 'y', 'ANGLE', timeperiod))
    df_g_cnt['abs_ANGLE'] = df_g_cnt['ANGLE'].abs()
    df_g_cnt['INTERCEPT'] = df_g_cnt['group'].map(get_LINEARREG_X(df_g_r2c, 'group', 'y', 'INTERCEPT', timeperiod))
    df_g_cnt['abs_INTERCEPT'] = df_g_cnt['INTERCEPT'].abs()
    df_g_cnt['SLOPE'] = df_g_cnt['group'].map(get_LINEARREG_X(df_g_r2c, 'group', 'y', 'SLOPE', timeperiod))
    df_g_cnt['abs_SLOPE'] = df_g_cnt['SLOPE'].abs()
    df_g_cnt = df_g_cnt.sort_values(['weight', 'abs_ANGLE'], ascending=False)
    df_g_cnt2 = df_g_cnt.iloc[:topk]
    df_g_cnt2 = df_g_cnt2.sort_values('ANGLE')
    return df_g_cnt2


def wrap_LINEARREG_X(x, how, timeperiod=14):
    '''
    how = ANGLE, INTERCEPT, SLOPE
    '''
    _func = getattr(talib, f'LINEARREG_{how}')
    if len(x) < 2:
        return np.ones(len(x)) * np.nan
    if np.isnan(x).all():
        return x
    return _func(x, timeperiod=timeperiod)


def get_ts_linearreg_rank_old(df: pd.DataFrame,
                          index_cols: list,
                          y_cols: list = None,
                          dt_name: str = 'dt',
                          mode: str = 'linearreg',
                          is_rank: bool = True,
                          timeperiod: int = 14):
    dt_nd = np.sort(np.unique(df[dt_name].values))
    if y_cols is None:
        y_cols = [x for x in df.columns if x not in index_cols + [dt_name]]
    tmp_d = dict()
    for name, g in df.groupby(index_cols):
        for col in y_cols:
            tmp_key = '::'.join(name) + '::' + col
            v1 = g[col].fillna(method='ffill').fillna(method='bfill').values
            v1_max = np.nanmax(v1)
            v1_min = np.nanmin(v1)
            if v1_max - v1_min == 0.0:
                v1 = np.zeros(len(v1))
            else:
                v1 = (v1 - v1_min) / (v1_max - v1_min)
            if mode == 'linearreg':
                v2 = wrap_LINEARREG_X(v1, 'ANGLE', timeperiod=timeperiod)
            else:
                raise Exception(f'mode {mode} not define.')
            tmp_d[tmp_key] = pd.Series(data=v2,
                                       index=g[dt_name].values)
    out_df = pd.DataFrame(index=dt_nd)
    for k, v in tmp_d.items():
        out_df[k] = out_df.index.map(v)
    if is_rank:
        out_df = out_df.rank(pct=True)
    return out_df


def get_ts_linearreg_rank(df: pd.DataFrame,
                          index_cols: list,
                          y_cols: list = None,
                          dt_name: str = 'dt',
                          mode: str = 'linearreg',
                          is_rank: str = 'num',
                          timeperiod: int = 14):
    dt_nd = np.sort(np.unique(df[dt_name].values))
    out_df = pd.DataFrame(index=dt_nd)
    # for k, v in tmp_d.items():
    #     out_df[k] = out_df.index.map(v)
    if y_cols is None:
        y_cols = [x for x in df.columns if x not in index_cols + [dt_name]]
    # tmp_d = dict()
    for name, g in df.groupby(index_cols):
        name_l = name if not isinstance(name, str) else [name]
        for col in y_cols:
            _tmp = pd.Series(data=g[col].values, index=g[dt_name].values)
            tmp_key = '::'.join(name_l) + '::' + col
            out_df[tmp_key] = out_df.index.map(_tmp)
            v1 = out_df[tmp_key].fillna(method='ffill').fillna(method='bfill').values
            v1_max = np.nanmax(v1)
            v1_min = np.nanmin(v1)
            if v1_max - v1_min == 0.0:
                v1x = np.ones(len(v1)) * 0.5
            else:
                v1x = (v1 - v1_min) / (v1_max - v1_min)
            if mode == 'linearreg':
                v2 = wrap_LINEARREG_X(v1x, 'ANGLE', timeperiod=timeperiod)
            elif mode == 'move_mean_change':
                mvmean = bn.move_mean(v1, timeperiod, 1)
                v2 = np.divide(v1 - mvmean, mvmean)
            else:
                raise Exception(f'mode {mode} not define.')
            out_df[tmp_key] = v2
    if is_rank == 'pct':
        out_df = out_df.rank(pct=True)
    elif is_rank == 'num':
        out_df = out_df.rank(pct=False)
    else:
        pass
    return out_df


def get_ts_change_v(v1, mode, timeperiod):
    v1_max = np.nanmax(v1)
    v1_min = np.nanmin(v1)
    if v1_max - v1_min == 0.0:
        v1x = np.ones(len(v1)) * 0.5
    else:
        v1x = (v1 - v1_min) / (v1_max - v1_min)
    if mode == 'linearreg':
        v2 = wrap_LINEARREG_X(v1x, 'ANGLE', timeperiod=timeperiod)
    elif mode == 'move_mean_change':
        mvmean = bn.move_mean(v1, timeperiod, 1)
        v2 = np.divide(v1 - mvmean, mvmean)
    elif mode == 'shift_change':
        shifted_v1 = pd.Series(v1)
        shifted_v1 = shifted_v1.shift(timeperiod, fill_value=shifted_v1.iloc[0]).values
        v2 = np.divide(v1 - shifted_v1, shifted_v1)
    else:
        raise Exception(f'mode {mode} not define.')
    return v2


def get_ts_change_rank(
    dt: np.ndarray,
    group: np.ndarray,
    y: np.ndarray,
    agg: str = 'mean',
    mode: str = 'linearreg',
    is_rank: str = 'num',
    timeperiod: int = 14
):
    dt_nd = np.sort(np.unique(dt))
    df = pd.DataFrame({
        'dt': dt,
        'group': group,
        'y': y
    })
    out_df = pd.DataFrame(index=dt_nd)
    for name, g in df.groupby('group'):
        if agg == 'mean':
            _tmp = g.groupby('dt')['y'].mean()
        elif agg == 'count':
            _tmp = g.groupby('dt')['y'].count()
        else:
            raise Exception('unkown agg method')
        tmp_key = name
        out_df[tmp_key] = out_df.index.map(_tmp)
        v1 = out_df[tmp_key].fillna(method='ffill').fillna(method='bfill').values
        v2 = get_ts_change_v(v1, mode, timeperiod)
        out_df[tmp_key] = v2
    if is_rank == 'pct':
        out_df = out_df.rank(pct=True)
    elif is_rank == 'num':
        out_df = out_df.rank(pct=False)
    else:
        pass
    return out_df


def get_ts_change_rank_wide(
    df: pd.DataFrame,
    index_cols: list,
    y_cols: list = None,
    dt_name: str = 'dt',
    mode: str = 'linearreg',
    is_rank: str = 'num',
    timeperiod: int = 14
):
    dt_nd = np.sort(np.unique(df[dt_name].values))
    out_df = pd.DataFrame(index=dt_nd)
    # for k, v in tmp_d.items():
    #     out_df[k] = out_df.index.map(v)
    assert y_cols is not None
    if y_cols is None:
        y_cols = [x for x in df.columns if x not in index_cols + [dt_name]]
    # tmp_d = dict()
    for name, g in df.groupby(index_cols):
        name_l = name if not isinstance(name, str) else [name]
        for col in y_cols:
            _tmp = pd.Series(data=g[col].values, index=g[dt_name].values)
            tmp_key = '::'.join(name_l) + '::' + col
            out_df[tmp_key] = out_df.index.map(_tmp)
            v1 = out_df[tmp_key].fillna(method='ffill').fillna(method='bfill').values
            v2 = get_ts_change_v(v1, mode, timeperiod)
            out_df[tmp_key] = v2
    if is_rank == 'pct':
        out_df = out_df.rank(pct=True)
    elif is_rank == 'num':
        out_df = out_df.rank(pct=False)
    else:
        pass
    return out_df


def get_topk_group_single_key(
    group: np.ndarray,
    value: np.ndarray,
    topk: int = 20,
    mode: str = 'mean'
):
    df = pd.DataFrame({
        'group': group,
        'y': value
    })
    if mode == 'mean':
        out_df = df.groupby('group')['y'].mean().nlargest(topk)
    elif mode == 'count':
        out_df = df.groupby('group')['y'].count().nlargest(topk)
    return out_df.index.tolist()


def get_topk_group_multi_key(
    group: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    topk: int = 20,
    mode: str = 'mean'
):
    df = pd.DataFrame({
        'group': group,
        'key': key,
        'y': value
    })
    uni_keys = np.unique(key)
    tmp_l = []
    for k in uni_keys:
        tmp1 = df[df['key'] == k]
        if mode == 'mean':
            tmp2 = tmp1.groupby('group')['y'].mean().nlargest(topk).index.tolist()
        elif mode == 'count':
            tmp2 = tmp1.groupby('group')['y'].count().nlargest(topk).index.tolist()
        tmp_l += [x + '::' + k for x in tmp2]
    return tmp_l


def get_topk_group_wide(
    df: pd.DataFrame,
    group_col: str,
    y_cols: list,
    topk: int = 20,
    mode: str = 'mean'
):
    tmp_l = []
    for k in y_cols:
        if mode == 'mean':
            tmp2 = df[[group_col] + [k]].groupby(group_col)[k].mean().nlargest(topk).index.tolist()
        elif mode == 'count':
            tmp2 = df[[group_col] + [k]].groupby(group_col)[k].count().nlargest(topk).index.tolist()
        tmp_l += [x + '::' + k for x in tmp2]
    return tmp_l
