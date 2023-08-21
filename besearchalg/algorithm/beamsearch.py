from collections import defaultdict
import numpy as np
import pandas as pd
import tqdm
from copy import deepcopy
from collections import deque


def merge_contrib(y, mask_l):
    ml = mask_l[0][0]
    mr = mask_l[0][1]
    yl = y[ml]
    yr = y[mr]
    ai = ml.sum() / len(ml) if len(ml) > 0 else 0.0
    bi = mr.sum() / len(mr) if len(mr) > 0 else 0.0
    if len(mask_l) > 1:
        Ei = merge_contrib(yl, [[x[0][ml], x[1][ml]] for x in mask_l[1:]])
        Fi = merge_contrib(yr, [[x[0][mr], x[1][mr]] for x in mask_l[1:]])
    else:
        Ei = np.nanmean(yl) if len(yl) != 0 else 0.0
        Fi = np.nanmean(yr) if len(yr) != 0 else 0.0
    # return (bi*Fi - ai*Ei)/Ei if Ei != 0 else 0.0
    return bi*Fi - ai*Ei


class BeamSearchLift:
    DT_MASK_TEST = None
    DT_MASK_BASE = None
    SPLIT_MAPPING = {}
    def __init__(self, name, fea_d, y, prev=None, split_mode='lift_diff'):
        self.prev = prev
        self.name = name
        self.fea_d = fea_d
        self.y = y
        self.th = 0.5
        self.split_mode = split_mode
        self.part = 'None'

    @property
    def DT_MASK1(self):
        return self.DT_MASK_TEST

    @property
    def DT_MASK2(self):
        return self.DT_MASK_BASE

    @classmethod
    def set_dt_mask(cls, dt_mask_test, dt_mask_base):
        '''
        cls.DT_MASK_TEST: dt_mask_test: dt_mask1
        cls.DT_MASK_BASE: dt_mask_base: dt_mask2
        '''
        cls.DT_MASK_TEST = dt_mask_test
        cls.DT_MASK_BASE = dt_mask_base

    @classmethod
    def register(cls, name):
        def _thunk(func):
            cls.SPLIT_MAPPING[name] = func
            return func
        return _thunk

    def select_mask(self):
        assert self.split_mode in self.SPLIT_MAPPING, 'Unkown split_mode!'
        _select_mask = self.SPLIT_MAPPING[self.split_mode]
        _select_mask(self)

    def get_chain_mask(self):
        if self.prev is None:
            return self.mask
        else:
            return self.mask & self.prev.get_chain_mask()

    def get_chain_fea_set(self):
        if self.prev is None:
            return set([self.name])
        else:
            return set([self.name] + list(self.prev.get_chain_fea_set()))

    def get_lift(self):
        return self.lift

    def __lt__(self, other):
        # return self.lift < other.lift
        return self.abs_contrib < other.abs_contrib

    def backward_names(self):
        cur = self
        while cur is not None:
            yield cur.name
            cur = cur.prev

    def get_chain_list(self):
        bask_chain_l = []
        cur = self
        while cur is not None:
            bask_chain_l.append(cur)
            cur = cur.prev
        return list(reversed(bask_chain_l))


@BeamSearchLift.register('lift_diff')
def select_lift_diff(self):
    mask1 = self.fea_d[self.name] < self.th
    mask2 = self.fea_d[self.name] > self.th
    if self.prev is None:
        chain_mask1 = mask1
        chain_mask2 = mask2
    else:
        chain_mask1 = mask1 & self.prev.get_chain_mask()
        chain_mask2 = mask2 & self.prev.get_chain_mask()
    dt1c1m = self.DT_MASK1 & chain_mask1
    dt2c1m = self.DT_MASK2 & chain_mask1
    dt1c2m = self.DT_MASK1 & chain_mask2
    dt2c2m = self.DT_MASK2 & chain_mask2
    lift1 = self.y[dt1c1m].mean() - self.y[dt2c1m].mean()
    lift2 = self.y[dt1c2m].mean() - self.y[dt2c2m].mean()
    contrib1 = self.y[dt1c1m].mean() / self.y[dt2c1m].mean()
    contrib2 = self.y[dt1c2m].mean() / self.y[dt2c2m].mean()
    self.contrib1 = contrib1
    self.contrib2 = contrib2
    self.abs_contrib = np.abs(np.log(contrib1 / contrib2))
    if contrib1 > contrib2:
        self.lift = lift1
        self.mask = mask1
        self.part = 'left'
    else:
        self.lift = lift2
        self.mask = mask2
        self.part = 'right'

@BeamSearchLift.register('unbalance_pd_lift_diff')
def select_unbalance_pd_lift_diff(self):
    mask1 = pd.Series(self.fea_d[self.name]).rank(pct=True) <= self.th
    mask2 = pd.Series(self.fea_d[self.name]).rank(pct=True) > self.th
    if self.prev is None:
        chain_mask1 = mask1
        chain_mask2 = mask2
    else:
        chain_mask1 = mask1 & self.prev.get_chain_mask()
        chain_mask2 = mask2 & self.prev.get_chain_mask()
    min_group_ratio = np.min([chain_mask1.sum(), chain_mask2.sum()]) / len(chain_mask1)
    dt1c1m = self.DT_MASK1 & chain_mask1
    dt2c1m = self.DT_MASK2 & chain_mask1
    dt1c2m = self.DT_MASK1 & chain_mask2
    dt2c2m = self.DT_MASK2 & chain_mask2
    p11 = np.nanmean(self.y[dt1c1m]) if dt1c1m.sum() != 0 else 0.0
    p12 = np.nanmean(self.y[dt1c2m]) if dt1c2m.sum() != 0 else 0.0
    p21 = np.nanmean(self.y[dt2c1m]) if dt2c1m.sum() != 0 else 0.0
    p22 = np.nanmean(self.y[dt2c2m]) if dt2c2m.sum() != 0 else 0.0
    lift1 = (p11 - p21) / p21 if p21 > 0. else 0.
    lift2 = (p12 - p22) / p22 if p22 > 0. else 0.
    contrib1 = lift1
    contrib2 = lift2
    self.contrib1 = contrib1
    self.contrib2 = contrib2
    self.abs_contrib = np.abs(contrib1 - contrib2) * min_group_ratio
    # if contrib1 > contrib2:
    if np.abs(contrib1) > np.abs(contrib2):
        self.lift = lift1
        self.mask = mask1
        self.part = 'left'
    else:
        self.lift = lift2
        self.mask = mask2
        self.part = 'right'

@BeamSearchLift.register('ratio_pd_lift_diff')
def select_ratio_pd_lift_diff(self):
    mask1 = pd.Series(self.fea_d[self.name]).rank(pct=True) <= self.th
    mask2 = pd.Series(self.fea_d[self.name]).rank(pct=True) > self.th
    if self.prev is None:
        chain_mask1 = mask1
        chain_mask2 = mask2
    else:
        chain_mask1 = mask1 & self.prev.get_chain_mask()
        chain_mask2 = mask2 & self.prev.get_chain_mask()
    min_group_ratio = np.min([chain_mask1.sum(), chain_mask2.sum()]) / len(chain_mask1)
    dt1c1m = self.DT_MASK1 & chain_mask1
    dt2c1m = self.DT_MASK2 & chain_mask1
    dt1c2m = self.DT_MASK1 & chain_mask2
    dt2c2m = self.DT_MASK2 & chain_mask2
    p11 = dt1c1m.sum() / self.DT_MASK1.sum()
    p12 = dt1c2m.sum() / self.DT_MASK1.sum()
    p21 = dt2c1m.sum() / self.DT_MASK2.sum()
    p22 = dt2c2m.sum() / self.DT_MASK2.sum()
    # p11 = np.nanmean(self.y[dt1c1m]) if dt1c1m.sum() != 0 else 0.0
    # p12 = np.nanmean(self.y[dt1c2m]) if dt1c2m.sum() != 0 else 0.0
    # p21 = np.nanmean(self.y[dt2c1m]) if dt2c1m.sum() != 0 else 0.0
    # p22 = np.nanmean(self.y[dt2c2m]) if dt2c2m.sum() != 0 else 0.0
    lift1 = (p11 - p21) / p21 if p21 > 0. else 0.
    lift2 = (p12 - p22) / p22 if p22 > 0. else 0.
    contrib1 = lift1
    contrib2 = lift2
    self.contrib1 = contrib1
    self.contrib2 = contrib2
    self.abs_contrib = np.abs(contrib1 - contrib2) * min_group_ratio
    # if contrib1 > contrib2:
    if np.abs(contrib1) > np.abs(contrib2):
        self.lift = lift1
        self.mask = mask1
        self.part = 'left'
    else:
        self.lift = lift2
        self.mask = mask2
        self.part = 'right'

@BeamSearchLift.register('prop_diff')
def select_prop_diff(self):
    mask1 = self.fea_d[self.name] < self.th
    mask2 = self.fea_d[self.name] > self.th
    if self.prev is None:
        chain_mask1 = mask1
        chain_mask2 = mask2
    else:
        chain_mask1 = mask1 & self.prev.get_chain_mask()
        chain_mask2 = mask2 & self.prev.get_chain_mask()
    dt1c1m = self.DT_MASK1 & chain_mask1
    dt2c1m = self.DT_MASK2 & chain_mask1
    dt1c2m = self.DT_MASK1 & chain_mask2
    dt2c2m = self.DT_MASK2 & chain_mask2
    p11 = self.y[dt1c1m].sum() / self.DT_MASK1.sum()
    p12 = self.y[dt1c2m].sum() / self.DT_MASK1.sum()
    p21 = self.y[dt2c1m].sum() / self.DT_MASK2.sum()
    p22 = self.y[dt2c2m].sum() / self.DT_MASK2.sum()
    lift1 = p11 - p21
    lift2 = p12 - p22
    contrib1 = lift1
    contrib2 = lift2
    self.contrib1 = contrib1
    self.contrib2 = contrib2
    self.abs_contrib = np.abs(contrib1 - contrib2)
    if contrib1 > contrib2:
        self.lift = lift1
        self.mask = mask1
        self.part = 'left'
    else:
        self.lift = lift2
        self.mask = mask2
        self.part = 'right'

@BeamSearchLift.register('pd_prop_diff')
def select_pd_prop_diff(self):
    mask1 = pd.Series(self.fea_d[self.name]).rank(pct=True) <= self.th
    mask2 = pd.Series(self.fea_d[self.name]).rank(pct=True) > self.th
    if self.prev is None:
        chain_mask1 = mask1
        chain_mask2 = mask2
    else:
        chain_mask1 = mask1 & self.prev.get_chain_mask()
        chain_mask2 = mask2 & self.prev.get_chain_mask()
    dt1c1m = self.DT_MASK1 & chain_mask1
    dt2c1m = self.DT_MASK2 & chain_mask1
    dt1c2m = self.DT_MASK1 & chain_mask2
    dt2c2m = self.DT_MASK2 & chain_mask2
    p11 = np.nansum(self.y[dt1c1m]) / np.nansum(self.DT_MASK1)
    p12 = np.nansum(self.y[dt1c2m]) / np.nansum(self.DT_MASK1)
    p21 = np.nansum(self.y[dt2c1m]) / np.nansum(self.DT_MASK2)
    p22 = np.nansum(self.y[dt2c2m]) / np.nansum(self.DT_MASK2)
    lift1 = p11 - p21
    lift2 = p12 - p22
    contrib1 = lift1
    contrib2 = lift2
    self.contrib1 = contrib1
    self.contrib2 = contrib2
    self.abs_contrib = np.abs(contrib1 - contrib2)
    # if contrib1 > contrib2:
    if np.abs(contrib1) > np.abs(contrib2):
        self.lift = lift1
        self.mask = mask1
        self.part = 'left'
    else:
        self.lift = lift2
        self.mask = mask2
        self.part = 'right'

@BeamSearchLift.register('unbalance_pd_prop_diff')
def select_unbalance_pd_prop_diff(self):
    mask1 = pd.Series(self.fea_d[self.name]).rank(pct=True) <= self.th
    mask2 = pd.Series(self.fea_d[self.name]).rank(pct=True) > self.th
    if self.prev is None:
        chain_mask1 = mask1
        chain_mask2 = mask2
    else:
        chain_mask1 = mask1 & self.prev.get_chain_mask()
        chain_mask2 = mask2 & self.prev.get_chain_mask()
    unbalance_factor = np.abs(chain_mask1.sum() - chain_mask2.sum()) / len(chain_mask1)
    dt1c1m = self.DT_MASK1 & chain_mask1
    dt2c1m = self.DT_MASK2 & chain_mask1
    dt1c2m = self.DT_MASK1 & chain_mask2
    dt2c2m = self.DT_MASK2 & chain_mask2
    p11 = np.nansum(self.y[dt1c1m]) / np.nansum(self.DT_MASK1)
    p12 = np.nansum(self.y[dt1c2m]) / np.nansum(self.DT_MASK1)
    p21 = np.nansum(self.y[dt2c1m]) / np.nansum(self.DT_MASK2)
    p22 = np.nansum(self.y[dt2c2m]) / np.nansum(self.DT_MASK2)
    lift1 = p11 - p21
    lift2 = p12 - p22
    contrib1 = lift1
    contrib2 = lift2
    self.contrib1 = contrib1
    self.contrib2 = contrib2
    self.abs_contrib = np.abs(contrib1 - contrib2) * (1 - unbalance_factor)
    # if contrib1 > contrib2:
    if np.abs(contrib1) > np.abs(contrib2):
        self.lift = lift1
        self.mask = mask1
        self.part = 'left'
    else:
        self.lift = lift2
        self.mask = mask2
        self.part = 'right'

@BeamSearchLift.register('right_pd_prop_diff')
def select_right_pd_prop_diff(self):
    mask2 = pd.Series(self.fea_d[self.name]).rank(pct=True) > self.th
    if self.prev is None:
        chain_mask2 = mask2
    else:
        chain_mask2 = mask2 & self.prev.get_chain_mask()

    dt1c2m = self.DT_MASK1 & chain_mask2
    dt2c2m = self.DT_MASK2 & chain_mask2

    p12 = self.y[dt1c2m].sum() / self.DT_MASK1.sum()

    p22 = self.y[dt2c2m].sum() / self.DT_MASK2.sum()

    lift2 = p12 - p22

    contrib2 = lift2

    self.contrib2 = contrib2
    self.abs_contrib = np.abs(contrib2)

    self.lift = lift2
    self.mask = mask2
    self.part = 'right'

@BeamSearchLift.register('right_pd_lift_diff')
def select_right_pd_lift_diff(self):
    mask2 = pd.Series(self.fea_d[self.name]).rank(pct=True) > self.th
    if self.prev is None:
        chain_mask2 = mask2
    else:
        chain_mask2 = mask2 & self.prev.get_chain_mask()

    dt1c2m = self.DT_MASK1 & chain_mask2
    dt2c2m = self.DT_MASK2 & chain_mask2

    if dt1c2m.sum() < 10 or dt2c2m.sum() < 10:
        self.contrib2 = 0.0
        self.abs_contrib = 0.0

        self.lift = 0
        self.mask = mask2
        self.part = 'right'
        return

    p12 = self.y[dt1c2m].sum() / dt1c2m.sum()
    p22 = self.y[dt2c2m].sum() / dt2c2m.sum()
    lift2 = p12 - p22
    contrib2 = lift2

    self.contrib2 = contrib2
    self.abs_contrib = np.abs(contrib2)

    self.lift = lift2
    self.mask = mask2
    self.part = 'right'

@BeamSearchLift.register('right_w_pd_lift_diff')
def select_right_w_pd_lift_diff(self):
    mask2 = pd.Series(self.fea_d[self.name]).rank(pct=True) > self.th
    if self.prev is None:
        chain_mask2 = mask2
    else:
        chain_mask2 = mask2 & self.prev.get_chain_mask()

    dt1c2m = self.DT_MASK1 & chain_mask2
    dt2c2m = self.DT_MASK2 & chain_mask2

    if dt1c2m.sum() < 10 or dt2c2m.sum() < 10:
        self.contrib2 = 0.0
        self.abs_contrib = 0.0

        self.lift = 0
        self.mask = mask2
        self.part = 'right'
        return

    p12 = self.y[dt1c2m].sum() / dt1c2m.sum()
    p22 = self.y[dt2c2m].sum() / dt2c2m.sum()
    lift2 = (p12 - p22) * (dt1c2m.sum()/self.DT_MASK1.sum() + dt2c2m.sum()/self.DT_MASK2.sum()) / 2
    contrib2 = lift2

    self.contrib2 = contrib2
    self.abs_contrib = np.abs(contrib2)

    self.lift = lift2
    self.mask = mask2
    self.part = 'right'

@BeamSearchLift.register('nan_prop_diff')
def select_nan_prop_diff(self):
    mask1 = np.isnan(self.fea_d[self.name])
    mask2 = ~np.isnan(self.fea_d[self.name])
    if self.prev is None:
        chain_mask1 = mask1
        chain_mask2 = mask2
    else:
        chain_mask1 = mask1 & self.prev.get_chain_mask()
        chain_mask2 = mask2 & self.prev.get_chain_mask()
    dt1c1m = self.DT_MASK1 & chain_mask1
    dt2c1m = self.DT_MASK2 & chain_mask1
    dt1c2m = self.DT_MASK1 & chain_mask2
    dt2c2m = self.DT_MASK2 & chain_mask2
    p11 = self.y[dt1c1m].sum() / self.DT_MASK1.sum()
    p12 = self.y[dt1c2m].sum() / self.DT_MASK1.sum()
    p21 = self.y[dt2c1m].sum() / self.DT_MASK2.sum()
    p22 = self.y[dt2c2m].sum() / self.DT_MASK2.sum()
    lift1 = p11 - p21
    lift2 = p12 - p22
    contrib1 = lift1
    contrib2 = lift2
    self.contrib1 = contrib1
    self.contrib2 = contrib2
    self.abs_contrib = np.abs(contrib1 - contrib2)
    if contrib1 > contrib2:
        self.lift = lift1
        self.mask = mask1
        self.part = 'nan'
    else:
        self.lift = lift2
        self.mask = mask2
        self.part = 'has'



class BeamSearchMask(BeamSearchLift):
    SPLIT_MAPPING = dict()
    BINS_MASK_METHOD_MAPPING = dict()
    MASK_MAPPING = dict()
    fea_d = dict()
    y = None

    def __init__(self, name, prev=None, split_mode='lift_diff', bins_mode=None):
        self.prev = prev
        self.name = name
        self.split_mode = split_mode
        self.bins_mode = bins_mode
        self.part = 'None'

    @property
    def secondary_name(self):
        return f'{self.name}:{self.part}'

    @property
    def chain_secondary_name(self):
        return '|'.join([x.secondary_name for x in self.get_chain_list()])

    @classmethod
    def set_X_y(cls, fea_d, y):
        cls.fea_d = fea_d
        cls.y = y

    @property
    def x(self):
        return self.fea_d[self.name]

    @property
    def y(self):
        return self.y

    @classmethod
    def clear(cls):
        cls.DT_MASK_TEST = None
        cls.DT_MASK_BASE = None
        # cls.SPLIT_MAPPING = dict()
        # cls.BINS_MASK_METHOD_MAPPING = dict()
        cls.MASK_MAPPING = dict()
        cls.fea_d = dict()
        cls.y = None

    @classmethod
    def set_bins_mask_method(cls, name):
        def _thunk(func):
            cls.BINS_MASK_METHOD_MAPPING[name] = func
        return _thunk

    def get_bins_mask(self):
        _bins_mask_method = self.BINS_MASK_METHOD_MAPPING[self.bins_mode]
        return _bins_mask_method(self)

    @property
    def bins_mask(self):
        if self.name not in self.MASK_MAPPING:
            self.MASK_MAPPING[self.name] = self.get_bins_mask()
        return self.MASK_MAPPING[self.name]

    @property
    def prev_mask(self):
        if self.prev is None:
            prev_mask = np.ones(len(self.y)).astype(bool)
        else:
            prev_mask = self.prev.get_chain_mask()
        return prev_mask

    def get_chain_product(self, attr):
        v = 1.0
        curr = self
        while curr:
            v *= getattr(curr, attr)
            curr = curr.prev
        return v

    @classmethod
    def expend_node(cls, name, node, split_mode, bins_mode='unique'):
        '''
        使用这个方法时, split_mode 必须指定为 contrib_III,contrib_IV 或类似结构的贡献度算法
        '''
        _ = cls(name, None, split_mode, bins_mode)
        _ = _.bins_mask  # 这两句为了生成这个 name 下的 mask
        nxt_l = list()
        for part in cls.MASK_MAPPING[name].keys():
            nxt_l.append(cls(name, node, split_mode, bins_mode))
            nxt_l[-1].part = part
            nxt_l[-1].select_mask()
        return nxt_l

    @classmethod
    def layer_traverse(cls, layer_names, split_mode, bins_mode='unique'):
        """Short summary.

        Parameters
        ----------
        cls : type
            Description of parameter `cls`.
        layer_names : list
            ['feature1', 'feature2', ..., 'featureN']
        split_mode : str
            contrib_III,contrib_IV
        bins_mode : str
            Description of parameter `bins_mode`.

        Returns
        -------
        type
            Description of returned object.

        """
        level = 0
        stack = deque([])
        while level < len(layer_names):
            if level == 0:
                next_l = cls.expend_node(layer_names[level], None, split_mode, bins_mode)
                stack.extend(next_l)
                level += 1
                continue
            N = len(stack)
            for i in range(N):
                curr = stack.popleft()
                next_l = cls.expend_node(layer_names[level], curr, split_mode, bins_mode)
                stack.extend(next_l)
            level += 1
        return stack


@BeamSearchMask.set_bins_mask_method('unique')
def mask_unique(self):
    unique_l = np.unique(self.x)
    mask_d = {}
    for x in unique_l:
        mask_d[x] = self.x == x
    return mask_d


@BeamSearchMask.register('contrib_I')
def select_contrib_I(self):
    mask_d = self.bins_mask
    E = np.nansum(self.y[self.DT_MASK_BASE])
    F = np.nansum(self.y[self.DT_MASK_TEST])
    F_E_E = (F - E)/E
    self.ringr = dict()
    self.ringr_contrib_v = dict()
    self.ringr_contrib_r = dict()
    self.ringr['total'] = F_E_E
    self.ringr_contrib_v['total'] = F_E_E
    self.ringr_contrib_r['total'] = 1.0
    for k, v in mask_d.items():
        Fi = np.nansum(self.y[self.DT_MASK_TEST & v])
        Ei = np.nansum(self.y[self.DT_MASK_BASE & v])
        self.ringr[k] = (Fi - Ei) / Ei
        self.ringr_contrib_v[k] = (Fi - Ei) / E
        self.ringr_contrib_r[k] = (Fi - Ei) / (F - E)


@BeamSearchMask.register('contrib_II')
def select_contrib_II(self):
    mask_d = self.bins_mask
    E = np.nanmean(self.y[self.DT_MASK_BASE])
    F = np.nanmean(self.y[self.DT_MASK_TEST])
    F_E_E = (F - E)/E
    self.ringr = dict()
    self.ringr_contrib_v = dict()
    self.ringr_contrib_r = dict()
    self.ringr['total'] = F_E_E
    self.ringr_contrib_v['total'] = F_E_E
    self.ringr_contrib_r['total'] = 1.0
    self.debug_d = dict()
    for k, v in mask_d.items():
        a = self.DT_MASK_BASE.sum()
        ai = np.nansum(self.DT_MASK_BASE & v) / a
        self.debug_d[f'{k}_ai'] = ai
        b = self.DT_MASK_TEST.sum()
        bi = np.nansum(self.DT_MASK_TEST & v) / b
        self.debug_d[f'{k}_bi'] = bi
        Ei = np.nanmean(self.y[self.DT_MASK_BASE & v])
        self.debug_d[f'{k}_Ei'] = Ei
        Fi = np.nanmean(self.y[self.DT_MASK_TEST & v])
        self.debug_d[f'{k}_Fi'] = Fi
        aiEi = ai * Ei
        self.debug_d[f'{k}_aiEi'] = aiEi
        biFi = bi * Fi
        self.debug_d[f'{k}_biFi'] = biFi
        self.ringr[k] = (Fi - Ei) / Ei
        self.ringr_contrib_v[k] = (biFi - aiEi) / E
        self.ringr_contrib_r[k] = (biFi - aiEi) / (F - E)


@BeamSearchMask.register('ab_contrib_II')
def select_ab_contrib_II(self):
    '''
    假设:
    - y: 0 代表通过, 1 代表拒绝 (对于一个通过率的分析, 从拒绝的角度分析是有效的)
    - 现象: TEST 时段相比 BASE 时段，B组通过率提升的幅度比A组小（B组拒绝率下降的幅度比A组小）
    - 结果:
        ringr_contrib_v 代表不同纬度的贡献值，正值代表对<现象>的贡献是正向的, 正值越大
        代表对<现象>的正向作用越大; 负值代表对<现象>的贡献是负向的, 负值越大代表对<现象>
        的负向作用越大;
    '''
    mask_d = self.bins_mask
    mask_base = self.DT_MASK_BASE
    mask_test = self.DT_MASK_TEST
    mask_A = self.mask_a
    mask_B = self.mask_b
    self.ringr_contrib_v_base = dict()
    self.ringr_contrib_v_test = dict()
    self.ringr_contrib_v = dict()
    self.debug_d = dict()
    for tname, tmask in [('base', mask_base), ('test', mask_test)]:
        _a_mask = tmask & mask_A
        _b_mask = tmask & mask_B
        E = np.nanmean(self.y[_a_mask])
        F = np.nanmean(self.y[_b_mask])
        a = (_a_mask).sum()
        b = (_b_mask).sum()
        self.debug_d[f'{tname}_E'] = E
        self.debug_d[f'{tname}_F'] = F
        self.debug_d[f'{tname}_a'] = a
        self.debug_d[f'{tname}_b'] = b
        for k, v in mask_d.items():
            ai = np.nansum(_a_mask & v) / a
            bi = np.nansum(_b_mask & v) / b
            Ei = np.nanmean(self.y[_a_mask & v])
            Fi = np.nanmean(self.y[_b_mask & v])
            aiEi = ai * Ei
            biFi = bi * Fi
            if tname == 'base':
                self.ringr_contrib_v_base[k] = (biFi - aiEi) / E
            elif tname == 'test':
                self.ringr_contrib_v_test[k] = (biFi - aiEi) / E
            self.debug_d[f'{tname}_{k}_ai'] = ai
            self.debug_d[f'{tname}_{k}_bi'] = bi
            self.debug_d[f'{tname}_{k}_Ei'] = Ei
            self.debug_d[f'{tname}_{k}_Fi'] = Fi
            self.debug_d[f'{tname}_{k}_aiEi'] = aiEi
            self.debug_d[f'{tname}_{k}_biFi'] = biFi
    for k, v in mask_d.items():
        self.ringr_contrib_v[k] = self.ringr_contrib_v_test[k] - self.ringr_contrib_v_base[k]


@BeamSearchMask.register('contrib_IV')
def select_contrib_IV(self):
    mask_d = self.bins_mask
    part_mask = mask_d[self.part]
    # self.mask = part_mask  # 级联的 mask 结合 self.prev_mask 生成
    prev_mask = self.prev_mask
    imask = prev_mask & part_mask
    self.mask = imask  # 保留最终级联结果 imask, 这个再和 self.prev_mask 级联时不改变结果
    E = np.nanmean(self.y[self.DT_MASK_BASE])
    F = np.nanmean(self.y[self.DT_MASK_TEST])
    F_E_E = (F - E)/E
    self.E, self.F, self.F_E_E = E, F, F_E_E
    self.ringr = dict()
    self.ringr_contrib_v = dict()
    self.ringr_contrib_r = dict()
    self.distribution_change_v = dict()
    self.distribution_change_r = dict()
    self.numeric_change_v = dict()
    self.numeric_change_r = dict()
    self.a = 1.0
    self.b = 1.0
    self.ai = np.nansum(self.DT_MASK_BASE & imask) / self.DT_MASK_BASE.sum()
    self.bi = np.nansum(self.DT_MASK_TEST & imask) / self.DT_MASK_TEST.sum()
    self.Ei = np.nanmean(self.y[self.DT_MASK_BASE & imask])
    self.Fi = np.nanmean(self.y[self.DT_MASK_TEST & imask])
    self.aiEi = self.ai * self.Ei
    self.biFi = self.bi * self.Fi
    self.E_F_b_a_2 = (self.Ei + self.Fi)*(self.bi - self.ai)/2
    self.a_b_F_E_2 = (self.ai + self.bi)*(self.Fi - self.Ei)/2
    self.ringr = (self.Fi - self.Ei) / self.Ei
    self.ringr_contrib_v = (self.biFi - self.aiEi) / self.E
    self.ringr_contrib_r = (self.biFi - self.aiEi) / (self.F - self.E)
    self.distribution_change_v = self.E_F_b_a_2 / (self.E)
    self.distribution_change_r = self.E_F_b_a_2 / (self.F - self.E)
    self.numeric_change_v = self.a_b_F_E_2 / self.E
    self.numeric_change_r = self.a_b_F_E_2 / (self.F - self.E)


@BeamSearchMask.set_bins_mask_method('bool')
def mask_bool(self):
    mask_d = {}
    mask_d['True'] = self.x > 0
    return mask_d


@BeamSearchMask.register('contrib_III')
def select_contrib_III(self):
    mask_d = self.bins_mask
    part_mask = mask_d[self.part]
    prev_mask = self.prev_mask
    imask = prev_mask & part_mask
    self.mask = imask  # 保留最终级联结果 imask, 这个再和 self.prev_mask 级联时不改变结果
    self.Ei = np.nanmean(self.x[prev_mask & self.DT_MASK_BASE])
    self.Fi = np.nanmean(self.x[prev_mask & self.DT_MASK_TEST])
    self.valuer = self.Fi / self.Ei
    self.ringr = self.valuer - 1
    self.log_Ei_Fi = np.log10(self.valuer)
    self.E_ = self.get_chain_product('Ei')
    self.F_ = self.get_chain_product('Fi')
    self.valuer_ = self.F_ / self.E_
    self.ringr_ = self.valuer_ - 1
    self.log_E_F = np.log10(self.valuer_)
    self.contrib_r = self.log_Ei_Fi / self.log_E_F


@BeamSearchMask.set_bins_mask_method('bipart')
def mask_bipart(self):
    rank = pd.Series(self.x).rank(pct=True)
    mask_d = {}
    mask_d['left'] = rank <= 0.5
    mask_d['right'] = rank > 0.5
    return mask_d


@BeamSearchMask.register('unbalance_pd_lift_diff')
def select_unbalance_pd_lift_diff(self):
    mask_d = self.bins_mask
    prev_mask = self.prev_mask
    mask1 = mask_d['left']
    mask2 = mask_d['right']
    chain_mask1 = mask1 & prev_mask
    chain_mask2 = mask2 & prev_mask
    min_group_ratio = np.min([chain_mask1.sum(), chain_mask2.sum()]) / len(chain_mask1)
    dt1c1m = self.DT_MASK1 & chain_mask1
    dt2c1m = self.DT_MASK2 & chain_mask1
    dt1c2m = self.DT_MASK1 & chain_mask2
    dt2c2m = self.DT_MASK2 & chain_mask2
    p11 = np.nanmean(self.y[dt1c1m]) if dt1c1m.sum() != 0 else 0.0
    p12 = np.nanmean(self.y[dt1c2m]) if dt1c2m.sum() != 0 else 0.0
    p21 = np.nanmean(self.y[dt2c1m]) if dt2c1m.sum() != 0 else 0.0
    p22 = np.nanmean(self.y[dt2c2m]) if dt2c2m.sum() != 0 else 0.0
    lift1 = (p11 - p21) / p21 if p21 > 0. else 0.
    lift2 = (p12 - p22) / p22 if p22 > 0. else 0.
    contrib1 = lift1
    contrib2 = lift2
    self.contrib1 = contrib1
    self.contrib2 = contrib2
    self.abs_contrib = np.abs(contrib1 - contrib2) * min_group_ratio
    # if contrib1 > contrib2:
    if np.abs(contrib1) > np.abs(contrib2):
        self.lift = lift1
        self.mask = mask1
        self.part = 'left'
    else:
        self.lift = lift2
        self.mask = mask2
        self.part = 'right'



def get_stat_of_chain_end(chain_end, fea_d, y):
    chain_l = chain_end.get_chain_list()
    stat_l_d = dict()
    for dt_part in ['test', 'base']:
        stat_l = list()
        cum_mask = None
        if dt_part == 'test':
            dt_mask = chain_end.DT_MASK1
        elif dt_part == 'base':
            dt_mask = chain_end.DT_MASK2
        for node in chain_l:
            a_min = np.nanmin(fea_d[node.name])
            a_max = np.nanmax(fea_d[node.name])
            v_min = fea_d[node.name][node.mask].min()
            v_max = fea_d[node.name][node.mask].max()
            # pre_cum_mask = cum_mask
            if cum_mask is None:
                samples_before = dt_mask.sum()
                cum_mask = node.mask
                samples_after = (dt_mask & cum_mask).sum()
            else:
                samples_before = samples_after
                cum_mask = cum_mask & node.mask
                samples_after = (dt_mask & cum_mask).sum()
            d = {
                'name': node.name,
                'v_min': v_min,
                'v_max': v_max,
                'a_min': a_min,
                'a_max': a_max,
                'samples_before': samples_before,
                'samples_after': samples_after,
                'part': node.part,
            }
            stat_l.append(d)
        stat_l_d[dt_part] = stat_l
    m1 = np.logical_and.reduce([chain_end.DT_MASK1]
                               + [x.mask for x in chain_l])
    m2 = np.logical_and.reduce([chain_end.DT_MASK2]
                               + [x.mask for x in chain_l])
    desc = {
        # 'bad_rate_test': y[m1].mean(),
        # 'bad_rate_base': y[m2].mean(),
        'bad_rate_test': np.nanmean(y[m1]),
        'bad_rate_base': np.nanmean(y[m2]),
        'sample_ratio_test': m1.sum() / chain_end.DT_MASK1.sum(),
        'sample_ratio_base': m2.sum() / chain_end.DT_MASK2.sum(),
        'sample_number_test': m1.sum(),
        'sample_number_base': m2.sum(),
    }
    return stat_l_d, desc


def get_top_lift_chain(fea_d, y, top_contrib_cols, min_population=500, beam_width=20, MAX_level=6, split_mode='lift_diff', _Searcher=BeamSearchLift):
    """Execute beamsearch based on BeamSearchLift.

    Parameters
    ----------
    fea_d : type
        Description of parameter `fea_d`.
    y : type
        Description of parameter `y`.
    top_contrib_cols : type
        Description of parameter `top_contrib_cols`.
    min_population : type
        Description of parameter `min_population`.
    beam_width : type
        Description of parameter `beam_width`.
    MAX_level : type
        Description of parameter `MAX_level`.
    split_mode : type
        Description of parameter `split_mode`.
    _Searcher : type
        Description of parameter `_Searcher`.

    Returns
    -------
    type
        Description of returned object.

    """
    curr_level = 0
    chain_l_d = defaultdict(list)
    while curr_level < MAX_level:
        if curr_level == 0:
            next_pool = [_Searcher(k, fea_d, y, None, split_mode) for k in top_contrib_cols]
            for n in tqdm.tqdm(next_pool):
                n.select_mask()
            next_pool = sorted(next_pool, reverse=True)
            # [x.name for x in next_pool]
            chain_list = next_pool[:beam_width]
            chain_l_d[curr_level] = next_pool[:beam_width]
        else:
            next_pool = []
            next_pool_tmp = []
            chain_fea_set_pool = []
            for curr in tqdm.tqdm(chain_list):
                for k in top_contrib_cols:
                    if k in list(curr.backward_names()):
                        continue
                    tmp = _Searcher(k, fea_d, y, curr, split_mode)
                    tmp_chain_fea_set = tmp.get_chain_fea_set()
                    if tmp_chain_fea_set in chain_fea_set_pool:  # 去掉重复
                        # print(f"{tmp_chain_fea_set} has in.")
                        continue
                    tmp.select_mask()
                    tmp_chain_mask = tmp.get_chain_mask()
                    mask_sum1 = (tmp_chain_mask & tmp.DT_MASK1).sum()
                    mask_sum2 = (tmp_chain_mask & tmp.DT_MASK2).sum()
                    # if 'dc_sina_pd_id_gender' in tmp_chain_fea_set:
                    #     print(f'curr_level: {curr_level}, tmp_chain_fea_set: {tmp_chain_fea_set}, mask_sum1: {mask_sum1}, mask_sum2: {mask_sum2}, tmp.lift: {tmp.lift}')
                    if (mask_sum1 < min_population) or (mask_sum2 < min_population):  # 去掉样本点太少的
                        continue
                    next_pool_tmp.append(tmp)
                    chain_fea_set_pool.append(tmp_chain_fea_set)
            next_pool_tmp = sorted(next_pool_tmp, reverse=True)
            for n1 in next_pool_tmp:
                n1_ok = True
                for n2 in next_pool:
                    intersec_len = len(n1.get_chain_fea_set().intersection(n2.get_chain_fea_set()))
                    if intersec_len >= curr_level:
                        n1_ok = False
                    # if (intersec_len > 0) & (intersec_len >= curr_level - 1):
                    #     n1_ok = False
                    # if intersec_len >= curr_level/2:
                    #     n1_ok = False
                    # if intersec_len > 0:
                    #     n1_ok = False
                if n1_ok:
                    next_pool.append(n1)
            chain_list = next_pool[:beam_width]
            chain_l_d[curr_level] = next_pool[:beam_width]
            print(f'len(next_pool_tmp): {len(next_pool_tmp)}, len(next_pool): {len(next_pool)}, len(chain_fea_set_pool): {len(chain_fea_set_pool)}')
        curr_level += 1
    return chain_l_d


def get_top_lift_chain_v2(top_contrib_cols, min_population=500, beam_width=20, MAX_level=6, split_mode='lift_diff', bins_mode='bipart', _Searcher=BeamSearchMask):
    """Execute beamsearch based on BeamSearchMask.

    Parameters
    ----------
    top_contrib_cols : type
        Description of parameter `top_contrib_cols`.
    min_population : type
        Description of parameter `min_population`.
    beam_width : type
        Description of parameter `beam_width`.
    MAX_level : type
        Description of parameter `MAX_level`.
    split_mode : type
        Description of parameter `split_mode`.
    bins_mode : type
        Description of parameter `bins_mode`.
    _Searcher : type
        Description of parameter `_Searcher`.

    Returns
    -------
    type
        Description of returned object.

    """
    curr_level = 0
    chain_l_d = defaultdict(list)
    while curr_level < MAX_level:
        if curr_level == 0:
            next_pool = [_Searcher(k, None, split_mode, bins_mode) for k in top_contrib_cols]
            for n in tqdm.tqdm(next_pool):
                n.select_mask()
            next_pool = sorted(next_pool, reverse=True)
            # [x.name for x in next_pool]
            chain_list = next_pool[:beam_width]
            chain_l_d[curr_level] = next_pool[:beam_width]
        else:
            next_pool = []
            next_pool_tmp = []
            chain_fea_set_pool = []
            for curr in tqdm.tqdm(chain_list):
                for k in top_contrib_cols:
                    if k in list(curr.backward_names()):
                        continue
                    tmp = _Searcher(k, curr, split_mode, bins_mode)
                    tmp_chain_fea_set = tmp.get_chain_fea_set()
                    if tmp_chain_fea_set in chain_fea_set_pool:  # 去掉重复
                        # print(f"{tmp_chain_fea_set} has in.")
                        continue
                    tmp.select_mask()
                    tmp_chain_mask = tmp.get_chain_mask()
                    mask_sum1 = (tmp_chain_mask & tmp.DT_MASK1).sum()
                    mask_sum2 = (tmp_chain_mask & tmp.DT_MASK2).sum()
                    # if 'dc_sina_pd_id_gender' in tmp_chain_fea_set:
                    #     print(f'curr_level: {curr_level}, tmp_chain_fea_set: {tmp_chain_fea_set}, mask_sum1: {mask_sum1}, mask_sum2: {mask_sum2}, tmp.lift: {tmp.lift}')
                    if (mask_sum1 < min_population) or (mask_sum2 < min_population):  # 去掉样本点太少的
                        continue
                    next_pool_tmp.append(tmp)
                    chain_fea_set_pool.append(tmp_chain_fea_set)
            next_pool_tmp = sorted(next_pool_tmp, reverse=True)
            for n1 in next_pool_tmp:
                n1_ok = True
                for n2 in next_pool:
                    intersec_len = len(n1.get_chain_fea_set().intersection(n2.get_chain_fea_set()))
                    if intersec_len >= curr_level:
                        n1_ok = False
                    # if (intersec_len > 0) & (intersec_len >= curr_level - 1):
                    #     n1_ok = False
                    # if intersec_len >= curr_level/2:
                    #     n1_ok = False
                    # if intersec_len > 0:
                    #     n1_ok = False
                if n1_ok:
                    next_pool.append(n1)
            chain_list = next_pool[:beam_width]
            chain_l_d[curr_level] = next_pool[:beam_width]
            print(f'len(next_pool_tmp): {len(next_pool_tmp)}, len(next_pool): {len(next_pool)}, len(chain_fea_set_pool): {len(chain_fea_set_pool)}')
        curr_level += 1
    return chain_l_d
