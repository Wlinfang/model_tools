from collections import defaultdict
import hashlib
import numpy as np
import tqdm


def get_bins_edges(v1, max_bins):
    finite_num = np.isfinite(v1).sum()
    sorted_v1 = np.sort(v1)[:finite_num]
    step = finite_num // max_bins
    if step <= 1:
        raw = sorted_v1
    else:
        raw = sorted_v1[::step]
    effect_edges = list([raw[0]])
    for x in raw:
        if x == effect_edges[-1]:
            continue
        effect_edges.append(x)
    return np.array(effect_edges)[1:]


class Tree:
    def __init__(self, root=None, depth=0):
        self.root = root
        self.depth = depth

class Node:
    fea_d = dict()
    y = None
    BINS_EDGES = dict()
    FEA_TYPES = dict()
    max_bins = None
    DT_MASK_TEST = None
    DT_MASK_BASE = None
    SPLIT_MAPPING = dict()
    DEFAULT_CRITERION = 'contrib_II'
    def __init__(self, name, op, value, prev=None, next=None, criterion=None):
        '''
        op: <, >=, ==
        op:'==', value:'np.nan' 时, 使用 np.isnan
        '''
        self.name = name
        self.op = op
        self.value = value
        self.prev = prev
        self.next = next
        self._score = None
        self.criterion = self.DEFAULT_CRITERION if criterion is None else criterion

    @property
    def rule_name(self):
        return f'{self.name}{self.op}{self.value}'

    def get_chain_list(self):
        back_chain_l = []
        cur = self
        while cur is not None:
            back_chain_l.append(cur)
            cur = cur.prev
        return list(reversed(back_chain_l))

    @property
    def chain_rule_names(self):
        chain_l = self.get_chain_list()
        return [x.rule_name for x in chain_l]

    @property
    def hash_name(self):
        names = self.chain_rule_names
        return '$'.join(sorted(list(set(names))))

    @property
    def seq_rule_name(self):
        names = self.chain_rule_names
        return '$'.join(names)

    def __repr__(self):
        return f'Node:{self.rule_name}'

    def __hash__(self):
        return int(hashlib.md5(self.hash_name.encode('utf8')).hexdigest(),16)

    def __eq__(self, other):
        # another object is equal to self, iff
        # it is an instance of MyClass
        return isinstance(other, Node)

    @classmethod
    def set_X_y(cls, fea_d, y, max_bins=7):
        cls.fea_d = fea_d
        cls.y = y
        cls.max_bins = max_bins
        for k, v in fea_d.items():
            if v.dtype.type is np.float_:
                cls.FEA_TYPES[k] = np.float_
                cls.BINS_EDGES[k] = get_bins_edges(v, max_bins)
            elif v.dtype.type is np.str_:
                cls.FEA_TYPES[k] = np.str_
                cls.BINS_EDGES[k] = np.sort(np.unique(v))
                cat_num = len(cls.BINS_EDGES[k])
                if cat_num > max_bins:
                    logger.warning(f"category of {k} is {cat_num} > {max_bins}(max_bins).")
            else:
                raise Exception('Unkown type, value of fea_d must be np.float_ or np.str_')

    @classmethod
    def set_dt_mask(cls, dt_mask_base, dt_mask_test):
        cls.DT_MASK_BASE = dt_mask_base
        cls.DT_MASK_TEST = dt_mask_test

    @classmethod
    def register(cls, name):
        def _thunk(func):
            cls.SPLIT_MAPPING[name] = func
            return func
        return _thunk

    @classmethod
    def clear(cls):
        cls.fea_d = dict()
        cls.y = None
        cls.BINS_EDGES = dict()
        cls.FEA_TYPES = dict()
        cls.max_bins = None
        cls.DT_MASK_TEST = None
        cls.DT_MASK_BASE = None
        # cls.SPLIT_MAPPING = dict()
        cls.DEFAULT_CRITERION = 'contrib_II'

    @property
    def mask(self):
        if self.op == '<':
            m = self.fea_d[self.name] < self.value
        elif self.op == '>=':
            m = self.fea_d[self.name] >= self.value
        elif self.op == '==':
            if isinstance(self.value, str):
                m = self.fea_d[self.name] == self.value
            else:
                m = np.isnan(self.fea_d[self.name])
        elif self.op == '!=':
            if isinstance(self.value, str):
                m = self.fea_d[self.name] != self.value
            else:
                m = ~np.isnan(self.fea_d[self.name])
        # elif self.op == '0':
        #     m = np.zeros(self.y.shape).astype(bool)
        else:
            raise Exception('Unkown op')
        return m

    @property
    def chain_mask(self):
        if self.prev is None:
            return self.mask
        else:
            return self.mask & self.prev.chain_mask

    @property
    def sample_num(self):
        return self.chain_mask.sum()

    @property
    def total_num(self):
        return len(self.y)

    @property
    def sample_ratio(self):
        return self.sample_num / self.total_num

    @property
    def base_value(self):
        vmask = self.chain_mask
        base_mask = self.DT_MASK_BASE
        mask_E = vmask & base_mask
        Ei = np.nanmean(self.y[mask_E]) if mask_E.sum() > 0 else 0.0
        return Ei

    @property
    def test_value(self):
        vmask = self.chain_mask
        test_mask = self.DT_MASK_TEST
        mask_F = vmask & test_mask
        Fi = np.nanmean(self.y[mask_F]) if mask_F.sum() > 0 else 0.0
        return Fi

    @property
    def base_ratio(self):
        vmask = self.chain_mask
        base_mask = self.DT_MASK_BASE
        mask_E = vmask & base_mask
        return mask_E.sum() / base_mask.sum()

    @property
    def test_ratio(self):
        vmask = self.chain_mask
        test_mask = self.DT_MASK_TEST
        mask_F = vmask & test_mask
        return mask_F.sum() / test_mask.sum()

    def get_score(self):
        # return np.random.rand()
        _split_func = self.SPLIT_MAPPING[self.criterion]
        return _split_func(self)

    @property
    def score(self):
        if self._score is None:
            self._score = self.get_score()
        return self._score

    def __lt__(self, other):
        return self.score < other.score


@Node.register('contrib_II')
def _contrib_II(self):
    vmask = self.chain_mask
    base_mask = self.DT_MASK_BASE
    test_mask = self.DT_MASK_TEST
    E = np.nanmean(self.y[base_mask])
    F = np.nanmean(self.y[test_mask])
    a = (base_mask).sum()
    b = (test_mask).sum()
    mask_E = vmask & base_mask
    mask_F = vmask & test_mask
    Ei = np.nanmean(self.y[mask_E]) if mask_E.sum() > 0 else 0.0
    Fi = np.nanmean(self.y[mask_F]) if mask_F.sum() > 0 else 0.0
    ai = mask_E.sum() / a
    bi = mask_F.sum() / b
    F_E = F - E
    F_E_E = (F - E)/E
    aiEi = ai * Ei
    biFi = bi * Fi
    score = (biFi - aiEi) / F_E
    return score


@Node.register('ab_contrib_II')
def _ab_contrib_II(self):
    vmask = self.chain_mask
    mask_base = self.DT_MASK_BASE
    mask_test = self.DT_MASK_TEST
    mask_A = self.mask_a
    mask_B = self.mask_b
    for tname, tmask in [('base', mask_base), ('test', mask_test)]:
        _a_mask = tmask & mask_A
        _b_mask = tmask & mask_B
        E = np.nanmean(self.y[_a_mask])
        F = np.nanmean(self.y[_b_mask])
        a = (_a_mask).sum()
        b = (_b_mask).sum()
        # for k, v in mask_d.items():
        ai = np.nansum(_a_mask & vmask) / a
        bi = np.nansum(_b_mask & vmask) / b
        Ei = np.nanmean(self.y[_a_mask & vmask]) if (_a_mask & vmask).sum() > 0 else 0.0
        Fi = np.nanmean(self.y[_b_mask & vmask]) if (_b_mask & vmask).sum() > 0 else 0.0
        aiEi = ai * Ei
        biFi = bi * Fi
        if tname == 'base':
            ringr_contrib_v_base = (biFi - aiEi) / E
        elif tname == 'test':
            ringr_contrib_v_test = (biFi - aiEi) / E
    ringr_contrib_v = ringr_contrib_v_test - ringr_contrib_v_base
    return ringr_contrib_v


def convert_fea_d(fea_d):
    ret = dict()
    for k, v in fea_d.items():
        if (v.dtype.type is np.float_) or (v.dtype.type is np.str_):
            ret[k] = v
            continue
        try:
            x = v.astype(float)
        except:
            x = v.astype(str)
        ret[k] = x
    return ret


def split(prev, min_samples_leaf, Node, choose_high_score=True):
    if choose_high_score:
        _is_better = lambda x, y: x > y
    else:
        _is_better = lambda x, y: x < y
    node_l = []
    for name in Node.fea_d.keys():
        tmp_node = None
        if Node.FEA_TYPES[name] is np.float_:
            edges = Node.BINS_EDGES[name].tolist() + [np.nan]
        elif Node.FEA_TYPES[name] is np.str_:
            edges = Node.BINS_EDGES[name].tolist()
        else:
            raise Exception('Unkown type')
        for th in edges:
            if isinstance(th, str):
                op1, op2 = '==', '!='
            elif isinstance(th, float):
                if np.isnan(th):
                    op1, op2 = '==', '!='
                else:
                    op1, op2 = '<', '>='
            left_node = Node(name, op1, th, prev=prev)
            right_node = Node(name, op2, th, prev=prev)
            if (left_node.sample_num < min_samples_leaf) or (right_node.sample_num < min_samples_leaf):
                continue
            if tmp_node is None:
                tmp_node = left_node
            if tmp_node is not None:
                if _is_better(left_node.score, tmp_node.score):
                    tmp_node = left_node
                if _is_better(right_node.score, tmp_node.score):
                    tmp_node = right_node
        if tmp_node is not None:
            node_l.append(tmp_node)
    return node_l


def enum_split(prev, min_samples_leaf, Node, beam_width, choose_high_score=True):
    reverse = choose_high_score
    if choose_high_score:
        _is_better = lambda x, y: x > y
    else:
        _is_better = lambda x, y: x < y
    node_l = []
    for name in Node.fea_d.keys():
        tmp_node_l = list()
        if Node.FEA_TYPES[name] is np.float_:
            edges = Node.BINS_EDGES[name].tolist() + [np.nan]
            for th in edges:
                if np.isnan(th):
                    op1, op2 = '==', '!='
                else:
                    op1, op2 = '<', '>='
                left_node = Node(name, op1, th, prev=prev)
                right_node = Node(name, op2, th, prev=prev)
                if (left_node.sample_num < min_samples_leaf) or (right_node.sample_num < min_samples_leaf):
                    continue
                tmp_node = left_node
                if _is_better(right_node.score, tmp_node.score):
                    tmp_node = right_node
                tmp_node_l.append(tmp_node)
        elif Node.FEA_TYPES[name] is np.str_:
            edges = Node.BINS_EDGES[name].tolist()
            for th in edges:
                op1, op2 = '==', '!='
                left_node = Node(name, op1, th, prev=prev)
                right_node = Node(name, op2, th, prev=prev)
                if (left_node.sample_num < min_samples_leaf) or (right_node.sample_num < min_samples_leaf):
                    continue
                tmp_node_l.append(left_node)
        else:
            raise Exception('Unkown type')
        node_l += sorted(list(set(tmp_node_l)), reverse=reverse)[:beam_width]
    node_l = sorted(list(set(node_l)), reverse=reverse)[:beam_width]
    return node_l


def get_top_node_rules(
    beam_width,
    max_depth,
    splitter,
    Node,
    min_samples_split=2,
    min_samples_leaf=1,
    choose_high_score=True
):
    reverse = choose_high_score
    level_nodes = defaultdict(list)
    level_nodes[-1].append(None)
    for i in range(max_depth):
        for prev in level_nodes[i-1]:
            if splitter == 'bipart':
                tmp_l = split(prev, min_samples_leaf, Node, choose_high_score)
            elif splitter == 'enum':
                tmp_l = enum_split(prev, min_samples_leaf, Node, beam_width, choose_high_score)
            level_nodes[i] += tmp_l
            # for x in tmp_l:
            #     print(f'{x.seq_rule_name}: {x.score}')
        level_nodes[i] = sorted(list(set(level_nodes[i])), reverse=reverse)[:beam_width]
    return level_nodes


def show_level_nodes(level_nodes):
    for k, nodes in level_nodes.items():
        if k < 0:
            continue
        print(f'----------- level [{k}] -----------')
        for i, x in enumerate(nodes):
            print(f'{i}: score: {x.score:.6f}, count: {x.sample_num}, rule: {x.seq_rule_name}')


def plot_tree(lns, figsize=(8, 6)):
    filename = 'tree_gv'
    from graphviz import Digraph, Source
    id_names = set()
    e = Digraph(filename, filename=filename, format='png')

    def _get_name(i, node, mode='label'):
        if mode == 'label':
            lname = f'[{i}]{node.rule_name}\n'
            lname += f'score: {node.score:.6f}\n'
            lname += f'sample_num: {node.sample_num}, '
            lname += f'sample_ratio: {node.sample_ratio:.6f}\n'
            lname += f'base_value: {node.base_value:.6f}, test_value: {node.test_value:.6f}\n'
            lname += f'base_ratio: {node.base_ratio:.6f}, test_ratio: {node.test_ratio:.6f}'
        elif mode == 'id':
            lname = f'[{i}]{node.rule_name}'
        return lname

    for i, nodes in lns.items():
        if i < 0:
            continue
        for node in nodes:
            hname = _get_name(i, node, mode='id')
            lname = _get_name(i, node, mode='label')
            if lname not in id_names:
                id_names.add(lname)
                e.attr('node', shape='box')
                e.node(hname, lname, shape='box')
            if i == 0:
                continue
            prev_hname = _get_name(i-1, node.prev, mode='id')
            e.edge(prev_hname, hname)
    e.render()
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread(f'{filename}.png')
    plt.figure(figsize=figsize)
    # imgplot = plt.imshow(img)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


class Ranker:
    BASE_NODE = Node
    def __init__(
        self,
        dt_mask_base,
        dt_mask_test,
        criterion: str = 'contrib_II',
        splitter: str = 'bipart',
        max_bins: int = 7,
        beam_width: int = 3,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        choose_high_score: bool = True,
        **kwargs
    ):
        self._Node = type(
            '_Node',
            (self.BASE_NODE,),
            {
                'fea_d': dict(),
                'y': None,
                'DEFAULT_CRITERION': criterion
            }
        )
        # 可以设置一些全局的其他参数
        for k, v in kwargs.items():
            setattr(self._Node, k, v)
        self.dt_mask_base = dt_mask_base
        self.dt_mask_test = dt_mask_test
        self.criterion = criterion
        self.splitter = splitter
        self.max_bins = max_bins
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.choose_high_score = choose_high_score

    @classmethod
    def register(cls, name):
        def _thunk(func):
            cls.BASE_NODE.register(name)(func)
            return func
        return _thunk

    def fit(self, fea_d, y, **kwargs):
        self.fea_d = convert_fea_d(fea_d)
        self.y = y
        self._Node.set_X_y(self.fea_d, self.y, max_bins=self.max_bins)
        self._Node.set_dt_mask(self.dt_mask_base, self.dt_mask_test)
        self.level_nodes = get_top_node_rules(
            beam_width=self.beam_width,
            max_depth=self.max_depth,
            splitter=self.splitter,
            Node=self._Node,
            min_samples_leaf=self.min_samples_leaf,
            choose_high_score=self.choose_high_score
        )
        return self

    def show_level_nodes(self):
        show_level_nodes(self.level_nodes)

    def plot_tree(self, figsize=(8, 6)):
        plot_tree(self.level_nodes, figsize)
