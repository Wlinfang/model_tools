from copy import deepcopy
import numpy as np
from typing import Callable, Mapping
from wejoy_analysis.component.algorithm.beamsearch import BeamSearchLift, BeamSearchMask, get_top_lift_chain, get_top_lift_chain_v2


class ContribRanker:
    # BASE_SEARCHER = type('ContribRanker_BASE_SEARCHER', (BeamSearchMask,), {})
    BASE_SEARCHER = BeamSearchMask

    def __init__(self,
                 dt_mask1,
                 dt_mask2,
                 min_population: int = 100,
                 MAX_level: int = 1,
                 beam_width: int = 0,
                 split_mode='unbalance_pd_prop_diff',
                 bins_mode='bipart',
                 **kwargs):
        """Ranking of anomaly features in conjunction with business

        Parameters
        ----------
        dt_mask1 : type, numpy vector
            mask for samples which are abnormal.
        dt_mask2 : type, numpy vector
            mask for samples which are normal.
        min_population : int
            minimal sample number per split.
        MAX_level : int
            beam search MAX_level.
        beam_width : int
            beam search candidate width.
        split_mode : type
            choose split method to calculate the contribution of a feature.
        **kwargs : type
            Description of parameter `**kwargs`.

        Returns
        -------
        type, self
            Description of returned object.

        """
        self._Searcher = type(
            '_Searcher',
            (self.BASE_SEARCHER,),
            {
                'MASK_MAPPING': dict(),
                'fea_d': dict(),
                'y': None
            }
        )
        self.dt_mask1 = dt_mask1
        self.dt_mask2 = dt_mask2
        self._Searcher.set_dt_mask(dt_mask1, dt_mask2)
        self.min_population = min_population
        self.MAX_level = MAX_level
        self.beam_width = beam_width
        if isinstance(split_mode, str):
            assert split_mode in self._Searcher.SPLIT_MAPPING
            self.split_mode = split_mode
        elif isinstance(split_mode, Callable):
            self.split_mode = split_mode.__name__
            self._Searcher.register(self.split_mode)(split_mode)
        if isinstance(bins_mode, str):
            assert bins_mode in self._Searcher.BINS_MASK_METHOD_MAPPING
            self.bins_mode = bins_mode
        elif isinstance(bins_mode, Callable):
            self.bins_mode = bins_mode.__name__
            self._Searcher.set_bins_mask_method(self.bins_mode)(bins_mode)

    @classmethod
    def register(cls, name):
        def _thunk(func):
            cls.BASE_SEARCHER.register(name)(func)
            return func
        return _thunk

    @classmethod
    def set_bins_mask_method(cls, name):
        def _thunk(func):
            cls.BASE_SEARCHER.set_bins_mask_method(name)(func)
        return _thunk

    def fit(self, fea_d: Mapping, y: np.ndarray, **kwargs):
        """Short summary.

        Parameters
        ----------
        fea_d : Mapping
            key: feature name, value: numpy vector of this feature
            {'feature1': np.array(1,2,...,N), 'feature2': np.array(1,2,...,N)}.
        y : type, numpy vector
            business label.

        Returns
        -------
        type
            Description of returned object.

        """
        self._Searcher.set_X_y(fea_d, y)
        # 可以设置一些全局的其他参数
        for k, v in kwargs.items():
            setattr(self._Searcher, k, v)
        fea_keys = list(fea_d.keys())
        _beam_width = len(fea_keys) if not self.beam_width else self.beam_width
        self.multi_level_chain_list = get_top_lift_chain_v2(
            fea_keys,
            self.min_population,
            _beam_width,
            self.MAX_level,
            split_mode=self.split_mode,
            bins_mode=self.bins_mode,
            _Searcher=self._Searcher)
        return self
