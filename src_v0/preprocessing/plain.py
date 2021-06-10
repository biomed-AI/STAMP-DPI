# -*- encoding: utf-8 -*-
'''
@Time       : 2020/07/15 11:55:43
@Author     : Yize Jiang
@Email      : yize.jiang@galixir.com
@File       : plain.py
@Project    : X-DPI
@Description: 一维特征处理
'''

import json
from functools import lru_cache

import numpy as np

from .const import RNG


@lru_cache(None)
def get_rd200_feat(smiles):
    rd200_feat = RNG.process(smiles)
    if rd200_feat is not None and rd200_feat[0] is True:
        result = np.array(rd200_feat[1:])
        for i in [39, 41, 43, 45]:
            if np.isnan(result[i]) or np.isinf(result[i]):
                result[i] = 0.0
        return result
    else:
        raise RuntimeError('RD200 SMILES Error: {}'.format(smiles))


class PlainFeatureGenerator:
    def __init__(self, data_path='/gxr/jiangyize/data/DTC-Extend-Features/'):
        rd200_feat_dict = {}
        for i in range(1, 7):
            file_name = 'dtc_extend_rdkit200_norm_feat_{}.json'.format(i)
            temp_dict = json.load(open(data_path + file_name, 'r'))
            rd200_feat_dict = {**rd200_feat_dict, **temp_dict}
            print('Load {} Finish.'.format(file_name))

        self.rd200_feat_dict = rd200_feat_dict

    def get_features(self, smiles):
        return self.rd200_feat_dict[smiles]
