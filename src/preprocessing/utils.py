# -*- encoding: utf-8 -*-
'''
@Time       : 2020/07/15 11:03:22
@Author     : Yize Jiang
@Email      : yize.jiang@galixir.com
@File       : utils.py
@Project    : X-DPI
@Description: 预处理所需的工具函数
'''


def one_of_k_encoding(x, allowable_set):
    # if x not in allowable_set:
    #     raise Exception("input {0} not in allowable set{1}:".format(
    #         x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]
