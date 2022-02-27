#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:44:01 2022

@author: mariapalazzi
"""
from numpy.random import rand
from typing import List

from .mod_param import mod_param


def heterogenousBlockSizes(B: int, N: int, min_block_size: int = 0, alpha: float = 2.5) -> List[int]:
    """
    This function will generate heterogenous block sizes, following a powerlaw distribution.
    As in Clauset A, et al. SIAM Rev., 51(4), 661–703 2009.

    The transformation method is used to generate x samples from a powerlaw distribution.
    These samples will be the sizes of the blocks across each dimension, the functions is called
    for col and rows, separately.
    Two elements are needed: uniformly distributed random source r where 0 <= r < 1 and
    the functional inverse of the cumulative density function (CDF) of the power-law distribution

    The source r is simply given as a parameter to the CDF.

    the inputs are:

    N: int
        numnber of nodes to split into block
    B: number
        number of blocks on which the nodes will be divided
    x_min: int
        be the lower bound, i.e, minimum block size, default 10% of N
    alpha: float bounded in (x1,x2)
        be the scaling parameter of the distribution, default 2.5
    output:

    colsizes: list of ints
        a list containing B random numbers determining the size of each block
    """
    if min_block_size == 0:
        min_block_size = int(N * 0.1)
    if alpha == 0:
        if N % B != 0:
            raise ValueError(f"The number of nodes must be divisible by B. {N} % {B} = {N%B}")
        else:
            return [mod_param(N, N, B)[0]] * B
    if 3 <= alpha or alpha < 1:
        raise ValueError("alpha must be between (1,3)")
    clsum = 1
    maxn = N
    while (clsum != N) or (maxn >= N / 2):
        r = rand(B)
        x_smp = min_block_size * (1 - r) ** (-1 / (alpha - 1))
        colsizes = x_smp.round(0).astype(int).tolist()
        clsum = sum(colsizes)
        maxn = max(colsizes)
    colsizes.sort(reverse=True)
    return colsizes
