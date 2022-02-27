#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:36:46 2022

@author: mariapalazzi
"""
from numpy import linspace
from scipy.interpolate import interp1d
from typing import List
from .xiConnRelationship import xiConnRelationship


def xiFunConn(
    rowsList: List[int], colsList: List[int], rowTot: int, colTot: int, C: float
) -> float:
    """
    This function estimates the xi value for a matrix of given size, connectance
    and distribution of blocks sizes

    inputs:
    rowsList: list of int,
        containing the size of each block on rows dimension
    colsList: list of int,
        containing the size of each block on cols dimension
    rowTot: int,
        total number of rows nodes
    colTot: int,
        total number of cols nodes
    C: float
        the global connectance;

    output:
     xi: float
         the matrix nestedness perfile;
    """

    # total number of links
    links = C * rowTot * colTot

    xiList = linspace(0.001, 5, 100)
    edgeList = []
    for xi in xiList:
        E = 0.0  # edge counter
        # block loop
        for i in range(len(rowsList)):
            blockRow = rowsList[i]
            blockCol = colsList[i]
            # block connectance
            connBlock = xiConnRelationship(blockRow, blockCol, xi)
            # block edges
            edgeBlock = connBlock * blockRow * blockCol
            E += edgeBlock
        edgeList.append(E)

    # interpolation
    f = interp1d(edgeList, xiList)

    # approximate input edge number to the interpolation range
    if links < min(edgeList):
        links = min(edgeList)
    elif links > max(edgeList):
        links = max(edgeList)

    # output xi
    xi = f([links])[0]
    xi = round(xi, 3)

    return xi
