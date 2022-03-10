#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:18:06 2022

@author: mariapalazzi
"""


def xiConnRelationship(rw: int, cl: int, xi: float) -> int:
    """
     Thi function calculates the connectance of a matrix (assuming B=1) for a given
     nested profile given by the shape parameter xi

     inputs:
    cl: (int)
         column number;
     M: (int)
         row number;
     xi: (float)
         nested profile;

     OUTPUT
     C: (float)
         matrix connectance;
    """
    E = 0  # edge counter
    for i in range(cl):
        x = i / cl  # tessellate
        y = 1 - (1 - x ** (1 / xi)) ** xi  # unite ball equation
        for j in range(rw):
            if j / rw >= y:
                E += 1

    return E
