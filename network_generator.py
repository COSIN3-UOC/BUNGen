#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:11:07 2017

@author: maria
"""
#%%
import numpy as np
#%%
def ballcurve(x,xi):
    '''
    function to generate the curve for the nested structure, given a shape
    parameter xi. If xi= 1 is linear.
    input:
    ----------
    x: 1D array, [0,1]
        initial values to be evaluated on the function
    xi: number, >=1
        shape parameter of how stylised is the curve
    output:
    ----------
    y: 1D array, [0,1]
        evaluated function
    '''
    y=1-(1-(x)**(1/xi))**xi
    return y
#%%
def mod_param(rows,cols,B):
    '''
    function to obtain the number of rows and cols belonging to a block, given the
    total size of the initial matrix and the number of blocks. It was made this way
    so it also works on not squared matrices and to preserve the blocks along the
    matrix's diagonal.

    inputs:
    ----------
    rows: int
        number of rows of the main matrix
    cols: int
        number of cols of the main matrix
    B: number
        number of blocks on which the main matrix will be divided

    output:
    ----------
    cx: int
        number of cols of each block
    cy: int
        number of rows of each block
    '''
    N=(rows+cols) #size matrix
    Nb=int(N//B) #size of blocks
    pr=0.5 if (rows==cols) else( rows/N )
    cx=int(Nb*(1-pr))
    cy=int(Nb*pr)
    return cy,cx
#%%
def network_generator(*args):
    '''
    function to generate synthetic networks with nested, modular and in-block nested structures. Generates
    networks with a fixed block size and increasing number of blocks (hence, increasing network size), 
    instead of networks with fixed size. This benchmark is a modification of the one introduced by 
    ASR et al (PRE 2018). If the number of columns nodes is not given, the function will assume that we want 
    to generate a unipartite network. The parameters must be passed respecting the following order.

    inputs:
    ----------
    rw: int >1
        number of row nodes that form a block
    cl: int >1
        number of col nodes that form a block
    B: number >=1
        number of blocks on which the main matrix will be divided
    xi: number, >=1
        shape parameter to indicate how stylised is the nested curve
    p: number in [0, 1]
        paramteter that control the amount of noise outside a perfectly nested structure
    mu: number in [0, 1]
        parameteter that control the amount of noise outside the blocks

    output:
    ----------
    M: array
        The synthetic network matrix with the predefined structure
        
    example:
    ---------
    network_matrix=network_generator(rw,cl,B,xi,p,mu)
    
    '''
    if len(args)==5:
        bipartite=False
        rw=cl=args[0]
        b=args[1]
        xi=args[2]
        P=args[3]
        mu=args[4]
    if len(args)==6:
        bipartite=True
        rw=args[0]
        cl=args[1]
        b=args[2]
        xi=args[3]
        P=args[4]
        mu=args[5]

    if rw < 3 or cl <3:
        raise ValueError('MATRIX TOO SMALL: row and col sizes should be larger than 3')
    else:
        Mij=np.random.uniform(0,1,size=(int(rw*b),int(cl*b)))
        cy,cx=mod_param(int(rw*b),int(cl*b),b)
        M_no=np.zeros(Mij.shape)
        le=[]
        Pi = ((b-1)*mu)/b
        lb=0
        for ii in range(int(b)):
            #for each block generate a nested structure
            j,i=np.indices(M_no[cy*ii:cy*(ii+1),cx*ii:cx*(ii+1)].shape)
            H=((j[::-1,:]+1)/cy)>ballcurve((i/cx),xi) #heaviside function to produce the nested structure
            M_no[cy*ii:cy*(ii+1),cx*ii:cx*(ii+1)]=H
            le+=[M_no[cy*ii:cy*(ii+1),cx*ii:cx*(ii+1)].sum()]
            lb+=(cy*cx)
        #generate the nested structure of the remaining block
        j,i=np.indices(M_no[(ii+1)*cy:,(ii+1)*cx:].shape)
        H=((j[::-1,:]+1)/j.shape[0])>ballcurve((i/i.shape[1]),xi)  #heaviside function to produce the nested structure
        M_no[(ii+1)*cy:,(ii+1)*cx:]=H
        le+=[M_no[(ii+1)*cy:,(ii+1)*cx:].sum()]
        lb+=((int(rw*b)-((ii+1)*cy))*(int(cl*b)-((ii+1)*cx)))
        Et=M_no.sum(dtype=int)
    
        p_inter=(mu*Et)/(lb*b) if ((lb*b)!=0) else 0 #prob of having a link outside blocks
        M_no[M_no==0]=p_inter
        for ix in range(int(b)):
            j,i=np.indices(M_no[cy*ix:cy*(ix+1),cx*ix:cx*(ix+1)].shape)
            Pr=(P*le[ix])/((cx*cy)-le[ix]+(P*le[ix])) if (((cx*cy)-le[ix]+(P*le[ix]))!=0) else 0
            H=((j[::-1,:]+1)/cy)>ballcurve((i/cx),xi)  #heaviside function to produce the nested structure
            p_intra=((1-P+(P*Pr))*H + Pr*(1-H))*(1-Pi)  #prob of having a link within blocks
            M_no[cy*ix:cy*(ix+1),cx*ix:cx*(ix+1)]= p_intra
        
        #calculate to the remaining block
        j,i=np.indices(M_no[(ix+1)*cy:,(ix+1)*cx:].shape)
        Pr=(P*le[ix+1])/(((int(rw*b)-(ix+1)*cy)*(int(cl*b)-(ix+1)*cx))-le[ix+1]+(P*le[ix+1])) if (le[ix+1]>0)&(P!=0) else 0
        H=((j[::-1,:]+1)/j.shape[0])>ballcurve((i/i.shape[1]),xi) # #heaviside function to produce the nested structure
        p_intra=((1-P+(P*Pr))*H + Pr*(1-H))*(1-Pi)  #prob of having a link within blocks
        M_no[(ix+1)*cy:,(ix+1)*cx:]=p_intra
        if bipartite==True:
            M=(M_no>Mij).astype(int)
        else:
            M=(M_no>Mij).astype(int)
            np.fill_diagonal(M, 0)
            M=np.triu(M,k=1)+(np.triu(M,k=1)).T

    return M