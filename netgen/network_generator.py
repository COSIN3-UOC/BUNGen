"""
Created on Thu Sep 14 13:11:07 2017

@author: maria
"""
from numpy import fill_diagonal, indices, triu, zeros
from numpy.random import uniform
from numpy.typing import ArrayLike

from .mod_param import mod_param
from .ballcurve import ballcurve


def network_generator(
    rw: int,
    cl: int,
    b: float,
    xi: float,
    P: float,
    mu: float,
    bipartite: bool,
) -> ArrayLike:
    """
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
    bipartite:
           a boolean to indicate if you want to generate bipartite (True) or unipartite (False) networks
    output:
    ----------
    M: array
        The synthetic network matrix with the predefined structure

    example:
    ---------
    network_matrix=network_generator(rw,cl,B,xi,p,mu)

    """

    if rw < 3 or cl < 3:
        raise ValueError("MATRIX TOO SMALL: row and col sizes should be larger than 3")

    Mij = uniform(0, 1, size=(int(rw * b), int(cl * b)))
    cy, cx = mod_param(int(rw * b), int(cl * b), b)
    M_no = zeros(Mij.shape)
    le = []
    Pi = ((b - 1) * mu) / b
    lb = 0

    # for each block generate a nested structure
    for ii in range(int(b)):
        j, i = indices(M_no[cy * ii : cy * (ii + 1), cx * ii : cx * (ii + 1)].shape)

        # heaviside function to produce the nested structure
        H = ((j[::-1, :] + 1) / cy) > ballcurve((i / cx), xi)

        M_no[cy * ii : cy * (ii + 1), cx * ii : cx * (ii + 1)] = H
        le += [M_no[cy * ii : cy * (ii + 1), cx * ii : cx * (ii + 1)].sum()]
        lb += cy * cx

    # generate the nested structure of the remaining block
    j, i = indices(M_no[(ii + 1) * cy :, (ii + 1) * cx :].shape)

    # heaviside function to produce the nested structure
    H = ((j[::-1, :] + 1) / j.shape[0]) > ballcurve((i / i.shape[1]), xi)

    M_no[(ii + 1) * cy :, (ii + 1) * cx :] = H
    le += [M_no[(ii + 1) * cy :, (ii + 1) * cx :].sum()]
    lb += (int(rw * b) - ((ii + 1) * cy)) * (int(cl * b) - ((ii + 1) * cx))
    Et = M_no.sum(dtype=int)

    # prob of having a link outside blocks
    p_inter = (mu * Et) / (lb * b) if ((lb * b) != 0) else 0
    M_no[M_no == 0] = p_inter
    for ix in range(int(b)):
        j, i = indices(M_no[cy * ix : cy * (ix + 1), cx * ix : cx * (ix + 1)].shape)
        Pr = (
            (P * le[ix]) / ((cx * cy) - le[ix] + (P * le[ix]))
            if ((cx * cy) - le[ix] + (P * le[ix])) != 0
            else 0
        )
        # heaviside function to produce the nested structure
        H = ((j[::-1, :] + 1) / cy) > ballcurve((i / cx), xi)

        # prob of having a link within blocks
        p_intra = ((1 - P + (P * Pr)) * H + Pr * (1 - H)) * (1 - Pi)
        M_no[cy * ix : cy * (ix + 1), cx * ix : cx * (ix + 1)] = p_intra

    # calculate to the remaining block
    j, i = indices(M_no[(ix + 1) * cy :, (ix + 1) * cx :].shape)
    Pr = (
        (P * le[ix + 1])
        / (
            ((int(rw * b) - (ix + 1) * cy) * (int(cl * b) - (ix + 1) * cx))
            - le[ix + 1]
            + (P * le[ix + 1])
        )
        if (le[ix + 1] > 0) & (P != 0)
        else 0
    )

    # heaviside function to produce the nested structure
    H = ((j[::-1, :] + 1) / j.shape[0]) > ballcurve((i / i.shape[1]), xi)
    # prob of having a link within blocks
    p_intra = ((1 - P + (P * Pr)) * H + Pr * (1 - H)) * (1 - Pi)
    M_no[(ix + 1) * cy :, (ix + 1) * cx :] = p_intra
    M = (M_no > Mij).astype(int)
    if not bipartite:
        fill_diagonal(M, 0)
        M = triu(M, k=1) + (triu(M, k=1)).T

    return M
