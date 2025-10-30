"""
Created on Thu Sep 14 13:11:07 2017

@author: maria
"""
from typing import List
from numpy import cumsum, indices, zeros
from numpy.typing import ArrayLike
from .utils import ballcurve


def generate_probability_matrix(
    rw: int,
    cl: int,
    b: int,
    cy: List[int],
    cx: List[int],
    xi: float,
    p: list,
    mu: float,
) -> ArrayLike:
    """
    function to generate a probability matrix of links. To produces synthetic networks with nested,
    modular and in-block nested structures. Generates networks with a fixed size and
    blocks with heterogenous sizes.
    This benchmark is a modification of the one introduced by ASR et al (PRE 2018).
    The parameters must be passed respecting the following order.
    inputs:
    ----------
    rw: int >1
        number of row nodes
    cl: int >1
        number of col nodes
    B: number >=1
        number of blocks on which the main matrix will be divided
    cy: List[int]
        the size of the block in the row dimension
    cx: List[int]
        the size of the block in the column dimension
    xi: number, >=1
        shape parameter to indicate how stylised is the nested curve
    p: number in [0, 1]
        paramteter that control the amount of noise outside a perfectly nested structure
    mu: number in [0, 1]
        parameteter that control the amount of noise outside the blocks
    output:
    ----------
    Mij: array
        The synthetic network matrix of link probabilities with the define parameters
    example:
    ---------
    Mij=network_generator(rw,cl,B,cy, cx, xi,p,mu)
    """

    if rw < 3 or cl < 3:
        raise ValueError("MATRIX TOO SMALL: row and col sizes should be larger than 3")

    cscx = cumsum(cx).tolist()
    cscy = cumsum(cy).tolist()
    cscy.insert(0, 0)
    cscx.insert(0, 0)
    M_no = zeros((int(rw), int(cl)))
    le = []
    lb = []

    # for each block generate a nested structure
    for ii in range(int(b)):
        j, i = indices((cy[ii], cx[ii]))

        # heaviside function to produce the nested structure
        H = ((j[::-1, :] ) / cy[ii]) >= ballcurve((i / cx[ii]), xi)
        
        # Ensure the block-local first row/col are in the noiseless nested mask
        H[0, :] = True
        H[:, 0] = True

        M_no[cscy[ii] : cscy[ii + 1], cscx[ii] : cscx[ii + 1]] = H
        le += [M_no[cscy[ii] : cscy[ii + 1], cscx[ii] : cscx[ii + 1]].sum()]
        lb += [cy[ii] * cx[ii]]

    

    cslb = sum(lb)
    Et=M_no.sum(dtype=int)
    p_i = mu*(1 - cslb/(rw*cl))
    p_inter = (mu*Et)/(rw*cl)
    for ix in range(int(b)):
        # prob of having a link outside blocks
        M_no[cscy[ix] : cscy[ix + 1], :] = p_inter
        M_no[:, cscx[ix] : cscx[ix + 1]] = p_inter
        j, i = indices((cy[ix], cx[ix]))
        
        Pr = (
            (p[ix] * le[ix]) / (lb[ix] - le[ix] + (p[ix] * le[ix]))
            if (lb[ix] - le[ix] + (p[ix] * le[ix])) != 0
            else 0
        )
        # heaviside function to produce the nested structure
        H = ((j[::-1, :]) / cy[ix]) >= ballcurve((i / cx[ix]), xi)
        
        # Ensure the block-local first row/col are in the noiseless nested mask
        H[0, :] = True
        H[:, 0] = True

        # prob of having a link within blocks
        p_intra = ((1 - p[ix] + (p[ix] * Pr)) * H + Pr * (1 - H)) * (1 - p_i)
        M_no[cscy[ix] : cscy[ix + 1], cscx[ix] : cscx[ix + 1]] = p_intra

    return M_no
