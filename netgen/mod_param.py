from typing import List


def mod_param(rows: int, cols: int, B: int) -> List[int]:
    """
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
    """
    N = rows + cols  # size matrix
    Nb = int(N // B)  # size of blocks
    pr = 0.5 if (rows == cols) else (rows / N)
    cx = int(Nb * (1 - pr))
    cy = int(Nb * pr)
    return [cy, cx]
