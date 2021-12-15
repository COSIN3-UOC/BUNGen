from numpy.typing import ArrayLike


def ballcurve(x: ArrayLike, xi: float) -> ArrayLike:
    """
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
    """
    return 1 - (1 - (x) ** (1 / xi)) ** xi
