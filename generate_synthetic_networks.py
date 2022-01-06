"""
Created on Mon Apr 16 11:07:47 2018

@author: mariapalazzi
"""
from numpy import savetxt
from netgen import network_generator
from os.path import isfile
import sys


def exploring_transitions(
    bipartite: bool,
    rw: int,
    cl: int,
    b: float,
    xi: float,
    P: float,
    mu: float,
) -> None:
    """
    function to generate synthetic unipartite networks with different planted partitions employing
    the benchmark model introduced by ASR et al, PRE 2018, and performs structural analysis by means
    of calculation of nestedness, modularity and in-block nestedness.

    The function generate a file containing the generated network in matrix format

    Inputs:
    ----------
       bipartite:
           a boolean to indicate if you want to generate bipartite (True) or unipartite (False) networks
       rw:
           number of row nodes per block
       cl: optional
           number of colum nodes per block
       b:
           number of blocks
       xi:
           shape parameter
       P:
          parameter that determines the amount of links outside the perfect nested structure
       mu:
           parameter that determines the amount of inter blocks links
    """
    xi = round(xi, 2)
    P = round(P, 2)
    mu = round(mu, 2)

    if bipartite == False:
        name = f"edges_synthetic_unipartite_B{b}_xi{xi}_P{P}_mu{mu}.csv"
    else:
        name = f"edges_synthetic_bipartite_B{b}_xi{xi}_P{P}_mu{mu}.csv"
    print(name.replace("_", " ")[:-4])

    if not isfile(f"./{name}"):
        # generating the synthetic networks (fill and  triu for uni(bi)partite)
        print("Creating the synthetic network for the parameters given")
        M_ = network_generator(rw, cl, b, xi, P, mu, bipartite)
        savetxt(name, M_.astype(int), fmt="%d", delimiter=",")
    else:
        print("Nothing to do:")
        print("\tA file with the same parameters already exists")


if __name__ == "__main__":
    shift = 0
    bipartite = False
    if len(sys.argv) == 7:
        bipartite = True
        shift = 1
    parameters = dict(
        bipartite=bipartite,
        rw=int(sys.argv[1]),
        cl=int(sys.argv[1 + shift]),
        b=float(sys.argv[2 + shift]),
        xi=float(sys.argv[3 + shift]),
        P=float(sys.argv[4 + shift]),
        mu=float(sys.argv[5 + shift]),
    )
    exploring_transitions(**parameters)
