#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:41:27 2022

@author: mariapalazzi
"""
from dataclasses import dataclass
from typing import List
from numpy.typing import ArrayLike
from numpy import array, savetxt, fill_diagonal, triu
from netgen import network_generator
from netgen.heterogenousBlockSizes import heterogenousBlockSizes
from netgen.xiFunConn import xiFunConn
from os.path import isfile
from numpy.random import uniform
import argparse


@dataclass
class NetworkGenerator:
    rows: int
    columns: int
    block_number: int
    P: float
    mu: float
    alpha: float
    min_block_size: int
    # net_type: str

    def __post_init__(self):
        self.P = round(self.P, 2)
        self.mu = round(self.mu, 2)
        self.columns = self.rows if self.columns is None else self.columns
        self.alpha = 2.5 if self.alpha is None else self.alpha
        self.min_block_size = 0 if self.min_block_size is None else self.min_block_size
        self.net_type = "unipartite" if self.rows == self.columns else "bipartite"
        self.get_block_sizes()

    def get_block_sizes(self) -> tuple[List[int], List[int]]:
        self.cy = heterogenousBlockSizes(
            self.block_number, self.rows, alpha=self.alpha, min_block_size=self.min_block_size
        )
        if self.rows == self.columns:
            self.cx = self.cy
        else:
            self.cx = heterogenousBlockSizes(
                self.block_number, self.columns, alpha=self.alpha, min_block_size=self.min_block_size
            )
        return self.cx, self.cy

    def generate_synthetic_network(self, xi: float) -> tuple[ArrayLike, ArrayLike]:
        Mij = network_generator(self.rows, self.columns, self.block_number, self.cy, self.cx, xi, self.P, self.mu)
        Mrand = array(uniform(0,1,size=(self.rows, self.columns)))
        M = (Mij > Mrand).astype(int)
        if self.rows == self.columns:
            fill_diagonal(M, 0)
            M = triu(M, k=1) + (triu(M, k=1)).T
        return Mij, M

    def safe_to_file(self, returnPij, xi, name):
        # generating the synthetic networks (fill and  triu for unipartite)
        if isfile(f"./{name}"):
            print("Nothing to do:")
            print("\tA file with the same parameters already exists")
        else:
            print("Creating the synthetic network for the parameters given")
            Mij, M = self.generate_synthetic_network(xi)
            savetxt(name, M, fmt="%d", delimiter=",")
            if returnPij == True:
                Pijname = f"pij_bipartite_B{self.block_number}_xi{xi}_P{self.P}_mu{self.mu}.csv"
                savetxt(Pijname, Mij, fmt="%4f", delimiter=",")

    def __call__(self, fixedConn: bool, returnPij: bool, shape: float) -> None:
        if fixedConn == True:
            xi = xiFunConn(self.cx, self.cy, self.rows, self.columns, shape)
            print(f"xi value for desired connectance {xi}")  # to verify
        else:
            xi = round(shape, 2)
        name = f"matrix_synthetic_{self.net_type}_B{self.block_number}_xi{xi}_P{self.P}_mu{self.mu}.csv"
        print(name.replace("_", " ")[:-4])

        self.safe_to_file(returnPij, xi, name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="NetworkGenerator",
        description="""Generator of synthetic unipartite (or bipartite) networks with different planted partitions
    employing a variation of the benchmark model introduced by ASR et al, PRE 2018.\n
    The function generate a file containing the generated network in matrix format""",
    )

    parser.add_argument("rows", type=int, help="number of row nodes")
    parser.add_argument("columns", type=int, help="number of column nodes")
    parser.add_argument("block_number", type=int, help="B")
    parser.add_argument("P", type=float, help="P")
    parser.add_argument("mu", type=float, help="mu")
    parser.add_argument("shape", type=float, help="shape")
    parser.add_argument("-a", "--alpha", type=float, help="alpha")
    parser.add_argument("-b", "--min_block_size", type=int, help="min block size")
    parser.add_argument("-f", "--fixedConn", action="store_true", help="fixed Conn")
    parser.add_argument("-p", "--returnPij", action="store_true", help="return Pij")

    # args = parser.parse_args("200 100 2 0 0 1.5 -a 0".split())
    args = parser.parse_args()

    generator = NetworkGenerator(
        rows=args.rows,
        columns=args.columns,
        block_number=args.block_number,
        P=args.P,
        mu=args.mu,
        alpha=args.alpha,
        min_block_size=args.min_block_size,
    )
    generator(fixedConn=args.fixedConn, returnPij=args.returnPij, shape=args.shape)
