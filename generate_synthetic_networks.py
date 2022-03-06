#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:41:27 2022

@author: mariapalazzi
"""
from genericpath import isfile

from numpy import savetxt
from netgen import NetworkGenerator
import argparse


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
    parser.add_argument("link_density", type=float, help="link_density")
    parser.add_argument("-a", "--alpha", type=float, default=2.5, help="alpha")
    parser.add_argument("-b", "--min_block_size", type=int, default=0, help="min block size")
    parser.add_argument("-u", "--unipartite", action="store_false", help="unipartite")
    parser.add_argument("-f", "--fixedConn", action="store_true", help="fixed Conn")

    args = parser.parse_args()

    gen = NetworkGenerator(
            rows=args.rows,
            columns=args.columns,
            block_number=args.block_number,
            bipartite=args.unipartite,
            P=args.P,
            mu=args.mu,
            alpha=args.alpha,
            min_block_size=args.min_block_size,
            fixedConn=args.fixedConn,
            link_density=args.link_density,
        )
    
    name = f"matrix_synthetic_{gen.net_type}_B{args.block_number}_xi{gen.xi}_P{args.P}_mu{args.mu}.csv"
    if isfile(f"./{name}"):
        print("Nothing to do:")
        print("\tA file with the same parameters already exists")
    else:
        print("Creating the synthetic network for the parameters given")
        M, _ = gen()
        # M,_ = NetworkGenerator.generate(
        #     rows=args.rows,
        #     columns=args.columns,
        #     block_number=args.block_number,
        #     bipartite=args.unipartite,
        #     P=args.P,
        #     mu=args.mu,
        #     alpha=args.alpha,
        #     min_block_size=args.min_block_size,
        #     fixedConn=args.fixedConn,
        #     link_density=args.link_density,
        # )
        savetxt(name, M, fmt="%d", delimiter=",")
