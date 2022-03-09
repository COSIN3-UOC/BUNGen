#  Generate synthetic networks with Nested, Modular and In-block nested structures

Generate synthetic networks with predetermined nested, modular or in-block nested structure, with different levels of noise. This code employs a modification of the benchmark model introduced by Solé-Ribalta et al, PRE 2018 (https://doi.org/10.1103/PhysRevE.97.062302), in which one can produce networks with different block sizes (following a powerlaw), instead of networks with equally sized blocks. 

The parameters of the model are the following.
        
## Inputs:
       
1) rw  = int: number of row nodes that form a block
2) cl  = int: number of col nodes that form a block
3) B   = int >= 1: number of blocks on which the main matrix will be divided
4) P   = in [0, 1] value of the parameter that control the amount of noise outside a perfect nested 
5) mu  = in [0, 1] value of the parameter that control the amount of noise outside the blocks
6) alpha = float: bounded in (x1,x2) be the scaling parameter of the distribution, default 2.5. 
	For networks with equal block size set alpha = 0.
7) bipartite = bool: True for bipartite networks, false for unipartite. If not given it will generate a unipartite network.
8) min_block_size = int: minimum block size, default 10% of rw and cl, respectively.
9) fixedConn = bool: True if you want to produce with fixed connetance and estimate a xi value. False if you want to set an specific xi value.
10) link_density = float: If fixedConn = True, this parameter specifies the desired connectance [0,1]. If fixedConn = False it specifies xi >= 1.

## Output:
An array that corresponds to the binary synthetic adjacency matrix (biadjacency for bipartite cases), and/or an array with the matrix of link probabilities.

## System Requirements 	
	
Python 3.x.
numpy>=1.20.0
scipy=*

## Use examples: 
### To use as a library

### From the commmand line
This script will save the output matrices in csv files
``` sh
python generate_synthetic_networks.py 100 200 

```
For help type:
``` sh
python generate_synthetic_networks.py -h

```

# Citations
MJ Palazzi, A Lampo, A Solé-Ribalta, and J Borge-Holthoefer. To fill in

A. Solé-Ribalta, CJ. Tessone, M S. Mariani, and J Borge-Holthoefer. Revealing in-block nestedness: Detection and benchmarking, Phys. Rev. E 97, 062302 (2018). DOI: [10.1103/PhysRevE.97.062302](https://doi.org/10.1103/PhysRevE.97.062302)

MJ Palazzi, J Borge-Holthoefer, CJ Tessone and A Solé-Ribalta. Macro- and mesoscale pattern interdependencies in complex networks. J. R. Soc. Interface, 16, 159, 20190553 (2019). DOI: [10.1098/rsif.2019.0553](https://doi.org/10.1098/rsif.2019.0553)
