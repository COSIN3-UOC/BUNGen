#  Generate synthetic networks with Nested, Modular and In-block nested structures

Generate synthetic networks with predetermined nested, modular or in-block nested structure, with different levels of noise. This code employs a modification of the benchmark model introduced by Solé-Ribalta et al, PRE 2018 (https://doi.org/10.1103/PhysRevE.97.062302), in which one can produce networks with different block sizes (following a powerlaw), instead of networks with equally sized blocks. 

The parameters of the model are the following.
        
## Inputs:
positional arguments:
1) rows  = int: number of row nodes.
2) cols  = int: number of column nodes.
3) block_number   = int >= 1: number of prescribed blocks (i.e. modules) in the network.
4) P   = in [0, 1] parameter that controls the amount of noise outside a perfectly nested structure.
5) mu  = in [0, 1] parameter that controls the amount of inter-block (i.e. between modules) noise.
6) alpha = float: bounded in (x1,x2), alpha is the scaling parameter of the distribution of block sizes. Default is alpha = 2.5. 
	For networks with equally-sized blocks set alpha = 0.
7) bipartite = bool: True for bipartite networks, false for unipartite. If not given, it will generate a unipartite network.
8) min_block_size = int: minimum block size, default 10% of rows and cols, respectively.
9) fixedConn = bool: True if you want to produce a network with prescribed connetance. False if you want to set a specific xi value.
10) link_density = float: If fixedConn = True, this parameter specifies the desired connectance [0,1]. If fixedConn = False, it specifies xi >= 0.

## Output:
A numpy matrix that corresponds to the binary synthetic adjacency matrix (biadjacency for bipartite cases), and/or a numpy matrix with link probabilities.

## System Requirements 	
	
- Python 3.x.
- numpy >= 1.20.0
- scipy = *

## Use examples: 
### To use as a library
To produce a single network with desired parameters within a custom made script. User can proceed in the following way.
```python
from netgen import NetworkGenerator

M,_ = NetworkGenerator.generate(500, 500, 4, bipartite=True, P=0.5, mu=0.5, 
	alpha=2.5, min_block_size=0, fixedConn=False, link_density=2.45)

```
Keep in mind that the parameters are positional. If user does not pass the parameters as named arguments, then order must be respected. If the user wants the function to return the matrix of link probabilities edit the line above by replacing M,_ with M,Pij or  _ ,Pij. 

To produce several networks simultaneously while varying some parameter and keeping others fixed:
```python
from netgen import NetworkGenerator

gen =  NetworkGenerator(500, 500, 4, bipartite=True, P=0.5,mu=0.5, alpha=2.5, 
	min_block_size=0, fixedConn=False, link_density=2.45)

for p in np.arange(0,1,0.2):
     M,_ = gen(P=p)
     # do something with each M (plot, save, append, etc)

```

### From the commmand line
This script will save the output matrices in csv files. Produce a network with certain number of rows, cols, blocks, xi and noise. (fixedConn false by default).
``` sh
python generate_synthetic_networks.py 100 200 2 0.1 0.1 2.05

```
If you want to produce a network with a given connectance replace fixedConn to true and change xi value for the desired connectance value by typing:
``` sh
python generate_synthetic_networks.py 100 200 2 0.1 0.1 .005 -f

```
To modify the remaining parameters just add -a value if you want to modify the alpha default value from the powerlaw
``` sh
python generate_synthetic_networks.py 100 200 2 0.1 0.1 .005 -f -a 2.1

```

For help and description of the parameters type:
``` sh
python generate_synthetic_networks.py -h

```

# Citations
MJ Palazzi, A Lampo, A Solé-Ribalta, and J Borge-Holthoefer. To fill in

A. Solé-Ribalta, CJ. Tessone, M S. Mariani, and J Borge-Holthoefer. Revealing in-block nestedness: Detection and benchmarking, Phys. Rev. E 97, 062302 (2018). DOI: [10.1103/PhysRevE.97.062302](https://doi.org/10.1103/PhysRevE.97.062302)

MJ Palazzi, J Borge-Holthoefer, CJ Tessone and A Solé-Ribalta. Macro- and mesoscale pattern interdependencies in complex networks. J. R. Soc. Interface, 16, 159, 20190553 (2019). DOI: [10.1098/rsif.2019.0553](https://doi.org/10.1098/rsif.2019.0553)
