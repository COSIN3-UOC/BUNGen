#  Generate synthetic networks with Nested, Modular and In-block nested structures

Generate synthetic networks with predetermined nested, modular or in-block nested structure, with different levels of noise. This code employs a modification of the benchmark model introduced by Solé-Ribalta et al, PRE 2018 (https://doi.org/10.1103/PhysRevE.97.062302), in which one can networks with a fixed block size and increasing number of blocks, instead of networks with fixed size. 

The parameters must be passed respecting the following order.
    
        
## Inputs:
       
1) rw  = int: number of row nodes that form a block
2) cl  = int: number of col nodes that form a block
3) B   = float >= 1: number of blocks on which the main matrix will be divided
4) xi  = float >=1: shape parameter to indicate how stylized is the nested curve
5) P   = in [0, 1] value of the parameter that control the amount of noise outside a perfect nested 
6) mu  = in [0, 1] value of the parameter that control the amount of noise outside the blocks
## output:
1) The function return each generated network in matrix format in a .csv file with the word prefix "edges".
	
If the number of columns nodes is not given, the function will assume that we want to generate a unipartite networks.

### example: 
```
python3 generate_synthetic_networks.py 20 30 2 2.2 0.1 0.1

```
### example 2: 

If you want to generate several networks with varying values of a parameter. To generate networks for different parameters reproduce the following example in a nested loop
```
for B in $(seq 1 0.5 5); # B parameter from 1 to 5 with 0.5 increments
do
   python generate_synthetic_networks.py 20 30 $B 2 0.1 0.1 &
done
```
# Citations

A. Solé-Ribalta, CJ. Tessone, M S. Mariani, and J Borge-Holthoefer. Revealing in-block nestedness: Detection and benchmarking, Phys. Rev. E 97, 062302 (2018). DOI: [10.1103/PhysRevE.97.062302](https://doi.org/10.1103/PhysRevE.97.062302)

MJ Palazzi, J Borge-Holthoefer, CJ Tessone and A Solé-Ribalta. Macro- and mesoscale pattern interdependencies in complex networks. J. R. Soc. Interface, 16, 159, 20190553 (2019). DOI: [10.1098/rsif.2019.0553](https://doi.org/10.1098/rsif.2019.0553)
