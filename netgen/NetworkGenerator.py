from dataclasses import dataclass
from typing import List
from numpy.typing import ArrayLike
from numpy import array, fill_diagonal, triu, repeat
from .network_generator import network_generator
from .heterogenousBlockSizes import heterogenousBlockSizes
from .xiFunConn import xiFunConn
from numpy.random import uniform
import warnings


@dataclass
class NetworkGenerator:
    rows: int
    columns: int
    block_number: int
    P: float
    mu: float
    gamma: float
    bipartite: bool
    min_block_size: int
    fixedConn: bool
    link_density: float

    def get_block_sizes(self) -> tuple[List[int], List[int]]:
        self.cy = heterogenousBlockSizes(
            self.block_number, self.rows, gamma=self.gamma, min_block_size=self.min_block_size
        )
        if self.rows == self.columns:
            self.cx = self.cy
        else:
            self.cx = heterogenousBlockSizes(
                self.block_number, self.columns, gamma=self.gamma, min_block_size=self.min_block_size
            )
        return self.cx, self.cy

    def synthetic_network(self) -> tuple[ArrayLike, ArrayLike,List[int],List[int]]:
        Mij = network_generator(self.rows, self.columns, self.block_number, self.cy, self.cx, self.xi, self.P, self.mu)
        Mrand = array(uniform(0, 1, size=(self.rows, self.columns)))
        labelRows = repeat(range(len(self.cy)),self.cy).tolist()
        labelCols = repeat(range(len(self.cx)),self.cx).tolist()
        M = (Mij > Mrand).astype(int)
        if not self.bipartite:
            fill_diagonal(M, 0)
            M = triu(M, k=1) + (triu(M, k=1)).T
        return M, Mij, labelRows, labelCols

    @property
    def xi(self) -> float:
        if self.fixedConn == True:
            maxConn = sum([(x*y) for x,y in zip(self.cx,self.cy)])/(self.rows*self.columns)
            xi = xiFunConn(self.cy, self.cx, self.rows, self.columns, self.link_density)
            if maxConn < self.link_density:
                raise ValueError(f"Desired connectance not possible for parameters combination. Max connectance {maxConn:.3f}")
            else:
                print(f"xi value for desired connectance {xi:.2f}") 
        else:
            xi = round(self.link_density, 2)
        return xi
    
    @property
    def net_type(self) -> str:
        return "bipartite" if self.bipartite else "unipartite"
    
    def __post_init__(self) -> None:
        self.get_block_sizes()
        if self.fixedConn and self.link_density>1:
            raise ValueError("If parameter 'fixedConn' is True, then 'link_density' cannot be greater than 1")
    
    def __call__(self, **kwargs) -> tuple[ArrayLike, ArrayLike, List[int], List[int]]:
        for param in self.__annotations__.keys():
            if param in kwargs:
                setattr(self, param, kwargs[param])
        
        self.P = round(self.P, 2)
        self.mu = round(self.mu, 2)

        if not self.bipartite and self.columns != self.rows:
            raise ValueError("For unipartite configuration, the number of columns and rows must be the same.")

        return self.synthetic_network()

    @classmethod
    def generate(
        cls,
        rows: int,
        columns: int,
        block_number: int,
        P: float,
        mu: float,
        gamma: float,
        bipartite: bool,
        min_block_size: int,
        fixedConn: bool,
        link_density: float,
    ):
        return cls(
            rows=rows,
            columns=columns,
            block_number=block_number,
            P=P,
            mu=mu,
            gamma=gamma,
            bipartite=bipartite,
            min_block_size=min_block_size,
            fixedConn=fixedConn,
            link_density=link_density,
        )()
