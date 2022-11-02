import numpy as np
from numpy.random import Generator, PCG64
from numba import njit
from enum import Enum

seed = 1013
rs = Generator(PCG64(seed))


class CellType(Enum):
    BLACK = 0
    GREEN = 1
    ORANGE = 2


class ForestFire():

    def __init__(self, shape: tuple, n_trees: int, n_fire: int) -> None:

        self.m, self.n = shape
        self.grid = ForestFire.init_grid(self.m, self.n, n_trees, n_fire)
        self._neighborhood_fun = None
        self._n_fire = []
        self._n_trees = []
        self._n_empty = []
        self._frames = []

    @staticmethod
    @njit()
    def init_grid(m: int, n: int, n_trees: int, n_fire: int):
        ""
        grid = np.full(m*n, CellType.BLACK.value, dtype=np.int8)
        index_trees = np.random.choice(grid.shape[0], size=n_trees, replace=False)
        grid[index_trees] = CellType.GREEN.value
        index_fire = np.random.choice(grid.shape[0], size=n_fire, replace=False)
        grid[index_fire] = CellType.ORANGE.value
        return grid.reshape((m, n))

    def simulate(self, p_g, p_f, time):
        if self._neighborhood_fun is None:
            raise ValueError('neighborhood rule is not define')

        for _ in range(time):
            self._frames.append(self.grid.copy())
            self._n_fire.append(np.sum(self.grid==CellType.ORANGE.value)) 
            self._n_trees.append(np.sum(self.grid==CellType.GREEN.value))
            self._n_empty.append(np.sum(self.grid==CellType.BLACK.value))
            self.grid = ForestFire.update(self.grid, p_g, p_f, self._neighborhood_fun)

    def set_neighborhood_rule(self, neighborhood_fun):
        self._neighborhood_fun = neighborhood_fun

    @property
    def N_fire(self):
        return np.array(self._n_fire)

    @property
    def N_trees(self):
        return np.array(self._n_trees)
    
    @property
    def N_empty(self):
        return np.array(self._n_empty)
    
    @property
    def frames(self):
        return np.array(self._frames)

    @staticmethod
    @njit()
    def update(grid, p_g, p_f, neighborhood_fun):
    
        grid_previous = grid.copy()
        
        for i, row in enumerate(grid_previous):
            for j, cell in enumerate(row):
                if cell == 1:
                    grid[i,j] = neighborhood_fun((i,j), grid_previous)
                    
        for i, row in enumerate(grid_previous):
            for j, cell in enumerate(row):
                if cell == 2:
                    grid[i,j] = CellType.BLACK.value 
        
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == 0:
                    if np.random.rand() <= p_g:
                        grid[i,j] = CellType.GREEN.value
                if cell == 1:
                    if np.random.rand() <= p_f:
                        grid[i,j] = CellType.ORANGE.value
        
        return grid

    @staticmethod
    @njit()
    def neighborhood_plus(cell, grid):
        """Обновить отдельную ячейку `cell` КА `grid`."""
        i, j = cell
        
        if j-1 < 0:
            n_0 = 0
        else:
            n_0 = grid[i][j-1]
            
        if i-1 < 0:
            n_1 = 0
        else:
            n_1 = grid[i-1][j]
            
        if i+1 >= grid.shape[0]:
            n_3 = 0
        else:
            n_3 = grid[i+1][j]
            
        if j+1 >= grid.shape[1]:
            n_2 = 0
        else:
            n_2 = grid[i][j+1]
        
        if CellType.ORANGE.value in [n_0, n_1, n_2, n_3]:
            return CellType.ORANGE.value
        return CellType.GREEN.value

    @staticmethod
    @njit
    def neighborhood_round(cell, grid):

        i, j = cell
        border = -1

        index_i = [i-1, i, i+1, i-1, i+1, i-1, i, i+1]
        for ind, k in enumerate(index_i):
            if k < 0 or k >= grid.shape[0]:
                index_i[ind] = border
        index_j = [j-1, j-1, j-1, j, j, j+1, j+1, j+1]
        for ind, k in enumerate(index_j):
            if k < 0 or k >= grid.shape[1]:
                index_j[ind] = border

        for cell_neigh in zip(index_i, index_j):
            if border in cell_neigh:
                continue
            if CellType.ORANGE.value == grid[cell_neigh]:
                return CellType.ORANGE.value

        return CellType.GREEN.value