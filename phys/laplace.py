# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:44:43 2023

@author: reonid
"""
#%%
from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from datetime import datetime
import time

from ..misc.cienptas import indexing, StopWatch, save_array
from ..misc.grid import (Grid, _grid_to_raw_points, scalar_field_on_grid, 
                         _xx_yy_etc_from_grid)
from ..misc.interp import FastGridInterpolator
from ..geom.group import Group3D
from ..geom.groupstrat import JustCallStrategy

# %%

_1_ = indexing[1:-1]

kernel1D = np.array([1.0,  0.0,  1.0]) # just for generality

kernel2D = np.array([[0.0,  1.0,  0.0],
                     [ 1.0,  0.0,  1.0],
                     [ 0.0,  1.0,  0.0]], dtype=np.float64)

kernel3D = np.zeros((3, 3, 3), dtype=np.float64)
kernel3D[-1, 1, 1] = 1.0; kernel3D[0, 1, 1] = 1.0
kernel3D[ 1,-1, 1] = 1.0; kernel3D[1, 0, 1] = 1.0
kernel3D[ 1, 1,-1] = 1.0; kernel3D[1, 1, 0] = 1.0 

def mid_indexing(U): 
    if len(U.shape) == 1: 
        return U.shape[0]//2
    elif len(U.shape) == 2: 
        return indexing[U.shape[0]//2, U.shape[1]//2]
    elif len(U.shape) == 3: 
        return indexing[U.shape[0]//2, U.shape[1]//2, U.shape[2]//2]

def get_kernel(U): 
    if len(U.shape) == 1: 
        return kernel1D
    elif len(U.shape) == 2: 
        return kernel2D
    elif len(U.shape) == 3: 
        return kernel3D

#%%
class AbstractLaplaceBoundaryConditions(ABC):
    @abstractmethod
    def reset(self, U):
        pass
    
    @abstractmethod
    def set_grid(self, grid):
        pass

class EmptyBoundaryConditions(AbstractLaplaceBoundaryConditions):
    def __init__(self, *args, **kwargs):
        pass
    
    def reset(self, U):
        pass
        # return U
    
    def set_grid(self, grid):
        pass
    
class VolumeBoundaryConditions2D(AbstractLaplaceBoundaryConditions):
    def __init__(self, U_interp):
        self.interp = U_interp
        self.U = None
        
    def reset(self, U):
        U[0, :]  = self.U[ 0, :]
        U[-1, :] = self.U[-1, :]
        U[:, 0]  = self.U[:,  0]
        U[:, -1] = self.U[:, -1]
        # return U
        
    def set_grid(self, grid):
        self.U = scalar_field_on_grid(grid, self.interp)
        
    @classmethod
    def from_gabarits(cls, gabarits, grid_step, U):
        xx = np.arange(gabarits[0, 0], gabarits[1, 0] + 0.001*grid_step, grid_step)
        yy = np.arange(gabarits[0, 1], gabarits[1, 1] + 0.001*grid_step, grid_step)
        U_interp = FastGridInterpolator([xx, yy], U, fill_value=0.0)
        return cls(U_interp)

class VolumeBoundaryConditions2DAnyIndexing(VolumeBoundaryConditions2D):
    def __init__(self, U_interp, indexing):
        self.interp = U_interp
        self.indexing = indexing
        self.U = None
        
    def reset(self, U):
        U[self.indexing] = self.U[self.indexing]
        
    @classmethod
    def from_gabarits(cls, gabarits, grid_step, U, indexing):
        xx = np.arange(gabarits[0, 0], gabarits[1, 0] + 0.001*grid_step, grid_step)
        yy = np.arange(gabarits[0, 1], gabarits[1, 1] + 0.001*grid_step, grid_step)
        U_interp = FastGridInterpolator([xx, yy], U, fill_value=0.0)
        return cls(U_interp, indexing)

class MasksBoundaryConditions(AbstractLaplaceBoundaryConditions): 
    def __init__(self, masks_voltages_func):
        self.masks_voltages_func = masks_voltages_func
        self.masks_voltages = None
        
    def reset(self, U): 
        for mask, voltage in self.masks_voltages: 
            U[mask] = voltage     # include boundary conditions U[edge_flag] = 0.0
            # !!! here can use indexing as mask

        # Neumann boundary condition on left and right sides
        # ??? is it equivalent to 'nearest' ??? yes
        # U[   0,  _1_,  _1_ ] =  U[   1,  _1_,  _1_ ]  
        # U[  -1,  _1_,  _1_ ] =  U[  -2,  _1_,  _1_ ]

        # return U # !!!
        
    def set_grid(self, grid):
        self.masks_voltages = self.masks_voltages_func(grid)

#%%
class AbstractLaplaceStopCondition(ABC):
    @abstractmethod
    def __call__(self, U, _U, i):
        pass
    
    def set_grid(self, grid):
        self.log.append([[], []])
        self.grid = grid
        
    def dump(self):
        # t = int(datetime.now().strftime('%H'))
        t = time.time()/3600.
        if t - self.save_time > self.dt:
            save_array(self.dump_name + '_U.npy', self.U)
            save_array(self.dump_name + '_grid.npy', self.grid)
            self.save_time = t

class MidIndexingStopCondition(AbstractLaplaceStopCondition):
    def __init__(self, eps, dump_name, t_between_dumps=4):
        self.eps = eps
        self.log = [[[], []]]
        self.U = None
        self.grid = None
        self.save_time = 0
        self.dump_name = dump_name
        self.dt = t_between_dumps
        
    def __call__(self, U, _U, i):
        idx = mid_indexing(U)
        delta = np.max(np.abs(U[idx] - _U[idx]))
        print(f'\rSolving laplace. Discrepancy = {delta:.3}              ', end='', flush=True)
        self.log[-1][0].append(i)
        self.log[-1][1].append(delta)
        self.U = U
        self.dump()
        # print(delta, i)
        return (delta < self.eps)
    
class MaxStopCondition(AbstractLaplaceStopCondition):
    def __init__(self, eps, dump_name, t_between_dumps=4):
        self.eps = eps
        self.log = [[[], []]]
        self.U = None
        self.grid = None
        self.save_time = 0
        self.dump_name = dump_name
        self.dt = t_between_dumps
        
    def __call__(self, U, _U, i):
        delta = np.max(np.abs(U - _U))
        print(f'\rSolving laplace. Discrepancy = {delta:.3}              ', end='', flush=True)
        self.log[-1][0].append(i)
        self.log[-1][1].append(delta)
        self.U = U
        self.dump()
        # idx = np.argmax(np.abs(U - _U))
        # print(delta, i, idx, U.ravel()[idx])
        return (delta < self.eps)
    
class TimeMaxDivStopCondition(AbstractLaplaceStopCondition):
    def __init__(self, epsilons, max_steps, max_div, dump_name, t_between_dumps=4):
        self.max_steps = max_steps
        self.max_div = max_div
        self.epseps = epsilons
        self.counter = 0
        self.eps = self.epseps[0]
        self.log = [[[], []]]
        self.U = None 
        self.exitlog = None
        self.grid = None
        self.save_time = 0
        self.dump_name = dump_name
        self.dt = t_between_dumps
         
    def __call__(self, U, _U, i):
        delta = np.max(np.abs(U - _U))
        print(f'\rSolving laplace. Discrepancy = {delta:.3}              ', end='', flush=True)
        self.log[-1][0].append(i)
        self.log[-1][1].append(delta)
        self.U = U
        self.dump()
        if (delta < self.eps):
            self.exitlog = 'Laplace solved'
            return True
        if (delta > self.max_div):
            self.exitlog = 'The Laplace problem divergented'
            return True
        if (i > self.max_steps):
            self.exitlog = 'Limit of the Laplace solver iterations'
            return True
        return False
    
    def set_grid(self, grid):
        self.grid = grid
        self.log.append([[], []])
        self.eps = self.epseps[self.counter] if self.counter < len(self.epseps) else self.epseps[-1]
        self.counter += 1
        
#%%
def create_edge_mask3D(U, x_edge, y_edge, z_edge): # x_edge=[], y_edge=[0,-1], z_edge=[0,-1]
    edge_mask = np.full_like(U, False, dtype=bool)
    edge_mask[x_edge, :, :] = True 
    edge_mask[:, y_edge, :] = True
    edge_mask[:, :, z_edge] = True

    return edge_mask

# indexing[:, [0, -1], :], indexing[:, :, [0, -1]]

#%%
def stop_by_mid_indexing(eps=1e-5):
    def f(U, _U):
        idx = mid_indexing(U)
        delta = np.max(np.abs(U[idx] - _U[idx]))
        return (delta < eps)
    return f

def laplace_solve(U, boundary_conditions, stop_condition):

    kernel = get_kernel(U)
    kernel = kernel/np.sum(kernel) 
    
    boundary_conditions.reset(U)
    _U = U.copy()

    step = 0
    # skip_steps = max(U.shape)*2
    # epseps = []

    while True:
        step += 1
        _U, U = U, _U
        # boundary_conditions.reset(_U)      # always _U -> U
        convolve(_U, kernel, mode='nearest', output=U)
        boundary_conditions.reset(U)

        # if (step < skip_steps): 
        #     continue
        
        if step % 100 == 0: # Check can be expensive
            # delta = np.max(np.abs(U[test_indexing] - _U[test_indexing]))
            # epseps.append(delta)
            if stop_condition(U, _U, step): 
                break

    print('\b\b\b\b\b\b\b\b\b\b\b\b\b\b. Total number of steps = {}'.format(step), end='')
    
    # plt.figure()
    # plt.plot(epseps)

    boundary_conditions.reset(U) 
    return U

def gauss_SOR(U, boundary_conditions, stop_condition, omega=1.7):
    kernel = get_kernel(U)
    kernel = kernel/np.sum(kernel) 
    
    kernel *= omega
    kernel[mid_indexing(kernel)] = 1. - omega
    
    boundary_conditions.reset(U)
    _U = U.copy()

    step = 0

    while True:
        step += 1
        _U, U = U, _U
        convolve(_U, kernel, mode='nearest', output=U)  
        boundary_conditions.reset(U)
        
        if step % 100 == 0: # Check can be expensive
            if stop_condition(U, _U, step): 
                break

    print('\b\b\b\b\b\b\b\b\b\b\b\b\b\b. Total number of steps = {}'.format(step), end='')
    
    boundary_conditions.reset(U) 
    return U

#%%
def calcE(U, grid_step, plates_masks): 
    # explicitely for 3D and 2D
    if len(U.shape) == 3: 
        Ex, Ey, Ez = np.gradient(-1.0*U, grid_step)
        # set zero E in the cells corresponding to plates
        for mask in plates_masks: 
            Ex[mask], Ey[mask], Ez[mask] = 0.0, 0.0, 0.0
        
        return Ex, Ey, Ez

    elif len(U.shape) == 2: 
        Ex, Ey = np.gradient(-1.0*U, grid_step)
        # set zero E in the cells corresponding to plates
        for mask in plates_masks: 
            Ex[mask], Ey[mask] = 0.0, 0.0
        
        return Ex, Ey
    else: 
        return None # ???
    
#%%
def reinterp_U(U, old_grid, new_grid):
    points = _xx_yy_etc_from_grid(old_grid)
    U_interp = FastGridInterpolator(points, U, fill_value=0.0)
    new_U = scalar_field_on_grid(new_grid, U_interp)
    return new_U

def iterative_laplace(grid_gabarits, grid_steps, boundary_conditions, stop_condition, U0_interp=None, omega=1.7):
    U = None
    grid = None
    for step in grid_steps:
        print(f'iterative_laplace : Solving for grid resolution {step}')
        old_grid, grid = grid, Grid.from_domain(grid_gabarits[0], grid_gabarits[1], step).grid 
        
        print('Setting initial guess...', end='')
        with StopWatch('\rSetting initial guess : Done '):
            if U is None:
                if U0_interp is None:
                    U = np.zeros(grid.shape[1:])
                else:
                    U = scalar_field_on_grid(grid, U0_interp)
            else:
                U = reinterp_U(U, old_grid, grid)
        
        print('Setting conditions...', end='')
        with StopWatch('\rSetting conditions : Done '):
            boundary_conditions.set_grid(grid)
            stop_condition.set_grid(grid)
        with StopWatch('. Time '):
            U = gauss_SOR(U, boundary_conditions, stop_condition, omega=omega)
        print('\n')
    return U
            
#%%
Group3D.call_strategies['set_grid'] = JustCallStrategy()
Group3D.call_strategies['reset'] = JustCallStrategy()