# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:07:06 2023

@author: Eliseev_LG, Krohalev_OD
"""
#%% imports
import numpy as np

from .laplace import (MasksBoundaryConditions, MaxStopCondition, 
                      MidIndexingStopCondition, laplace_solve, iterative_laplace,
                      calcE)
from ..misc.grid import vector_field_on_grid, resolution, func_on_points

#%% calculators
def B_dict_on_grid(grid, coils):
    """
    Meant to calc B for poloidal fields coils
    Coils must have attribute 'name'.
    """
    B_dict = {}
    try:     
        for coil in coils:
            B = vector_field_on_grid(grid, coil.calcB)
            B_dict[coil.name] = B
    except AttributeError:
        raise AttributeError("calc_B_dict_on_grid : Coils must have names")
    return B_dict

def B_dict_on_points(points, coils):
    """
    Meant to calc B for poloidal fields coils
    Coils must have attribute 'name'.
    """
    B_dict = {}
    try:     
        for coil in coils:
            B = func_on_points(points, coil.calcB)
            B_dict[coil.name] = B
    except AttributeError:
        raise AttributeError("calc_B_dict_on_grid : Coils must have names")
    return B_dict

def calc_efield(plates, grid, eps, dump_name):
    masks_voltages = plates.masks_with_voltages(grid)
    plates_masks = np.array(masks_voltages, dtype=object)[:, 0]
    boundary_conditions = MasksBoundaryConditions(plates.masks_with_voltages)
    boundary_conditions.set_grid(grid)
    stop_condition = MaxStopCondition(eps, dump_name)
    stop_condition.set_grid(grid)
    U = np.zeros(grid.shape[1:])
    U = laplace_solve(U, boundary_conditions, stop_condition)
    grid_res = resolution(grid)
    E = np.array(calcE(U, grid_res, plates_masks))
    return E, U

def iterative_calc_efield(plates, gabarits, steps, dump_name, eps, U_interp=None, omega=1.7):
    boundary_conditions = MasksBoundaryConditions(plates.masks_with_voltages)
    stop_condition = MaxStopCondition(eps, dump_name)
    try:
        U = iterative_laplace(gabarits, steps, boundary_conditions, stop_condition, 
                              U0_interp=U_interp, omega=omega)
    except KeyboardInterrupt:
        U = stop_condition.U
    E = np.array(calcE(U, steps[-1], []))
    return E, U, stop_condition