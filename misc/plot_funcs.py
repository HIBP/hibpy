# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:09:37 2024

@author: Ammosov, Krokhalev
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

from ..phys.constants import SI_1keV
from ..geom.geom import plot_point, get_coord_indexes
#%% basic plots
def color_by_Ebeam(Ebeam):
    """
    Color code generator using beam energy value.
    
    Parameters
    ----------
    Ebeam : float
        Beam energy in Joules [J].
    Returns
    -------
    str
        Matplotlib color code.
        
    """
    i = round(Ebeam/SI_1keV/20)
    return 'C%d' % i 

def plot_grid(traj_list, axes_code=None, legend_on=True, **kwargs):
    """
    Plot detector grid function.

    Parameters
    ----------
    traj_list : list[Trajectory, ...]
        List of Trajectory class objects.
    axes_code : str, optional
        Show different axes: "XY", "YZ", etc. The default is None (shows "XY" axes).
    legend_on : bool, optional
        Enable legend. The default is True.
    **kwargs : kwargs
        kwargs for plot_point (plt.plot - matplotlib backend).

    Returns
    -------
    None.

    """
    E_cache = 0.
    for tr in traj_list:
        E_tmp = round(tr.Ebeam/SI_1keV)
        if not np.isclose(E_tmp, E_cache):
            plot_point(np.asarray(tr.rrvv)[tr.secondary_index, :3], color=color_by_Ebeam(tr.Ebeam),
                       label=str(E_tmp) + ' keV', axes_code=axes_code, **kwargs)
            E_cache = E_tmp
        else:
            plot_point(np.asarray(tr.rrvv)[tr.secondary_index, :3], color=color_by_Ebeam(tr.Ebeam), 
                       axes_code=axes_code, **kwargs)
    if legend_on:
        plt.legend()

def plot_fan(traj_list, legend_on=True, **kwargs):
    if traj_list:
        if 'label' in kwargs and legend_on:
            lbl = kwargs.pop('label')
            for tr in traj_list[:-1]:
                tr.plot(**kwargs)
            traj_list[-1].plot(**kwargs, label=lbl)
        else:
            for tr in traj_list:
                tr.plot(**kwargs)

#%% fatbeam
def plot_filament_to_slits(tr, slit_numbers=None, geometry_group=None, 
                           title_on=True, legend_on=True, axes_code=None):
    """
    Plot trajectories from partial_dense_fan which goes to specific slits.

    Parameters
    ----------
    tr : Trajectory class object
        Trajectory with calculated partial_dense_fan.
    slit_numbers : list[int, ...], optional
        List with slit numbers. The default is None.
    geometry_group : Group3D(), optional
        Group3D with different geometry objects with plot method. The default is None.
    title_on : bool, optional
        Enable plot title. The default is True.
    legend_on : bool, optional
        Enable legend. The default is True.
    axes_code : str, optional
        Show different axes: "XY", "YZ", etc. The default is None (shows "XY" axes).

    Returns
    -------
    None.

    """
    
    if len(tr.partial_dense_fan) > 0:
        fans_to_slits = tr.slit_bins
        
        if geometry_group is not None:
            geometry_group.plot(axes_code=axes_code)
        
        if slit_numbers is None:
            slit_numbers = fans_to_slits.keys()
        
        for slit_number in slit_numbers:
            plot_fan(fans_to_slits[slit_number], color='C%d'%slit_number, 
                     axes_code=axes_code, label=slit_number)
            
            for tr_in_fan in fans_to_slits[slit_number]:
                plot_point(tr_in_fan.obstacle.intersect_with_segment(tr_in_fan.rrvv[-1, :3], tr_in_fan.rrvv[-2, :3]), 
                           color='C%d'%slit_number, axes_code=axes_code)
        
        if legend_on:
            plt.legend()
        
        if title_on:
            plt.title(f'Ebeam = {int(tr.Ebeam/SI_1keV)} keV, UA2 = {int(tr.U["A2"])} kV')
    
   
#%% fatbeam maps
def plot_lambdas_map(traj_list, slit_number, grid_on=True, title_on=True, **kwargs):
    """
    Plot detector grid where dot color represents lambda parameter.

    Parameters
    ----------
    traj_list : list[Trajectory, ...]
        List of Trajectory class objects.
    slit_number : int
        Slit number of energy analyzer aperture.
    grid_on : bool, optional
        Enable plot grid. The default is True.
    title_on : bool, optional
        Enable plot title. The default is True.
    **kwargs : kwargs
        kwargs for plt.scatter (matplotlib backend).

    Returns
    -------
    None.

    """
    
    cm = plt.cm.get_cmap('jet')
        
    for tr in traj_list:
        pos = tr.secondary.rrvv[0, :3]
        
        lam = 0
        if slit_number in tr.lambdas.keys():
            lam = tr.lambdas[slit_number]
            
        sc = plt.scatter(pos[0], pos[1], c=lam*1000, cmap=cm, **kwargs)     
        
    plt.xlabel('X, m')
    plt.ylabel('Y, m')
    plt.colorbar(sc, label = r'$\lambda$, mm')
    if grid_on:
        plt.grid()
    plt.show()
    
def plot_slits_text_map(traj_list, axes_code=None, **kwargs):
    """
    Plot detector grid where instead of dots there are numbers representing
    how many entrance slits the trajectory from a given point hit.

    Parameters
    ----------
    traj_list : list[Trajectory, ...]
        List of Trajectory class objects.
    axes_code : str, optional
        Show different axes: "XY", "YZ", etc. The default is None (shows "XY" axes).
    **kwargs : kwargs
        kwargs for plt.text (matplotlib backend).

    Returns
    -------
    None.

    """
    X, Y = get_coord_indexes(axes_code)
    for traj in traj_list:
        xy = [traj.secondary.rrvv[0, X], traj.secondary.rrvv[0, Y]]
        plt.text(xy[0], xy[1], len(traj.lambdas), ha='center', va='center', **kwargs)
        
    plt.xlabel('X, m')
    plt.ylabel('Y, m')
    plt.show()