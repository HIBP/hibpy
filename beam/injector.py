# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:52:10 2023

@author: Sarancha_GA, Krohalev_OD
"""
#%% imports
from copy import deepcopy
import numpy as np

from ..misc.grid  import _xx_yy_etc_from_grid, _mgrid
from ..geom.group import Group3D, Group2D
from ..geom.geom  import (vec3D, identMx, invMx, outside_gabarits, 
                          join_gabarits, mx2Dto3D, vNorm)
from ..geom.prim  import gabaritbox2D
from ..geom.box   import gabaritbox3D

#%%
class Injector2D:
    def __init__(self, electrodes2D, raw_data=None):
        self.electrodes = Group2D(electrodes2D)
        self.dict = {}
        for i, el in enumerate(self.electrodes):
            if el.name is None:
                el.name = i
            self.dict[el.name] = el
        self._Udict = {}
        self._main_box = gabaritbox2D(self.electrodes.gabarits())
        self._transformable = Group2D([self.electrodes, self._main_box])
        self.raw_data = raw_data
        
    def __getitem__(self, name):
        return self.dict[name]
    
    def __setitem__(self, name, value):
        self.dict[name] = value
    
    @property
    def _vec(self):
        return self._main_box._vec
    
    @property
    def _mx(self):
        return self._main_box._mx
    
    @property
    def _imx(self):
        return self._main_box._imx
    
    @property
    def Udict(self):
        return self._Udict

    @Udict.setter
    def Udict(self, Udict):
        for name in Udict.keys():
            U = Udict[name]
            self._Udict[name] = U
            self.dict[name].set_U(U)
        
    def transform(self, mx):
        self._transformable.transform(mx)

    def translate(self, vec):
        self._transformable.translate(vec)
        
    def contains_pt(self, pt):
        return self.electrodes.contains_pt(pt)
        
    def masks_with_voltages(self, grid):
        return self.electrodes.masks_with_voltages(grid)
    
    def plot(self, axes_code=None, **kwargs):
        self.electrodes.plot(axes_code=axes_code, **kwargs)
        
    def restore(self):
        self.translate(-self._vec)
        self.transform(self._imx)
        electrodes3D = []
        for electrode in self.electrodes:
            electrode3D = Group3D(electrode.restore_3D())
            electrodes3D.append(electrode3D)
        self.transform(self._mx)
        self.translate(self._vec)
        vec = np.hstack((self._vec, [0.]))
        mx = mx2Dto3D(self._mx)
        injector3D = Injector3D(electrodes3D)
        injector3D.transform(mx)
        injector3D.translate(vec)
        return injector3D
    
    def gabarits(self):
        return self.electrodes.gabarits()
    
    def recalc_gabarits(self):
        self.electrodes.recalc_gabarits()
        
    def U(self, name):
        UU = [self.Udict[key] if (name in key) else 0.0 for key in self.Udict.keys()]
        return max(UU)
    
    def set_E_interp(self, E_interp_dict):
        self._interp_dict = E_interp_dict
        gabarits = None
        for name in E_interp_dict:
            gabarits = join_gabarits(gabarits, 
                                     np.asarray((E_interp_dict[name].volume_corner1, 
                                                 E_interp_dict[name].volume_corner2)))
        self._domain_box = gabaritbox2D(gabarits)
        self._domain_box.transform(self._main_box._vec)
        self._domain_box.translate(self._main_box._mx)
        self._transformable.append(self._domain_box)
    
    def E(self, point):
        if outside_gabarits(point, self._domain_box.gabarits()):
            return np.zeros((2,))        
        r_loc = self._imx.dot(point - self._vec)
        E = np.zeros((2,))
        for name in self._interp_dict:    
            E += self._interp_dict[name](r_loc)*self.U(name)
        return self._mx.dot(E)
            
#%%
class Injector3D:
    def __init__(self, electrodes):
        self.axis = vec3D(1, 0, 0)
        self._vec = vec3D(0, 0, 0)
        self._mx = identMx()
        self._imx = identMx()
        self.electrodes = Group3D(electrodes)
        self.dict = {}
        for i, el in enumerate(self.electrodes):
            if el.name is None:
                el.name = i
            self.dict[el.name] = el
        self._Udict = {}
        self._gabarit_box = gabaritbox3D(self.electrodes.gabarits())
        self._main_box = deepcopy(self._gabarit_box)
        self._main_box._vec = vec3D(0, 0, 0)
        self._transformable = Group3D([self.electrodes, self._main_box, 
                                       self._gabarit_box])
        
    def __getitem__(self, name):
        return self.dict[name]
    
    @property
    def Udict(self):
        return self._Udict
    
    @Udict.setter
    def Udict(self, Udict):
        for name in Udict.keys():
            U = Udict[name]
            self._Udict[name] = U
            # self.dict[name].set_U(U) #!!!
        
    def transform(self, mx):
        self._vec = mx.dot(self._vec)
        self._mx = mx.dot(self._mx)
        self._imx = invMx(self._mx)
        self.electrodes.transform(mx)
        
    def translate(self, vec):
        self._vec += vec
        self.electrodes.translate(vec)
        
    def intersect_with_segment(self, r0, r1):
        self.electrodes.intersect_with_segment(r0, r1)
        
    def contains_pt(self, pt):
        self.electrodes.contains_pt(pt)
        
    def masks_with_voltages(self, grid):
        if grid.shape[0] == 2:
            return self.masks_with_voltages2D(grid)
        return self.electrodes.masks_with_voltages(grid)
    
    def masks_with_voltages2D(self, grid):
        self.electrodes.translate(-self._vec)
        self.electrodes.transform(self._imx)
        xx, yy = _xx_yy_etc_from_grid(grid)
        zz = [0.]
        grid3D = _mgrid(xx, yy, zz)
        masks_voltages3D = self.electrodes.masks_with_voltages(grid3D)
        masks_voltages = []
        for mask, voltage in masks_voltages3D:
            mask2D = mask[:, :, 0]
            masks_voltages.append([mask2D, voltage])
        self.electrodes.transform(self._mx)
        self.electrodes.translate(self._vec)
        return masks_voltages 
    
    def plot(self, axes_code=None, **kwargs):
        self.electrodes.plot(axes_code=axes_code, **kwargs)
        
    def gabarits(self):
        return self.electrodes.gabarits()
    
    def recalc_gabarits(self):
        self.electrodes.recalc_gabarits()
        
    def U(self, name):
        UU = [self.Udict[key] if (name in key) else np.nan for key in self.Udict.keys()]
        return max(UU)
    
    def set_E_interp(self, E_interp_dict):
        self._interp_dict = E_interp_dict
        gabarits = None
        for name in E_interp_dict:
            gabarits = join_gabarits(gabarits, 
                                     np.asarray((E_interp_dict[name].lower_corner, 
                                                 E_interp_dict[name].upper_corner)))
        if len(gabarits[0]) == 2:
            gabarits = [(gabarits[0][0], gabarits[0][1], gabarits[0][1]),
                        (gabarits[1][0], gabarits[1][1], gabarits[1][1])]
            
        self._domain_box = gabaritbox3D(gabarits)
        # print(self._main_box._vec)
        self._domain_box.transform(self._main_box._mx)
        self._domain_box.translate(self._main_box._vec)
        self._transformable.append(self._domain_box)
    
    def E(self, point):
        point_local = self._imx.dot(point - self._vec)
        if outside_gabarits(point_local, self._domain_box.gabarits()):
            return np.zeros((3,))     
        
        # r_loc = self._imx.dot(point - self._vec)
        # z     = r_loc[0]
        # r_vec = np.asarray([0.0, r_loc[1], r_loc[2]])
        # r = vNorm(r_vec)
        
        # E_loc = np.zeros((2,))
        E_loc = np.zeros((3,))
        for name in self._interp_dict:    
            # E_loc += self._interp_dict[name](np.array([z, r]))*self.U(name)
            
            E_loc += self._interp_dict[name](point_local)*self.Udict[name]
            
        # E = E_loc[0]*self.axis + E_loc[1]*r_vec/r
        return self._mx.dot(E_loc)