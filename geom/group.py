# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:45:44 2023

@author: reonid
"""

#import numpy as np
#import matplotlib.pyplot as plt

from .geom import (vec3D, rotateMx, identMx, invMx, outside_gabarits, join_gabarits,
                   vec2D, identMx2D)
from .groupstrat import group_method, DefaultStrategy
from ..misc.cienptas import acc_vstack 

#%%

class Group3D: 
    call_strategies = {}
    
    def __init__(self, elemlist=None, name=None): 
        if elemlist is None:
            elemlist = []

        self._list = elemlist  # list(elemlist)
        self._gabarits = None
        self._active_elem = None
        self.name = name

        self._vec = vec3D(0, 0, 0)
        self._mx  = identMx()
        self._imx = identMx()
         
    def append(self, elem): 
        self._list.append(elem)
        self._gabarits = None
        
    def extend(self, elemlist): 
        self._list.extend(elemlist)
        self._gabarits = None

    def remove(self, elem): 
        self._list.remove(elem)
        self._gabarits = None

    def __iter__(self): 
        return self._list.__iter__()

    def __getitem__(self, arg): 
        if isinstance(arg, str):
            return self._get_by_name(arg)
        return self._list[arg]
    
    def _get_by_name(self, name):
        for elem in self._list:
            try:
                if elem.name == name:
                    return elem
            except: 
                pass
        raise KeyError(f'{name}')
    
    def __setitem__(self, arg, item):
        self._list[arg] = item
        self._gabarits = None

    def __getattr__(self, name): 
        if name.startswith('__'):
            raise AttributeError

        call_strategy = self.call_strategies.get(name, DefaultStrategy())
        return group_method(self, call_strategy, name)
    
    def elements(self): 
        return self._list 

    def translate(self, vec): 
        for elem in self._list: 
            elem.translate(vec)

        self._vec += vec
        self._gabarits = None

        return self

    def transform(self, mx): 
        for elem in self._list: 
            elem.transform(mx)

        self._vec = mx.dot(self._vec)
        self._mx  = mx.dot(self._mx)
        self._imx = invMx(self._mx)  
        self._gabarits = None

        return self

#    def trans(self, *args):
#        return _trans_(self, *args)

    
    def rotate(self, pivot_point, axis, angle): 
        mx = rotateMx(axis, angle)
        self.translate(-pivot_point)
        self.transform(mx)
        self.translate(pivot_point)
        return self


    def plot(self, axes_code=None, **kwargs): 
        for elem in self._list: 
            elem.plot(axes_code=axes_code, **kwargs)
        return self


    def recalc_gabarits(self): 
        self._gabarits = None # calc_gabarits(self.points() )   
        for elem in self._list: 
            self._gabarits = join_gabarits(self._gabarits, elem.gabarits())

    def gabarits(self): 
        if self._gabarits is None: 
            self.recalc_gabarits()
        return self._gabarits

    #--------------------------------------------------------------------------

    # electrostatic geometry: for initial conditions of Laplace eq.  
    def contains_pt(self, pt):
        self._active_elem = None
        
        if outside_gabarits(pt, self.gabarits()): 
            return False
        
        for elem in self._list: 
            if elem.contains_pt(pt): 
                self._active_elem = elem
                return True

        return False
    
    def masks_with_voltages(self, grid):
        result = []
        for elem in self._list:
            result.extend(elem.masks_with_voltages(grid))
        return result
    
    # obstacle geometry: trajectory passing
    def intersect_with_segment(self, pt0, pt1): 
        self._active_elem = None
        
        for elem in self._list: 
            intersect_pt = elem.intersect_with_segment(pt0, pt1)
            if intersect_pt is not None: 
                self._active_elem = elem
                return intersect_pt
        return None

    # magnetic geometry
    def calcB(self, r): 
        B = vec3D(0, 0, 0)
        for elem in self._list: 
            B += elem.calcB(r)
            
        return B    

    def array_of_IdLs(self): 
        rr = None
        IdL = None
        for elem in self._list: 
            _rr, _IdL = elem.array_of_IdLs() # 
            
            rr  = acc_vstack(rr,  _rr)  #np.vstack((rr,  _rr )) if rr  is not None else _rr
            IdL = acc_vstack(IdL, _IdL) #np.vstack((IdL, _IdL)) if IdL is not None else _IdL
        
        return rr, IdL

class Group2D(Group3D):
    def __init__(self, elemlist=None, name=None): 
        if elemlist is None:
            elemlist = []

        self._list = elemlist  # list(elemlist)
        self._gabarits = None
        self._active_elem = None
        self.name = name

        self._vec = vec2D(0, 0)
        self._mx  = identMx2D()
        self._imx = identMx2D()
        