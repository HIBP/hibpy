# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:14:29 2024

@author: Krohalev_OD
"""
#%%
import numpy as np

from ..geom.geom import vec3D

#%%
class VoltageLoop():
    def __init__(self, radius, center):
        self.normal = vec3D(0, 1, 0)
        self.center = center
        self.radius = radius
        self.signal = None   
        self._cache = None
        
    def __call__(self):
        self.measure()
        return self.signal
    
    def measure(self, currents):
        if self._cache is None:
            self.signal = currents.calc_psi()
            
        else:
            self.signal = 0.
            for current in currents:
                