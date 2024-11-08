# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:23:23 2024

@author: reonid
"""
#%%
from ..geom.geom import vec3D, identMx, invMx
#%%
class History:
    """
    For saving object transformation history.
    
    _vec: numpy.array([x, y, z], dtype=np.float64)
        3D vector.
    _mx: numpy.ndarray[np.float64, ...], shape=(3, 3)
        Transformation matrix.
    _mx: numpy.ndarray[np.float64, ...], shape=(3, 3)
        Invrse transformation matrix.
        
    Example:
        import numpy as np
        from hibpy.geom.history import History
        from hibpy.geom.geom import direct_transform, inverse_transform
        
        history = History()
        history.translate(geom.vec3D(0, 0, -0.185))
        history.transform(geom.xRotateMx(-np.pi/2))
        history.transform(geom.yRotateMx(np.deg2rad(11.18)))
        
        busbar = load_busbar(prefix + r'\programs\SynHIBP\devices\T-15MD\geometry\\')
        geom.direct_transform(busbar, history)
        geom.inverse_transform(busbar, history)
    """
    def __init__(self):
        self._vec = vec3D(0, 0, 0) 
        self._mx  = identMx()
        self._imx = identMx()
        
    def transform(self, mx):
        self._vec = mx.dot(self._vec)
        self._mx  = mx.dot(self._mx)
        self._imx = invMx(self._mx)
    
    def translate(self, vec):
        self._vec += vec        
