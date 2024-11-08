# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:22:54 2024

@author: reonid
"""
#%%
import numpy as np

from ..geom.box import gabaritbox3D
from ..geom.geom import outside_gabarits, vec3D, identMx, invMx
#%%
class VectorField:
    """
    For adding transformations to vector field e.g. adding transformation to
    magnetic field (RegularGridVectorInterpolator3D).
    
    _domain_box: geom.box.Box3D
        Box3D which defines domain (calculation area) of vector field.
    _interp: e.g. misc.interp.RegularGridVectorInterpolator3D
        Vector field interpolator.
    amplitude: float
        Vector field amplitude.
    _vec: numpy.array([x, y, z], dtype=np.float64)
        3D vector.
    _mx: numpy.ndarray[np.float64, ...], shape=(3, 3)
        Transformation matrix.
    _mx: numpy.ndarray[np.float64, ...], shape=(3, 3)
        Invrse transformation matrix.
    """
    def __init__(self, interp, amplitude=1.0):
        self._domain_box = gabaritbox3D(
            (interp.lower_corner, interp.upper_corner))
        self._interp = interp
        self.amplitude = amplitude

        self._vec = vec3D(0, 0, 0)
        self._mx = identMx()
        self._imx = identMx()

    def transform(self, mx):
        self._vec = mx.dot(self._vec)   # super().transform(mx)
        self._mx = mx.dot(self._mx)
        self._imx = invMx(self._mx)
        self._domain_box.transform(mx)

    def translate(self, vec):
        self._vec += vec
        self._domain_box.translate(vec)

    def value_at(self, r):
        if outside_gabarits(r, self._domain_box.gabarits()):
            return np.zeros((3,))
        r_loc = self._imx.dot(r - self._vec)
        result = self._interp(r_loc) * self.amplitude
        return self._mx.dot(result)
    
    def __call__(self, r):
        return self.value_at(r)
    