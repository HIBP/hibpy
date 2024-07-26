# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:15:14 2024

@author: Ammosov
"""

#%%
from copy import deepcopy
import numpy as np

from ..geom.group import Group3D
from ..geom.geom import get_coord_indexes, vec3D, normalizedVector
from ..geom.prim import Polygon3D

#%%

def rect3D(length, width, axes_code=None, polygon_cls=Polygon3D):
    pts = np.zeros((4, 3))
    X, Y = get_coord_indexes(axes_code)
    pts[[0, 1], X] = -length * 0.5
    pts[[2, 3], X] =  length * 0.5
    pts[[1, 2], Y] =  width * 0.5
    pts[[0, 3], Y] = -width * 0.5
    return polygon_cls(pts)

class SlitPolygon(Polygon3D): 
    def __init__(self, points):
        super().__init__(points)
        self.number = None
    
class SlitPad():
    def __init__(self, n, slit_width, slit_length, slit_dist, pad_width=None, pad_length=None):
        self.n = n
        self.slit_length = slit_length
        self.slit_width = slit_width
        self.slit_dist = slit_dist
        self.slits = []
        self.transparent_slits = False
        
        slit_prototype = rect3D(slit_length, slit_width, axes_code='ZY', polygon_cls=SlitPolygon)

        for i in range(n): 
            slit = deepcopy(slit_prototype)
            slit.translate( vec3D(0, (i - (n - 1)*0.5)*slit_dist, 0) )
            slit.number = i + 1
            self.slits.append(slit)
        
        if pad_width is None:
            self.pad_width = self.slit_dist * (n + 1) # default value
        else:
            assert pad_width > slit_width, f'Error: pad width {pad_width} < slit width {slit_width}'
            self.pad_width = pad_width
        
        if pad_length is None:
            self.pad_length = slit_length * 1.2 # 120% of slit length
        else:
            assert pad_length > slit_length, f'Error: pad length {pad_length} < slit length {slit_length}'
            self.pad_length = pad_length

        self.pad = rect3D(self.pad_length, self.pad_width, axes_code='ZY')
        self._active_elem = None
        
        self._transformable = Group3D(self.slits) 
        self._transformable.append(self.pad)
        
    @property
    def basis(self):
        """
        basis vector 0 is in preffered obj direction if where is any - pitch
        basis vector 1 is in polygon plane normal to bv0             - yaw/lateral
        basis vector 2 is normal to polygon                          - normal

        Returns
        -------
        basis : np.array of float with shape (3,3)
        """
        b = np.zeros((3, 3))
        b[0] = normalizedVector(self.slits[0]._points[0] - self.slits[0]._points[1])
        b[1] = normalizedVector(self.slits[0]._points[2] - self.slits[0]._points[1])
        b[2] = np.cross(b[0], b[1])
        return b
    
    @property
    def center(self):
        return self.pad._center
    
    @property
    def transformable(self):
        return self._transformable
    
    def transform(self, mx):
        self.transformable.transform(mx)
        
    def translate(self, vec):
        self.transformable.translate(vec)
        
    def intersect_with_segment(self, pt0, pt1):
        self._active_elem = None
        for s in self.slits:
            pt = s.intersect_with_segment(pt0, pt1)
            if pt is not None:
                if self.transparent_slits:
                    return None
                self._active_elem = s
                return pt
        
        pt = self.pad.intersect_with_segment(pt0, pt1)
        if pt is not None:
            self._active_elem = self.pad
            return pt
        
    def plot(self, axes_code=None, **kwargs):
        self.pad.plot(axes_code=axes_code, **kwargs)
        for s in self.slits:
            s.plot(axes_code=axes_code, **kwargs)
    
#%%

def test_slit():
    n, slit_width, slit_length, slit_dist = 7, 0.02, 0.1, 0.04
    slits = SlitPad(n, slit_width, slit_length, slit_dist)
    slits.plot(axes_code='ZY')
    print(slits.center)
        
if __name__ == '__main__':
    test_slit()