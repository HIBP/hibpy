# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:42:34 2023

@author: reonid
"""

import numpy as np
import matplotlib.pyplot as plt

from .geom import (pt3D, vec3D, size3D, 
                  identMx, invMx, rotateMx, xwardRotateMx, xScaleMx, skewMx, 
                  plot_point, plot_polygon, 
                  _ptInPolygon3D_, _intersect_plane_segm_, 
                  calc_gabarits, join_gabarits, outside_gabarits, testpoints)    # inside_gabarits

from .prim import Polygon3D

#%% additional functions
def _trans_(obj3D, *args): 
        for a in args: 
            if len(a.shape) == 1: 
                obj3D.translate(a)
            elif len(a.shape) == 2: 
                obj3D.transform(a)
            else: 
                raise Exception('_trans_: invalid argument')
        return obj3D

def _rotate_(obj3D, pivot_point, axis, angle): 
        mx = rotateMx(axis, angle)
        _trans_(obj3D, -pivot_point, mx, pivot_point)
        return obj3D

#%%
class Box3D: 
    def __init__(self, xlen, ylen, zlen, eps=0.0): 
        # history of transformations
        self._vec = vec3D(0, 0, 0)
        self._mx  = identMx()
        self._imx = identMx()
        # gabarits
        dx, dy, dz = xlen*0.5 + eps, ylen*0.5 + eps, zlen*0.5 + eps
        self._d = size3D(dx, dy, dz)
        self._gabarits = None

        p1 = pt3D( dx,  dy,  dz ) # 0
        p2 = pt3D(-dx,  dy,  dz ) # 1
        p3 = pt3D(-dx, -dy,  dz ) # 2
        p4 = pt3D( dx, -dy,  dz ) # 3
        p5 = pt3D( dx,  dy, -dz ) # 4
        p6 = pt3D(-dx,  dy, -dz ) # 5
        p7 = pt3D(-dx, -dy, -dz ) # 6
        p8 = pt3D( dx, -dy, -dz ) # 7
        
        self._points = [p1, p2, p3, p4, p5, p6, p7, p8]
        self.recalc_gabarits()

    def translate(self, vec): 
        self._vec += vec
        self._points = [pt + vec for pt in self._points]  
        self.recalc_gabarits()
        return self

    def transform(self, mx): 
        self._vec = mx.dot(self._vec)
        self._mx  = mx.dot(self._mx)
        self._imx = invMx(self._mx)  
        self._points = [mx.dot(pt) for pt in self._points]
        self.recalc_gabarits()
        return self

    def trans(self, *args):
        return _trans_(self, *args)

    def recalc_gabarits(self): 
        self._gabarits = calc_gabarits(self.points() )

    def plot(self, axes_code=None, **kwargs): 
        color = kwargs.get('color')
        if color is None:
            for pgn in self.polygons(): 
                color = plot_polygon(pgn, axes_code, color=color, **kwargs)[0].get_color()
        else:         
            for pgn in self.polygons(): 
                plot_polygon(pgn, axes_code, **kwargs)
        
    def contains_pt(self, pt): 
        if outside_gabarits(pt, self._gabarits): 
            return False
        
        pt0 = self._imx.dot(pt - self._vec)
        d = self._d
        return (  (pt0[0] >= -d[0]) and (pt0[1] >= -d[1]) and (pt0[2] >= -d[2]) and 
                  (pt0[0] <=  d[0]) and (pt0[1] <=  d[1]) and (pt0[2] <=  d[2])     )      
            

    def intersect_with_segment(self, pt0, pt1): 
        for pl, pgn_pts in zip( self.planes(), self.polygons() ): # check _intersect_plane_segm_ fast because normal is given
            if _intersect_plane_segm_(pl, (pt0, pt1)): 
                pgn = Polygon3D(pgn_pts) # normal is recalculated
                inters_pt = pgn.intersect_with_segment(pt0, pt1)
                if inters_pt is not None: # HER WAS AN ERROR
                    return inters_pt
            
        return None
        
    def points(self): 
        return self._points  # yield from self._points

    def gabarits(self): 
        return self._gabarits

    def tetragon(self, i, j, k, m):
        return [ self._points[i], self._points[j], self._points[k], self._points[m] ]
    
    def polygons(self): # as point list
        yield self.tetragon(0, 3, 7, 4)  # x > 0
        yield self.tetragon(1, 2, 6, 5)  # x < 0 
        yield self.tetragon(0, 1, 5, 4)  # y > 0
        yield self.tetragon(2, 3, 7, 6)  # y < 0 
        yield self.tetragon(0, 1, 2, 3)  # z > 0
        yield self.tetragon(4, 5, 6, 7)  # z < 0 

    def planes(self): # (point, normal) # ??? normal is not normalized (L != 1.0)
        yield self._points[0], self._points[0] - self._points[1]  # x > 0
        yield self._points[1], self._points[1] - self._points[0]  # x < 0 
        yield self._points[1], self._points[1] - self._points[2]  # y > 0
        yield self._points[2], self._points[2] - self._points[1]  # y < 0 
        yield self._points[2], self._points[2] - self._points[6]  # z > 0
        yield self._points[6], self._points[6] - self._points[2]  # z < 0 


    def rotate(self, pivot_point, axis, angle): 
        mx = rotateMx(axis, angle)
        self.trans(-pivot_point, mx, pivot_point)
        return self

    def rotate_with_skew(self, fixed_plane, angle): 
        # to emulate Philipp's transformation in plate_flags
        
        # prefix
        pln_pt0, pln_normal = fixed_plane
        xward_mx = xwardRotateMx(pln_normal)
        self.trans(-pln_pt0, xward_mx)
        
        angle_ = np.arctan(np.sin(angle))
        mx = skewMx(angle_, 'YX')
        self.trans(mx)
         
        mx = xScaleMx( np.cos(angle) )
        self.trans(mx)
        
        # postfix
        self.trans(invMx(xward_mx), pln_pt0)
        return self



#%% 
        
def limitbox3D(xlim, ylim, zlim): 
    xmin, xmax = xlim
    ymin, ymax = ylim
    zmin, zmax = zlim

    box = Box3D(xmax-xmin, ymax-ymin, zmax-zmin)
    box.translate( 0.5*vec3D(xmax+xmin, ymax+ymin, zmax+zmin) )
    return box    

def gabaritbox3D(gbr): 
    (xmin, ymin, zmin), (xmax, ymax, zmax) = gbr

    box = Box3D(xmax-xmin, ymax-ymin, zmax-zmin)
    box.translate( 0.5*vec3D(xmax+xmin, ymax+ymin, zmax+zmin) )
    return box    
    

if __name__ == '__main__': 
 
    b = Box3D(3.0, 1.0, 2.0)
    
    v = vec3D(1, 0.5, 0.3)
    mx = rotateMx(vec3D(1.3, 2.4, 0.2), 1.33)

    b.transform(mx)
    b.translate(v)

    plt.figure()
    b.plot('XY', color='r')      #b.plot('XZ')

    gbr = b.gabarits()
    
    testpts = testpoints(gbr, 10000)
    for pt in testpts: 
        if b.contains_pt(pt): 
            plot_point(pt, 'XY', ms=2)

    
    plt.figure()
    pgns = list( b.polygons() )

