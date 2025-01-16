# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:20:28 2024

@author: Krohalev_OD
"""
#%%
from copy import deepcopy
import numpy as np

from .prim import AbstractCarcass3D, as_polygon3D, Polygon3D
from .geom import main_basis, join_gabarits, _ptInPolygon3D_, outside_gabarits, normalizedVector

#%%
class Prism3D(AbstractCarcass3D):
    def __init__(self, polygon, height, basis=main_basis, main_axis=main_basis[-1]):
        '''
        Parameters
        ----------
        polygon : list or array of shape (n, 3) or (n, 2). Or Polygon3D, Polygon2D
            Base polygon of the prism
        height : float
            along Z axis. If you want prism to be oriented differently, you should give new basis
        basis : np.array of shape (3, 3)
            basis, in which polygon will lay in XY plane.
            The default is array([vec3D(1, 0, 0), vec3D(0, 1, 0), vec3D(0, 0, 1)])

        '''

        self.main_axis = normalizedVector(deepcopy(main_axis))

        self.polygons = []
        self.polygons.append(as_polygon3D(polygon, basis))

        pts = []
        for pt in self.polygons[0]._points:
            pts.append(pt + self.main_axis*height)
        self.polygons.append(Polygon3D(pts))

        self._gabarits = join_gabarits(self.polygons[0].gabarits(), self.polygons[1].gabarits())

        pts_dn = self.polygons[0]._points
        pts_up = self.polygons[1]._points
        for pt1, pt2, pt4, pt3 in zip(pts_dn[:-1], pts_dn[1:], pts_up[:-1], pts_up[1:]):
            points = np.asarray([pt1, pt2, pt3, pt4]).copy()
            self.polygons.append(Polygon3D(points))

        self.height = height

    def gabarits(self):
        return self._gabarits

    def recalc_gabarits(self):
        self._gabarits = join_gabarits(self.polygons[0].gabarits(), self.polygons[1].gabarits())

    def contains_pt(self, pt):
        if outside_gabarits(pt, self.gabarits()):
            return False

        normal = self.main_axis
        center = self.polygons[0]._center
        pt_loc = pt - center

        r_ver = pt_loc.dot(normal)
        # print(pt, r_ver)
        if r_ver > self.height or r_ver < 0.:
            return False

        pt_loc -= normal*r_ver
        if _ptInPolygon3D_(self.polygons[0]._points  - center, pt_loc):
            return True

        return False

    def plot(self, *args, **kwargs):
        for poly in self.polygons:
            poly.plot(*args, **kwargs)

    def transform(self, mx):
        for poly in self.polygons:
            poly.transform(mx)
        self.recalc_gabarits()
        self.main_axis = normalizedVector(mx.dot(self.main_axis))

    def translate(self, vec):
        for poly in self.polygons:
            poly.translate(vec)
        self.recalc_gabarits()

    def intersect_with_segment(self, r0, r1):
        for poly in self.polygons:
            pt = poly.intersect_with_segment(r0, r1)
            if pt is not None:
                return pt
