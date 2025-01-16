# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:49:09 2023

@author: Krohalev_OD
"""
#%% imports
import numpy as np

from .geom import (pt3D, vec3D, identMx, invMx, normalizedVector,
                   outside_gabarits, vNorm, _regularPolygon3D, plot_polygon,
                   mxScaleCoeff)
from .box import Box3D
from .prim import AbstractCarcass3D, AbstractCollider3D
from .plot_ex import plot_polygon_ex

#%%
class CylindricShell(AbstractCollider3D):
    def __init__(self, height, radius):
        self._vec = pt3D(0, 0, 0)
        self._center_offset = 0.0
        self._mx = identMx()
        self._imx = identMx()
        self._height = height
        self._half_height = self._height/2.
        self.height = height
        self.normal = vec3D(1, 0, 0)
        self._radius = radius
        self.radius = radius
        self._main_box = Box3D(height, 2*radius, 2*radius)

    def gabarits(self):
        return self._main_box.gabarits()

    def recalc_gabarits(self):
        self._main_box.recalc_gabarits()

    @property
    def center(self):
        return self._vec

    @property
    def main_axis(self):
        return self.normal

    def scale_size(self, scale_coeff):
        self.radius *= scale_coeff
        self.height *= scale_coeff

    def transform(self, mx):
        self._vec = mx.dot(self._vec)
        self.normal = normalizedVector(mx.dot(self.normal))
        self.scale_size(mxScaleCoeff(mx))
        self._mx = mx.dot(self._mx)
        self._imx = invMx(self._mx)

    def translate(self, vec):
        self._vec += vec

    def polygons(self, npolygons, npoints, closed=True):
        pgs = []
        hh = np.linspace(-self.height/2, self.height/2, npolygons)
        for h in hh:
            pg = _regularPolygon3D(npoints, self.center + h*self.normal,
                                   self.radius, -self.normal, closed=closed)
            pgs.append(pg)
        return pgs

    def plot(self, axes_code=None, *args, **kwargs):
        pgs = self.polygons(10, 100, True)
        color = kwargs.get('color')
        if color is None:
            for pg in pgs:
                color = plot_polygon_ex(pg, axes_code=axes_code, color=color, **kwargs)[0].get_color()
        else:
            for pg in pgs:
                plot_polygon_ex(pg, axes_code=axes_code, **kwargs)

    def _complex_intersect_with_line(self, r0_loc, dr):
        x0, y0, z0 = r0_loc
        dx, dy, dz = dr

        # finding intersection of line and circle on plane
        a = dy**2 + dz**2
        b = 2*(y0*(dy) + z0*(dz))
        c = y0**2 + z0**2 - self._radius**2
        return np.roots([a, b, c])

    def intersect_with_segment(self, r0, r1):
        # transfer to local coords
        r0_loc = self._imx.dot(r0 - self._vec)
        r1_loc = self._imx.dot(r1 - self._vec)
        r0_center_offset = r0_loc[0] - self._center_offset
        r1_center_offset = r1_loc[0] - self._center_offset
        if ((abs(r0_center_offset) > self._half_height) and
            (abs(r1_center_offset) > self._half_height) and
            r0_center_offset*r1_center_offset > 0):
            return None

        dr = r1_loc - r0_loc

        t = self._complex_intersect_with_line(r0_loc, dr)

        if len(t) == 0:
            return None
        t1, t2 = t

        # choosing 1 solution, which is real and lies in [0, 1]
        type_t1 = isinstance(t1, float)
        if not type_t1: # no intersection - both t are complex
            return None
        else:       # both t are real
            t1_out_of_seg = ((t1 < 0.) or (t1 > 1.))
            t2_out_of_seg = ((t2 < 0.) or (t2 > 1.))

            if t1_out_of_seg and t2_out_of_seg:
                return None
            if t1_out_of_seg:
                t = t2
                ri = r0_loc + t*dr
                if (abs(ri[0] - self._center_offset) > self._half_height):
                    return None
            elif t2_out_of_seg:
                t = t1
                ri = r0_loc + t*dr
                if (abs(ri[0] - self._center_offset) > self._half_height):
                    return None
            else:
                t = min(t1, t2)
                ri = r0_loc + t*dr
                if (abs(ri[0] - self._center_offset) > self._half_height):
                    t = max(t1, t2)
                    ri = r0_loc + t*dr
                    if (abs(ri[0] - self._center_offset) > self._half_height):
                        return None

        return self._mx.dot(ri) + self._vec

#%%
class CuttedConeShell(CylindricShell):
    def __init__(self, height, small_radius, big_radius):
        self._vec = vec3D(0, 0, 0)
        self.normal = vec3D(1, 0, 0)
        self._mx = identMx()
        self._imx = identMx()
        self.big_radius = big_radius
        self.small_radius = small_radius
        self._height = height
        self._half_height = self._height/2.
        self.height = height
        self._a = self._height/(self.big_radius - self.small_radius)
        self._center_offset = self._a*self.small_radius + self._half_height
        self._cone_coeff = 1.0/(self._a)**2
        self.translate(-self.center)
        self._main_box = Box3D(height, 2*big_radius, 2*big_radius)

    @property
    def center(self):
        return self._vec + (self._a*self.small_radius + self.height/2)*self.normal

    def scale_size(self, scale_coeff):
        self.small_radius *= scale_coeff
        self.big_radius *= scale_coeff
        self.height *= scale_coeff

    def polygons(self, npolygons, npoints, closed=True):
        pgs = []
        hh = np.linspace(-self.height/2, self.height/2, npolygons)
        for h in hh:
            pg = _regularPolygon3D(npoints, self.center + h*self.normal,
                                   (self.small_radius + self.big_radius)/2 + h/self._a,
                                   -self.normal, closed=closed)
            pgs.append(pg)
        return pgs

    def _complex_intersect_with_line(self, r0_loc, dr):
        x0, y0, z0 = r0_loc
        dx, dy, dz = dr

        # finding intersection of line and cone
        a = dy**2 + dz**2 - self._cone_coeff*dx**2
        b = 2.0*(y0*dy + z0*dz - self._cone_coeff*x0*dx)
        c = y0**2 + z0**2 - self._cone_coeff*x0**2
        return np.roots([a, b, c])

#%%
class Cylinder(AbstractCarcass3D):
    def __init__(self, height, radius):
        self.radius = radius
        self.height = height
        self.half_height = self.height/2
        self.normal = vec3D(1, 0, 0)
        self._vec = pt3D(0, 0, 0)
        self._mx = identMx()
        self._imx = identMx()
        self._main_box = Box3D(height, 2*radius, 2*radius)

    def gabarits(self):
        return self._main_box.gabarits()

    def recalc_gabarits(self):
        self._main_box.recalc_gabarits()

    @property
    def center(self):
        return self._vec

    @property
    def main_axis(self):
        return self.normal

    def scale_size(self, scale_coeff):
        self.radius *= scale_coeff
        self.height *= scale_coeff
        self.half_height *= scale_coeff

    def transform(self, mx):
        self.normal = normalizedVector(mx.dot(self.normal))
        self.scale_size(mxScaleCoeff(mx))
        self._vec = mx.dot(self._vec)
        self._mx = mx.dot(self._mx)
        self._imx = invMx(self._mx)
        self._main_box.transform(mx)

    def translate(self, vec):
        self._vec += vec
        self._main_box.translate(vec)

    def pt_in_circle(self, pt_radius_v, pt_z):
        return (vNorm(pt_radius_v) <= self.radius)

    def contains_pt(self, pt):
        if outside_gabarits(pt, self.gabarits()):
            return False

        r_loc = pt - self._vec
        r_ver = r_loc.dot(self.main_axis)
        if abs(r_ver) > self.half_height:
            return False

        r_hor = r_loc - r_ver*self.main_axis
        if self.pt_in_circle(r_hor, r_ver):
            return True
        return False

    def polygons(self, n_polygons_width, n_polygons_height, npoints, closed=True):
        pgs = []
        rr = np.linspace(0.1*self.radius, self.radius, n_polygons_width)
        for r in rr:
            for sign in [-1, 1]:
                pg = _regularPolygon3D(npoints, self.center +
                                       sign*self.main_axis*self.half_height, r,
                                       self.main_axis, closed=closed)
                pgs.append(pg)
        hh = np.linspace(-self.half_height, self.half_height, n_polygons_height)
        for h in hh:
            pg = _regularPolygon3D(npoints, self.center + self.main_axis*h,
                                   self.radius, self.main_axis, closed=closed)
            pgs.append(pg)
        return pgs

    def plot(self, axes_code=None, *args, **kwargs):
        pgs = self.polygons(10, 10, 100, True)
        color = kwargs.get('color')
        if color is None:
            for pg in pgs:
                color = plot_polygon_ex(pg, axes_code=axes_code, color=color, **kwargs)[0].get_color()
        else:
            for pg in pgs:
                plot_polygon_ex(pg, axes_code=axes_code, **kwargs)

#%%
class HollowCylinder(Cylinder):
    def __init__(self, height, in_radius, out_radius):
        self.in_radius = in_radius
        self.out_radius = out_radius
        self.height = height
        self.half_height = self.height/2
        self.normal = vec3D(1, 0, 0)
        self._vec = pt3D(0, 0, 0)
        self._mx = identMx()
        self._imx = identMx()
        self._main_box = Box3D(height, 2*out_radius, 2*out_radius)

    def scale_size(self, scale_coeff):
        self.in_radius *= scale_coeff
        self.out_radius *= scale_coeff
        self.height *= scale_coeff
        self.half_height *= scale_coeff

    def pt_in_circle(self, pt_radius_v, pt_z):
        pt_radius_norm = vNorm(pt_radius_v)
        return ((pt_radius_norm <= self.out_radius) and
                (pt_radius_norm >= self.in_radius))

    def polygons(self, n_polygons_width, n_polygons_height, npoints, closed=True):
        pgs = []
        rr = np.linspace(self.in_radius, self.out_radius, n_polygons_width)
        for r in rr:
            for sign in [-1, 1]:
                pg = _regularPolygon3D(npoints, self.center +
                                       sign*self.main_axis*self.half_height, r,
                                       self.main_axis, closed=closed)
                pgs.append(pg)
        hh = np.linspace(-self.half_height, self.half_height, n_polygons_height)
        for h in hh:
            for r in [self.in_radius, self.out_radius]:
                pg = _regularPolygon3D(npoints, self.center + self.main_axis*h,
                                       r, self.main_axis, closed=closed)
                pgs.append(pg)
        return pgs

#%%
class CuttedCone(Cylinder):
    def __init__(self, height, small_radius, big_radius):
        self._vec = vec3D(0, 0, 0)
        self.normal = vec3D(1, 0, 0)
        self._mx = identMx()
        self._imx = identMx()
        self.big_radius = big_radius
        self.small_radius = small_radius
        self.radius = (self.big_radius + self.small_radius)/2
        self._height = height
        self.height = height
        self.half_height = self.height/2
        self._a = self._height/(self.big_radius - self.small_radius)
        self._a_rev = (self.big_radius - self.small_radius)/self._height
        self._main_box = Box3D(height, 2*big_radius, 2*big_radius)

    def scale_size(self, scale_coeff):
        self.small_radius *= scale_coeff
        self.big_radius *= scale_coeff
        self.radius *= scale_coeff
        self.height *= scale_coeff
        self.half_height *= scale_coeff

    def pt_in_circle(self, pt_radius_v, pt_z):
        radius_loc = self.radius + self._a_rev*pt_z
        return (vNorm(pt_radius_v) < radius_loc)

    def polygons(self, n_polygons_width, n_polygons_height, npoints, closed=True):
        pgs = []
        for r_lim, sign in zip([self.small_radius, self.big_radius], [-1, 1]):
            rr = np.linspace(r_lim*0.1, r_lim, n_polygons_width)
            for r in rr:
                pg = _regularPolygon3D(npoints, self.center +
                                       sign*self.main_axis*self.height/2, r,
                                       self.main_axis, closed=closed)
                pgs.append(pg)
        hh = np.linspace(-self.height/2, self.height/2, n_polygons_height)
        for h in hh:
            pg = _regularPolygon3D(npoints, self.center + h*self.normal,
                                   self.radius + h/self._a,
                                   -self.normal, closed=closed)
            pgs.append(pg)
        return pgs

#%%
class HollowCuttedCone(Cylinder):
    def __init__(self, height, out_small_radius, out_big_radius, thick):
        self._vec = vec3D(0, 0, 0)
        self.normal = vec3D(1, 0, 0)
        self._mx = identMx()
        self._imx = identMx()
        self.out_big_radius = out_big_radius
        self.out_small_radius = out_small_radius
        self.radius = (self.out_big_radius + self.out_small_radius)/2
        self.thick = thick
        self._height = height
        self.height = height
        self.half_height = self.height/2
        self._a = self._height/(self.out_big_radius - self.out_small_radius)
        self._a_rev = (self.out_big_radius - self.out_small_radius)/self._height
        self._main_box = Box3D(height, 2*out_big_radius, 2*out_big_radius)

    def scale_size(self, scale_coeff):
        self.out_small_radius *= scale_coeff
        self.out_big_radius *= scale_coeff
        self.radius *= scale_coeff
        self.height *= scale_coeff
        self.half_height *= scale_coeff
        self.thick *= scale_coeff

    def pt_in_circle(self, pt_radius_v, pt_z):
        pt_radius_norm = vNorm(pt_radius_v)
        radius_loc_out = self.radius + self._a_rev*pt_z
        radius_loc_in = radius_loc_out - self.thick
        return ((pt_radius_norm < radius_loc_out) and
                (pt_radius_norm > radius_loc_in))

    def polygons(self, n_polygons_width, n_polygons_height, npoints, closed=True):
        pgs = []
        for r_lim, sign in zip([self.out_small_radius, self.out_big_radius], [-1, 1]):
            rr = np.linspace(r_lim - self.thick, r_lim, n_polygons_width)
            for r in rr:
                pg = _regularPolygon3D(npoints, self.center +
                                       sign*self.main_axis*self.height/2, r,
                                       self.main_axis, closed=closed)
                pgs.append(pg)
        hh = np.linspace(-self.height/2, self.height/2, n_polygons_height)
        for h in hh:
            for i in [0, 1]:
                pg = _regularPolygon3D(npoints, self.center + h*self.normal,
                                       self.radius + h/self._a - i*self.thick,
                                       -self.normal, closed=closed)
                pgs.append(pg)
        return pgs
