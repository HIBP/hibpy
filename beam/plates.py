# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:48:17 2023

@author: Eliseev_LG, Krohalev_OD

class AbstractPlates
    E(r)
    transform(mx)
    translate(vec)
    U
    aim
    intersect_with_segment(r0, r1)
    contains_pt(pt)
    masks_with_voltages(grid)
"""
#%% imports
from abc import abstractmethod, abstractproperty
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from ..geom.box import gabaritbox3D, Box3D, outside_gabarits, gabaritbox3D
from ..geom.prim import (Polygon3D, PolygonWithHole3D, CarcassWithHole,
                         ColliderCutWithCarcass, as_polygon3D)
from ..geom.group import Group3D
from ..geom.groupstrat import PlusStrategy
from ..geom.geom import (vec3D, pt3D, zRotateMx, sin, cos, invMx,
                         plot_polygon, normalizedVector, join_gabarits,
                         xRotateMx)
from ..geom.prism import PolygonalPrism3D
from ..misc.grid import fast_scalar_field_on_grid, _grid_to_raw_points
from ..misc.interp import EmptyVectorInterpolator3D
from .electrode import AbstractElectrode
from .aim import PolygonCenterAim

#%%
class AbstractPlates(AbstractElectrode):
    @abstractmethod
    def E(self, r):
        """
        Parameters
        ----------
        r : np.array of floats with shape (3,)

        Returns
        -------
        E : np.array of floats with shape (3,)
            electric field in V/m
        """

    @abstractmethod
    def aim(self):
        """
        Returns
        -------
        aim : Aim
        """

    @property
    def base_U(self):
        return self._base_U

    @base_U.setter
    def base_U(self, U):
        self._base_U = U

    @abstractmethod
    def masks_with_voltages(self, grid):
        """
        Parameters
        ----------
        grid : np.mgrid

        Returns
        -------
        umask, lmask : np.array of bool with shape grid.shape[1:] #!!!

        """

    @abstractproperty
    def center(self):
        """
        Returns
        -------
        center : np.array of float with shape (3,)
        """

    @abstractproperty
    def crate(self):
        """
        Returns
        -------
        crate : any obj with carcass and collider
        """

    @property
    def gabarit_box(self):
        return self._gabarit_box

    @gabarit_box.setter
    def gabarit_box(self, box):
        self._gabarit_box = box

    @property
    def transformable(self):
        if hasattr(self, '_transformable'):
            return self._transformable
        self._transformable = Group3D([self.main_box, self.gabarit_box, self.collider,
                                       self.carcass])
        return self._transformable

    def set_voltage(self, U, kind='main'):
        if kind == 'main':
            self.U = U

        elif kind == 'gnd':
            self.base_U = U

        else:
            raise KeyError(f'kind must be main or gnd, {kind} given')

#%%
class FlatPlates(AbstractPlates):
    def __init__(self, width, length, gap, thick, name=None):
        self.width  = width
        self.length = length
        self.gap    = gap
        self.thick  = thick
        self.name = name
        self._main_box = Box3D(2*self.length, 2*self.gap, 2*self.width)
        self._gabarit_box = deepcopy(self._main_box)
        self._crate = None
        self._carcass  = self.carcass
        self._collider = self.collider
        self.crate = EmptyCrate()
        self._U = 0.
        self._base_U = 0.
        self._interp = None
        self._base_interp = EmptyVectorInterpolator3D()
        self._domain_box = None
        self._transformable = Group3D([self._main_box, self._gabarit_box, self._collider,
                                       self._carcass])

    def _decoy(self):
        return DecoyFlatPlates(self.name)

    @property
    def main_box(self):
        return self._main_box

    def gabarits(self):
        return self._gabarit_box.gabarits()

    def resize_gabarit_box(self):
        mx = deepcopy(self.main_box._mx)
        imx = invMx(mx)
        vec = deepcopy(self.main_box._vec)

        self.transformable.translate(-vec)
        self.transformable.transform(imx)

        #self._gabarit_box = gabaritbox3D(self.collider.gabarits())
        self._gabarit_box = gabaritbox3D(self.carcass.gabarits()) # !!! 11.12.2024 R&Y
        self.transformable[1] = self._gabarit_box

        self.transformable.transform(mx)
        self.transformable.translate(vec)

    @property
    def crate(self):
        return self._crate

    @crate.setter
    def crate(self, crate):
        crate.transform(self.main_box._mx)
        crate.translate(self.main_box._vec)
        self._crate = crate
        self._carcass[-1] = deepcopy(self._crate)
        self._collider[-1] = deepcopy(self._crate)
        self.resize_gabarit_box()

    def masks_with_voltages(self, grid):
        lower, upper, crate = self.carcass

        umask = fast_scalar_field_on_grid(grid, upper.contains_pt, gabarits=upper.gabarits(), outvalue=False)
        lmask = fast_scalar_field_on_grid(grid, lower.contains_pt, gabarits=lower.gabarits(), outvalue=False)
        cmask = fast_scalar_field_on_grid(grid, crate.contains_pt, gabarits=crate.gabarits(), outvalue=False)

        return [[lmask, self.base_U], [umask, self.U], [cmask, 0.0]]

    @property
    def carcass(self):
        if hasattr(self, "_carcass"):
            return self._carcass
        vec = vec3D(0, (self.gap + self.thick)/2, 0)
        upper = Box3D(self.length, self.thick, self.width, eps=1e-8)
        upper.translate(vec)
        lower = Box3D(self.length, self.thick, self.width, eps=1e-8)
        lower.translate(-vec)
        self._carcass = Group3D([lower, upper, self.crate])
        return self._carcass

    @property
    def collider(self):
        if hasattr(self, "_collider"):
            return self._collider
        points = [pt3D(-self.length/2, 0, -self.width/2),
                  pt3D(-self.length/2, 0,  self.width/2),
                  pt3D( self.length/2, 0,  self.width/2),
                  pt3D( self.length/2, 0, -self.width/2)]

        vec = vec3D(0, self.gap/2, 0)
        upper = Polygon3D(points + vec)
        lower = Polygon3D(points - vec)
        self._collider = Group3D([lower, upper, self.crate])
        return self._collider

    @property
    def center(self):
        return np.mean(self._main_box.points(), axis=0)

    def E(self, r):
        if outside_gabarits(r, self._domain_box.gabarits()):
            return np.zeros((3,))
        r_loc = self.main_box._imx.dot(r - self.main_box._vec)
        E = self._interp(r_loc)*self.U + self._base_interp(r_loc)*self.base_U
        return self.main_box._mx.dot(E)

    def set_E_interp(self, E_interp):
        self._interp = E_interp
        self._domain_box = gabaritbox3D((E_interp.lower_corner, E_interp.upper_corner))
        self._domain_box.transform(self._main_box._mx)
        self._domain_box.translate(self._main_box._vec)
        self._transformable.append(self._domain_box)
        self.test_E()

    def set_base_interp(self, E_interp):
        self._base_interp = E_interp

    def test_E(self):
        pass #???

    def aim(self, eps_main=None, eps_yaw=None, **kwargs):
        if hasattr(self, '_aim'):
            return self._aim
        pts = [self.collider[0].points()[0],
               self.collider[0].points()[1],
               self.collider[1].points()[1],
               self.collider[1].points()[0]]
        main_vec = normalizedVector(pts[2] - pts[1])
        self._aim = PolygonCenterAim(pts, main_vec, eps_pitch=eps_main, eps_yaw=eps_yaw)
        self._transformable.append(self._aim)
        return self._aim

class DecoyFlatPlates(FlatPlates):
    def __init__(self, name):
        self.name = name

#%%
class FlaredPlates(FlatPlates):
    def __init__(self, width, length, gap, thick, flared_length, swip_angle, name=None):
        self.width  = width
        self.length = length
        self.gap    = gap
        self.thick  = thick
        self.flared_length = flared_length
        self.swip_angle = swip_angle
        self.name = name
        self._main_box = Box3D(2*(self.length + self.flared_length*cos(self.swip_angle/2)),
                               2*(self.gap + self.flared_length*sin(self.swip_angle/2)),
                               2*self.width)
        self._gabarit_box = deepcopy(self._main_box)
        self._crate = None
        self._carcass  = self.carcass
        self._collider = self.collider
        self.crate = EmptyCrate()
        self._U = 0.
        self._base_U = 0.
        self._interp = None
        self._base_interp = EmptyVectorInterpolator3D()
        self._domain_box = None
        self._transformable = Group3D([self._main_box, self._gabarit_box, self._collider,
                                       self._carcass])

    def _decoy(self):
        return DecoyFlaredPlates(self.name)

    @property
    def carcass(self):
        if hasattr(self, "_carcass"):
            return self._carcass
        vec_up = vec3D(0, (self.gap + self.thick)/2, 0)
        mx = zRotateMx(-self.swip_angle/2)
        upper_h = Box3D(self.length, self.thick, self.width, eps=1e-8)
        upper_h.translate(vec_up)
        upper_s = Box3D(self.flared_length, self.thick, self.width, eps=1e-8)
        upper_s.transform(mx)
        upper_s.translate(upper_h.points()[2] - upper_s.points()[3])
        upper = Group3D([upper_h, upper_s])

        mx = zRotateMx(self.swip_angle/2)
        lower_h = Box3D(self.length, self.thick, self.width, eps=1e-8)
        lower_h.translate(-vec_up)
        lower_s = Box3D(self.flared_length, self.thick, self.width, eps=1e-8)
        lower_s.transform(mx)
        lower_s.translate(lower_h.points()[1] - lower_s.points()[0])
        lower = Group3D([lower_h, lower_s])

        vec = vec3D(self.flared_length*cos(self.swip_angle/2)/2, 0, 0)
        upper.translate(vec)
        lower.translate(vec)

        self._carcass = Group3D([lower, upper, self.crate])

        return self._carcass

    @property
    def collider(self):
        if hasattr(self, "_collider"):
            return self._collider
        pts = self.carcass[1][0].points()
        upper_h = Polygon3D([pts[2], pts[6], pts[7], pts[3]])
        pts = self.carcass[1][1].points()
        upper_s = Polygon3D([pts[2], pts[6], pts[7], pts[3]])
        upper = Group3D([upper_h, upper_s])

        pts = self.carcass[0][0].points()
        lower_h = Polygon3D([pts[1], pts[5], pts[4], pts[0]])
        pts = self.carcass[0][1].points()
        lower_s = Polygon3D([pts[1], pts[5], pts[4], pts[0]])
        lower = Group3D([lower_h, lower_s])

        self._collider = Group3D([lower, upper, self.crate])
        return self._collider

    def aim(self, eps_main, eps_yaw, **kwargs):
        pass # ???

class DecoyFlaredPlates(FlaredPlates):
    def __init__(self, name):
        self.name = name

#%%
class EmptyCrate():
    def __init__(self):
        self._vec = vec3D(0, 0, 0)

    def transform(self, mx):
        self._vec = mx.dot(self._vec)

    def translate(self, vec):
        self._vec += vec

    def contains_pt(self, pt):
        return False

    def intersect_with_segment(self, r0, r1):
        return None

    def plot(self, axes_code='XY', **kwargs):
        pass

    def gabarits(self):
        return np.array([self._vec, self._vec])

    @property
    def carcass(self):
        return None

#%%
class ComplexCrate(AbstractElectrode):
    def __init__(self, carcass, collider):
        self._carcass = carcass
        self._collider = collider
        self._main_box = gabaritbox3D(collider.gabarits()) # ???
        self._transformable = Group3D([self._main_box, self._collider,
                                       self._carcass])

    @property
    def main_box(self):
        return self._main_box

    @property
    def carcass(self):
        return self._carcass

    @property
    def collider(self):
        return self._collider

#%%
class RectangleCrate(AbstractElectrode):
    def __init__(self, length, in_width, in_height, thick):
        self.length = length
        self.thick = thick
        self.in_width = in_width
        self.in_height = in_height
        self._main_box = Box3D(2*self.length, 2*(self.in_width + 2*self.thick),
                               2*(self.in_height + 2*self.thick))
        self._carcass = self.carcass
        self._collider = self.collider
        self._transformable = Group3D([self._main_box, self._collider,
                                              self._carcass])

    @property
    def main_box(self):
        return self._main_box

    @property
    def carcass(self):
        if hasattr(self, "_carcass"):
            return self._carcass

        horizontal = Box3D(self.length, self.thick, self.in_width + 2*self.thick)
        up = deepcopy(horizontal).translate(vec3D(0,  (self.in_height + self.thick)/2, 0))
        dn = deepcopy(horizontal).translate(vec3D(0, -(self.in_height + self.thick)/2, 0))

        vertical   = Box3D(self.length, self.in_height + 2*self.thick, self.thick)
        left  = deepcopy(vertical).translate(vec3D(0, 0, -(self.in_width + self.thick)/2))
        right = deepcopy(vertical).translate(vec3D(0, 0,  (self.in_width + self.thick)/2))

        self._carcass = Group3D([up, dn, left, right])
        return self._carcass

    @property
    def collider(self):
        if hasattr(self, "_collider"):
            return self._collider

        points = [pt3D(-self.length/2, 0, -self.in_width/2),
                  pt3D(-self.length/2, 0,  self.in_width/2),
                  pt3D( self.length/2, 0,  self.in_width/2),
                  pt3D( self.length/2, 0, -self.in_width/2)]
        vec = vec3D(0, self.in_height/2, 0)
        up = Polygon3D(points + vec)
        dn = Polygon3D(points - vec)

        points = [pt3D(-self.length/2, -self.in_height/2, 0),
                  pt3D(-self.length/2,  self.in_height/2, 0),
                  pt3D( self.length/2,  self.in_height/2, 0),
                  pt3D( self.length/2, -self.in_height/2, 0)]
        vec = vec3D(0, 0, self.in_width/2)
        left  = Polygon3D(points + vec)
        right = Polygon3D(points - vec)

        self._collider = Group3D([up, dn, left, right])
        return self._collider

#%%
class RectangleDiafragm(AbstractElectrode):
    def __init__(self, thick, width, height, ap_width, ap_height, ap_off_w,
                 ap_off_h):
        self.thick = thick
        self.width = width
        self.height = height
        self.ap_width = ap_width
        self.ap_height = ap_height
        self.ap_off_w = ap_off_w
        self.ap_off_h = ap_off_h

        self._main_box = Box3D(2*self.thick, 2*self.height, 2*self.width)
        self._carcass = self.carcass
        self._collider = self.collider
        self._transformable = Group3D([self._collider, self._carcass,
                                       self._main_box])

    @property
    def main_box(self):
        return self._main_box

    @property
    def carcass(self):
        if hasattr(self, "_carcass"):
            return self._carcass

        box  = Box3D(self.thick, self.height, self.width)
        hole = Box3D(self.thick, self.ap_height, self.ap_width)
        hole.translate(vec3D(0, self.ap_off_h, self.ap_off_w))
        self._carcass = CarcassWithHole(box, hole)
        return self._carcass

    @property
    def collider(self):
        if hasattr(self, "_collider"):
            return self._collider

        points_h   = [pt3D(-self.thick/2, 0, -self.ap_width/2),
                      pt3D(-self.thick/2, 0,  self.ap_width/2),
                      pt3D( self.thick/2, 0,  self.ap_width/2),
                      pt3D( self.thick/2, 0, -self.ap_width/2)]

        points_v   = [pt3D(-self.thick/2, -self.ap_height/2, 0),
                      pt3D(-self.thick/2,  self.ap_height/2, 0),
                      pt3D( self.thick/2,  self.ap_height/2, 0),
                      pt3D( self.thick/2, -self.ap_height/2, 0)]

        points_in  = [pt3D(0, -self.ap_height/2, -self.ap_width/2),
                      pt3D(0,  self.ap_height/2, -self.ap_width/2),
                      pt3D(0,  self.ap_height/2,  self.ap_width/2),
                      pt3D(0, -self.ap_height/2,  self.ap_width/2)]

        points_out = [pt3D(0, -self.height/2, -self.width/2),
                      pt3D(0,  self.height/2, -self.width/2),
                      pt3D(0,  self.height/2,  self.width/2),
                      pt3D(0, -self.height/2,  self.width/2)]

        vec_h   = vec3D(0, self.ap_height/2, 0)
        vec_v   = vec3D(0, 0, self.ap_width/2)
        vec_f   = vec3D(self.thick/2, 0, 0)
        vec_off = vec3D(0, self.ap_off_h, self.ap_off_w)

        up    = Polygon3D(points_h + vec_h + vec_off)
        dn    = Polygon3D(points_h - vec_h + vec_off)
        left  = Polygon3D(points_v + vec_v + vec_off)
        right = Polygon3D(points_v - vec_v + vec_off)

        front = PolygonWithHole3D(points_out + vec_f, points_in + vec_f + vec_off)
        back  = PolygonWithHole3D(points_out - vec_f, points_in - vec_f + vec_off)

        self._collider = Group3D([up, dn, left, right, front, back])
        return self._collider

#%%
def crate_cut_with_carcass(crate, cut_carcass):
    mx  = deepcopy(crate.main_box._mx)
    imx = invMx(mx)
    vec = deepcopy(crate.main_box._vec)

    # move to crate coord sys to save info about transformations
    crate.translate(-vec)
    crate.transform(imx)
    cut_carcass.translate(-vec)
    cut_carcass.transform(imx)

    # get data
    carcass  = deepcopy(crate)
    collider = deepcopy(crate)
    hole     = deepcopy(cut_carcass)

    # put inital obj back on place
    crate.transform(mx)
    crate.translate(vec)
    cut_carcass.transform(mx)
    cut_carcass.translate(vec)

    # cut new crate parts
    carcass  = CarcassWithHole(carcass, deepcopy(hole))
    collider = ColliderCutWithCarcass(collider, deepcopy(hole))

    # create new crate and put it on place
    new_crate = ComplexCrate(carcass, collider)
    new_crate.transform(mx)
    new_crate.translate(vec)

    return new_crate

#%%
class PolygonalPlates(FlatPlates):
    def __init__(self, lower_poly2D, upper_poly2D, gap, thick, name=None):

        self.upper_poly = as_polygon3D(upper_poly2D)
        self.lower_poly = as_polygon3D(lower_poly2D)
        self.upper_poly.transform(xRotateMx(np.pi/2))
        self.lower_poly.transform(xRotateMx(np.pi/2))
        self.upper_poly.translate( vec3D(0, 1, 0)*gap*0.5)
        self.lower_poly.translate(-vec3D(0, 1, 0)*gap*0.5)

        self.gap    = gap
        self.thick  = thick
        self.name = name

        box = gabaritbox3D(join_gabarits(self.upper_poly.gabarits(), self.lower_poly.gabarits()))
        box._vec = vec3D(0, 0, 0)
        self._gabarit_box = deepcopy(box)

        box = gabaritbox3D(join_gabarits(self.upper_poly.gabarits(), self.lower_poly.gabarits()))
        vec = box._vec.copy()
        box.translate(-vec)
        box._points = list(np.asarray(box._points)*2.)
        box.translate(vec)
        self._main_box = deepcopy(box)

        self._crate = None
        self._carcass  = self.carcass
        self._collider = self.collider
        self.crate = EmptyCrate()

        self._U = 0.
        self._base_U = 0.

        self._interp = None
        self._base_interp = EmptyVectorInterpolator3D()
        self._domain_box = None

        self._transformable = Group3D([self._main_box, self._gabarit_box, self._collider,
                                       self._carcass])

    def _decoy(self):
        return DecoyPolygonalPlates(self.name)

    @property
    def carcass(self):
        if hasattr(self, '_carcass'):
            return self._carcass

        lower = PolygonalPrism3D(deepcopy(self.lower_poly), self.thick, main_axis=vec3D(0, -1, 0))
        upper = PolygonalPrism3D(deepcopy(self.upper_poly), self.thick, main_axis=vec3D(0,  1, 0))
        self._carcass = Group3D([lower, upper, self.crate])
        return self._carcass

    @property
    def collider(self):
        if hasattr(self, "_collider"):
            return self._collider

        self._collider = Group3D([self.lower_poly, self.upper_poly, self.crate])
        return self._collider

class DecoyPolygonalPlates(PolygonalPlates):
    def __init__(self, name):
        self.name = name

#%%
Group3D.call_strategies['E'] = PlusStrategy()
