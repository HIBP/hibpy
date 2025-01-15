# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:16:02 2023

@author: Krohalev_OD
"""
#%% imports
from abc import abstractproperty, abstractmethod
from copy import deepcopy
import numpy as np

from ..misc.grid import fast_scalar_field_on_grid
from ..geom.group import Group3D, Group2D
from ..geom.groupstrat import ListStrategy, JustCallStrategy
from ..geom.prim import (AbstractCollider3D, AbstractCarcass3D, HollowCircle3D,
                         Polygon2D, CarcassWithHole, ColliderCutWithCarcass)
from ..geom.geom import outside_gabarits, pt3D, vec3D, identMx2D, mx2Dto3D
from ..geom.box import Box3D
from ..geom.history import History
from ..geom.cylinder import (HollowCylinder, HollowCuttedCone, CuttedConeShell,
                             CylindricShell, CuttedCone)

#%%
class AbstractElectrode(AbstractCollider3D, AbstractCarcass3D):
    @property
    def history(self):
        if hasattr(self, "_history"):
            return self._history
        else:
            self._history = History()
            return self._history

    @property
    def U(self):
        if hasattr(self, "_U"):
            return self._U
        return 0.0

    @U.setter
    def U(self, U):
        self._U = U

    def set_U(self, U):
        self.U = U

    @abstractproperty
    def main_box(self):
        """

        Returns
        -------
        self._main_box : Box3D
            Box with center in center of obj

        """

    def gabarits(self):
        return self.main_box.gabarits()

    def recalc_gabarits(self):
        self.main_box.recalc_gabarits()

    @abstractproperty
    def carcass(self):
        """
        carcass must contain 1 obj with methods: transform, translate,
        contains_pt.

        Returns
        -------
        carcass : Group3D

        """

    def contains_pt(self, pt):
        if outside_gabarits(pt, self.gabarits()):
            return False
        return self.carcass.contains_pt(pt)

    def masks_with_voltages(self, grid):
        """
        Parameters
        ----------
        grid : np.mgrid

        Returns
        -------
        [[mask, voltage]]
        mask : np.array of bool with shape grid.shape[1:] #!!!
        voltage : float

        """
        mask = fast_scalar_field_on_grid(grid, self.carcass.contains_pt, gabarits=self.gabarits(), outvalue=False)
        return [[mask, self.U]]

    @abstractproperty
    def collider(self):
        """
        collider must contain 1 obj with methods: transform, translate,
        intersect_with_segment, plot.

        Returns
        -------
        collider : Group3D

        """

    def plot(self, axes_code='XY', **kwargs):
        self.collider.plot(axes_code=axes_code, **kwargs)

    def intersect_with_segment(self, r0, r1):
        if outside_gabarits(r0, self.gabarits()) and outside_gabarits(r1, self.gabarits()):
            return None
        return self.collider.intersect_with_segment(r0, r1)

    @property
    def transformable(self):
        if hasattr(self, '_transformable'):
            return self._transformable
        self._transformable = Group3D([self.collider, self.carcass,
                                       self.main_box, self.history])
        return self._transformable

    def transform(self, mx):
        self.transformable.transform(mx)

    def translate(self, vec):
        self.transformable.translate(vec)

    def make_hole(self, hole):
        self._carcass = CarcassWithHole(deepcopy(self._carcass), deepcopy(hole))
        self._collider = ColliderCutWithCarcass(deepcopy(self._collider), deepcopy(hole))
        if hasattr(self, '_transformable'):
            del(self._transformable)

#%%
class CylindricElectrode(AbstractElectrode):
    def __init__(self, height, out_radius, thick, name=None):
        self.height = height
        self.thick = thick
        self.in_radius = out_radius - thick
        self.out_radius = out_radius
        self._carcass = HollowCylinder(self.height, self.in_radius, self.out_radius)
        self._collider = self.collider
        self._main_box = Box3D(height + 0.01, 2*out_radius + 0.02, 2*out_radius + 0.02)
        self.name = name

    @property
    def main_box(self):
        return self._main_box

    @property
    def carcass(self):
        return self._carcass

    @property
    def collider(self):
        if hasattr(self, '_collider'):
            return self._collider
        in_cylinder = CylindricShell(self.height, self.in_radius)
        out_cylinder = CylindricShell(self.height, self.out_radius)
        up_lid = HollowCircle3D(pt3D( self.height/2, 0, 0), self.in_radius,
                                self.out_radius, vec3D(1, 0, 0))
        dn_lid = HollowCircle3D(pt3D(-self.height/2, 0, 0), self.in_radius,
                                self.out_radius, vec3D(1, 0, 0))
        self._collider = Group3D([in_cylinder, out_cylinder, up_lid, dn_lid])
        return self._collider

#%%
class ConeElectrode(AbstractElectrode):
    def __init__(self, height, out_small_radius, out_big_radius, thick, name=None):
        self.height = height
        self.out_small_radius = out_small_radius
        self.out_big_radius = out_big_radius
        self.thick = thick
        self._carcass = HollowCuttedCone(height, out_small_radius, out_big_radius, thick)
        self._collider = self.collider
        self._main_box = Box3D(height + 0.01, 2*out_big_radius + 0.02, 2*out_big_radius + 0.02)
        self.name = name

    @property
    def main_box(self):
        return self._main_box

    @property
    def carcass(self):
        return self._carcass

    @property
    def collider(self):
        if hasattr(self, '_collider'):
            return self._collider
        in_cone = CuttedConeShell(self.height, self.out_small_radius - self.thick,
                                  self.out_big_radius - self.thick)
        out_cone = CuttedConeShell(self.height, self.out_small_radius,
                                   self.out_big_radius)
        up_lid = HollowCircle3D(pt3D(self.height/2, 0, 0), self.out_big_radius -
                                self.thick, self.out_big_radius, vec3D(1, 0, 0))
        dn_lid = HollowCircle3D(pt3D(-self.height/2, 0, 0), self.out_small_radius -
                                self.thick, self.out_small_radius, vec3D(1, 0, 0))
        self._collider = Group3D([in_cone, out_cone, up_lid, dn_lid])
        return self._collider

#%%
class BellevilleWasherElectrode(AbstractElectrode): # Тарельчатая шайба
    def __init__(self, washer_height, full_height, out_radius, thick, name=None):
        self.height = washer_height
        self.full_height = full_height
        self.out_radius = out_radius
        self.thick = thick
        self._carcass = self.carcass
        self._collider = self.collider
        self._main_box = Box3D(full_height + 0.01, 2*out_radius + 0.02, 2*out_radius + 0.02)
        self.name = name

    @property
    def main_box(self):
        return self._main_box

    @property
    def carcass(self):
        if hasattr(self, '_carcass'):
            return self._carcass

        main = HollowCylinder(self.full_height, self.out_radius - self.thick,
                              self.out_radius)

        cone_1 = CuttedCone(self.full_height - self.height, self.out_radius -
                            self.thick, self.out_radius)
        cone_2 = HollowCuttedCone(self.full_height - self.height, self.out_radius,
                                  self.out_radius + self.thick, self.thick)
        cone_vec = vec3D(self.height/2, 0, 0)
        cone_1.translate( cone_vec)
        cone_2.translate(-cone_vec)
        hole = Group3D([cone_1, cone_2])

        self._carcass = Group3D([main, hole])
        return self._carcass

    def contains_pt(self, pt):
        if outside_gabarits(pt, self.gabarits()):
            return False
        return (self.carcass[0].contains_pt(pt) and
                (not self.carcass[1].contains_pt(pt)))

    @property
    def collider(self):
        if hasattr(self, '_collider'):
            return self._collider

        in_cylinder = CylindricShell(self.height, self.out_radius - self.thick)
        out_cylinder = CylindricShell(self.height, self.out_radius)
        in_cone = CuttedConeShell(self.full_height - self.height,
                                  self.out_radius - self.thick, self.out_radius)
        out_cone = CuttedConeShell(self.full_height - self.height,
                                  self.out_radius - self.thick, self.out_radius)

        cyl_vec  = vec3D((self.full_height - self.height)/2, 0, 0)
        cone_vec = vec3D(self.height/2, 0, 0)
        in_cylinder.translate(-cyl_vec)
        out_cylinder.translate(cyl_vec)
        in_cone.translate(cone_vec)
        out_cone.translate(-cone_vec)

        self._collider = Group3D([in_cone, out_cone, in_cylinder, out_cylinder])
        return self._collider

#%%
class AbstractElectrode2D:
    @property
    def U(self):
        if hasattr(self, "_U"):
            return self._U
        return 0.0

    @U.setter
    def U(self, U):
        self._U = U

    def set_U(self, U):
        self.U = U

    @abstractproperty
    def carcass(self):
        """
        carcass must contain 1 obj with methods: transform, translate,
        contains_pt.

        Returns
        -------
        carcass : Group3D

        """

    def contains_pt(self, pt):
        return self.carcass.contains_pt(pt)

    def masks_with_voltages(self, grid):
        """
        Parameters
        ----------
        grid : np.mgrid

        Returns
        -------
        [[mask, voltage]]
        mask : np.array of bool with shape grid.shape[1:] #!!!
        voltage : float

        """
        mask = fast_scalar_field_on_grid(grid, self.carcass.contains_pt, gabarits=self.gabarits(), outvalue=False)
        return [[mask, self.U]]

    def transform(self, mx):
        self.carcass.transform(mx)
        self._mx = mx.dot(self._mx)
        self._vec = mx.dot(self._vec)

    def translate(self, vec):
        self.carcass.translate(vec)
        self._vec += vec

    @abstractmethod
    def restore_3D(self):
        """

        Returns
        -------
        [electrode_3D]

        """

    @property
    def color(self):
        if not (hasattr(self, "edgecolor") and hasattr(self, "facecolor")):
            return [None, None]
        return [self.edgecolor, self.facecolor]

    @color.setter
    def color(self, color_array):
        self.edgecolor = color_array[0]
        self.facecolor = color_array[1]

    def set_color(self, edgecolor, facecolor):
        self.edgecolor = edgecolor
        self.facecolor = facecolor

    def plot(self, axes_code=None, **kwargs):
        self.carcass.plot(axes_code, facecolor = self.color[1], edgecolor = self.color[0], **kwargs)

    def gabarits(self):
        return self.carcass.gabarits()

    def recalc_gabarits(self):
        self.carcass.recalc_gabarits()

#%%
class CylindricElectrode2D(AbstractElectrode2D):
    def __init__(self, height, out_radius, thick, name=None):
        self.height = height
        self.thick = thick
        self.in_radius = out_radius - thick
        self.out_radius = out_radius
        self._carcass = self.carcass
        self._vec = np.array([0., 0.])
        self._mx = identMx2D()
        self.name = name

    @property
    def carcass(self):
        if hasattr(self, "_carcass"):
            return self._carcass
        x = self.height/2
        y2 = self.out_radius
        y1 = self.out_radius - self.thick
        up = Polygon2D([(-x,  y1), (-x,  y2), (x,  y2), (x,  y1)])
        dn = Polygon2D([(-x, -y1), (-x, -y2), (x, -y2), (x, -y1)])
        self._carcass = Group2D([dn, up])
        return self._carcass

    def restore_3D(self):
        electrode_3D = CylindricElectrode(self.height, self.out_radius, self.thick, name=self.name)
        vec = np.hstack((self._vec, [0.]))
        mx = mx2Dto3D(self._mx)
        electrode_3D.transform(mx)
        electrode_3D.translate(vec)
        return [electrode_3D]

#%%
class ConeElectrode2D(AbstractElectrode2D):
    def __init__(self, height, out_small_radius, out_big_radius, thick, name=None):
        self.height = height
        self.out_small_radius = out_small_radius
        self.out_big_radius = out_big_radius
        self.thick = thick
        self._carcass = self.carcass
        self._vec = np.array([0., 0.])
        self._mx = identMx2D()
        self.name = name

    @property
    def carcass(self):
        if hasattr(self, "_carcass"):
            return self._carcass
        x = self.height/2
        ys2 = self.out_small_radius
        ys1 = self.out_small_radius - self.thick
        yb2 = self.out_big_radius
        yb1 = self.out_big_radius - self.thick
        up = Polygon2D([(-x,  ys1), (-x,  ys2), (x,  yb2), (x,  yb1)])
        dn = Polygon2D([(-x, -ys1), (-x, -ys2), (x, -yb2), (x, -yb1)])
        self._carcass = Group2D([dn, up])
        return self._carcass

    def restore_3D(self):
        electrode_3D = ConeElectrode(self.height, self.out_small_radius,
                                     self.out_big_radius, self.thick, name=self.name)
        vec = np.hstack((self._vec, [0.]))
        mx = mx2Dto3D(self._mx)
        electrode_3D.transform(mx)
        electrode_3D.translate(vec)
        return [electrode_3D]

#%%
class BellevilleWasher2D(AbstractElectrode2D):
    def __init__(self, washer_height, full_height, out_radius, thick, name=None):
        self.height = washer_height
        self.full_height = full_height
        self.out_radius = out_radius
        self.thick = thick
        self._carcass = self.carcass
        self._vec = np.array([0., 0.])
        self._mx = identMx2D()
        self.name = name

    @property
    def carcass(self):
        if hasattr(self, "_carcass"):
            return self._carcass
        x_edge = self.full_height/2
        x_mid  = x_edge - self.height
        y1 = self.out_radius - self.thick
        y2 = self.out_radius
        up = Polygon2D([(-x_edge,  y1), (-x_mid,  y2), (x_edge,  y2), (x_mid,  y1)])
        dn = Polygon2D([(-x_edge, -y1), (-x_mid, -y2), (x_edge, -y2), (x_mid, -y1)])
        self._carcass = Group2D([dn, up])
        return self._carcass

    def restore_3D(self):
        electrode_3D = BellevilleWasherElectrode(self.height, self.full_height,
                                                 self.out_radius, self.thick,
                                                 name=self.name)
        vec = np.hstack((self._vec, [0.]))
        mx = mx2Dto3D(self._mx)
        electrode_3D.transform(mx)
        electrode_3D.translate(vec)
        return [electrode_3D]

#%%
Group3D.call_strategies['restore_3D'] = ListStrategy()
Group3D.call_strategies['set_U']      = JustCallStrategy()
Group3D.call_strategies['set_color']  = JustCallStrategy()
Group3D.call_strategies['make_hole']  = JustCallStrategy()
