# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:06:15 2023

@author: Eliseev_LG, Krohalev_OD

class AbstractObject3D
    transform(mx)
    translate(vec)
    plot(axes_code=None, *args, **kwargs)
    
class AbstractCollider3D(AbstractObject3D)
    intersect_with_segment(r0, r1)
    
class AbstractCarcass3D(AbstractObject3D)
    contains_point(pt)
"""
#%% imports
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .geom import (_intersect_plane_segm_, _regularPolygon3D, plot_point, 
                   vNorm, _ptInPolygon3D_, calc_gabarits,
                   normalizedVector, mxScaleCoeff,
                   invMx, ptInPolygon2D, calc_gabarits_2D, identMx2D, 
                   main_basis, transformMx)

from .geom_ex import plot_polygon_ex
from ..misc.grid import func_on_points
from ..geom.group import Group3D
from ..geom.groupstrat import PlusStrategy

#%% abstracts
class AbstractObject3D(ABC):
    @abstractmethod
    def transform(self, mx):
        """
        Transforms obj with transform matrix mx
        
        Parameters
        ----------
        mx : np.array of float with shape (3, 3)

        Returns
        -------
        None.
        """

    @abstractmethod    
    def translate(self, vec):
        """
        Translates obj on vec

        Parameters
        ----------
        vec : np.array of float with shape (3,)

        Returns
        -------
        None.
        """
        
        
    @abstractmethod
    def plot(self, axes_code=None, *args, **kwargs):
        """
        Plot obj in plt.gca()

        Parameters
        ----------
        axes_code : str, optional
            Sets pair of axes in which to plot. For example, 'XY' or 'XZ'. 
            The default is None.
        
        Returns
        -------
        None.
        """
        
class AbstractCollider3D(AbstractObject3D):
    @abstractmethod
    def intersect_with_segment(self, r0, r1):
        """
        Parameters
        ----------
        r0, r1 : np.array of float with shape (3,)

        Returns
        -------
        None if no intersectiron
        intersect_pt : np.array of float with shape (3,) if intersected
        """

class AbstractCarcass3D(AbstractObject3D):
    @abstractmethod
    def contains_pt(self, pt):
        """

        Parameters
        ----------
        pt : np.array of float with shape (3,)

        Returns
        -------
        contains_pt : bool
            True if pt is inside obj
        """
        
    def mask_on_points(self, points):
        return func_on_points(points, self.contains_pt)
        
#%% 
class Plane3D(AbstractCollider3D):
    def __init__(self, point, normal):
        self.point = np.asarray(point)
        self.normal = np.asarray(normal)
        
    def transform(self, mx):
        """ doesn't work for scew """
        self.normal = mx.dot(self.normal)
        self.point = mx.dot(self.point)
        
    def translate(self, vec):
        self.point += vec
        
    def intersect_with_segment(self, r0, r1):
        intersect_pt, t = _intersect_plane_segm_((self.point, self.normal), (r0, r1))
        if intersect_pt is None: 
            return None
        if (t < 0.0) or (t > 1.0): 
            return None        
        return intersect_pt

    def plot(self, axes_code=None, *args, **kwargs): 
        plot_point(self.point, axes_code=axes_code)
        for i in range(1, 4, 1):
            circle = _regularPolygon3D(16, self.point, vNorm(self.normal)*i, self.normal, closed=True)
            plot_polygon_ex(circle, axes_code=axes_code, **kwargs)
            
#%%
class Polygon2D:
    def __init__(self, pts):
        self._points = np.asarray(pts)
        self._gabarits = None
        self._vec = np.array([0., 0.])
        self._mx = identMx2D()
        self._imx = identMx2D()
        self.recalc_gabarits()
        
    def contains_pt(self, pt):
        return ptInPolygon2D(self._points, pt)
    
    @property
    def center(self):
        return np.average(self._points, axis=0)
    
    def transform(self, mx):
        self._points = [mx.dot(pt) for pt in self._points] 
        self._vec = mx.dot(self._vec)
        self._mx = mx.dot(self._mx)
        self._imx = invMx(self._mx)
        self.recalc_gabarits()
        
    def translate(self, vec):
        self._points = [pt + vec for pt in self._points]
        self._vec += vec
        self.recalc_gabarits()
        
    def plot(self, axes_code=None, **kwargs):
        facecolor = kwargs.get('facecolor')
        edgecolor = kwargs.get('edgecolor')
        # points = self._points
        # points.append(self._points[0])
        poly = Polygon(self._points, fill=True, facecolor=facecolor, edgecolor=edgecolor)
        ax = plt.gca()
        ax.add_patch(poly)
        plt.show()
        
    def gabarits(self):
        return self._gabarits
    
    def recalc_gabarits(self): 
        self._gabarits = calc_gabarits_2D( self._points )
        
def gabaritbox2D(gbr):
    gbr0 = gbr[0]
    gbr1 = gbr[1]
    return Polygon2D([gbr0, [gbr0[0], gbr1[1]], gbr1, [gbr1[0], gbr0[1]]])

#%%
class Polygon3D(AbstractCollider3D): 
    def __init__(self, points):
        self._points = np.asarray(points)
        self._center = np.average(self._points, axis=0)
        self._normal = self.calc_normal()
        self._gabarits = None
        self.recalc_gabarits()
    
    def calc_normal(self): 
        n3 = len(self._points) // 3
        p1, p2, p3 = self._points[0], self._points[n3], self._points[n3*2]
        return np.cross(p2 - p1, p3 - p1)
 
    def points(self): 
        return self._points  # yield from self._points       

    def intersect_with_segment(self, pt0, pt1): 
        intersect_pt, t = _intersect_plane_segm_((self._center, self._normal), (pt0, pt1))
        if intersect_pt is None: 
            return None
        elif (t < 0.0)or(t > 1.0): 
            return None        
        elif _ptInPolygon3D_(self._points, intersect_pt): 
            return intersect_pt
        else:
            return None

    def translate(self, vec): 
        self._points = [pt + vec for pt in self._points]
        self._center += vec
        #self._normal = self._normal
        self.recalc_gabarits()
        return self

    def transform(self, mx): 
        """ doen't work for skew """
        self._points = [mx.dot(pt) for pt in self._points]
        self._center = mx.dot(self._center)
        self._normal = self.calc_normal()  # mx.dot(self._normal) # ??? only for rotation !!! not for skew
        self.recalc_gabarits()
        return self
    
    def gabarits(self):
        return self._gabarits
    
    def recalc_gabarits(self): 
        self._gabarits = calc_gabarits( self.points() )

    def plot(self, axes_code=None, *args, **kwargs): 
        plot_polygon_ex(self._points, axes_code=axes_code, **kwargs)
        
def as_polygon3D(polygon, basis=main_basis):
    
    if not isinstance(polygon, (Polygon2D, Polygon3D)):
        pts = polygon
    elif isinstance(polygon, Polygon3D):
        return polygon
    else:
        pts = polygon._points
        
    dim = len(np.asarray(pts).shape)
    
    if dim == 3:
        return Polygon3D(pts)
    
    elif dim == 2:
        mx = transformMx(main_basis, basis)
        points = []
        for pt in pts:
            point = list(pt)
            point.append(0.)
            point = np.asarray(point)
            point = mx.dot(point)
            points.append(point)
        return Polygon3D(points)
    
    else:
        raise TypeError
        
#%%
class Circle3D(AbstractCollider3D):
    def __init__(self, center, radius, normal):
        self.center = deepcopy(center)
        self.normal = normalizedVector(normal) 
        self.radius = radius
        self.main_plane = Plane3D(deepcopy(center), deepcopy(self.normal))
        self._gabarits = None
        self.recalc_gabarits()
    
    def scale_size(self, scale_coeff):
        self.radius *= scale_coeff
    
    def transform(self, mx):
        self.center = mx.dot(self.center)
        self.normal = normalizedVector(mx.dot(self.normal))
        self.scale_size(mxScaleCoeff(mx))
        self.main_plane.transform(mx)
        
    def translate(self, vec):
        self.center += vec 
        self.main_plane.translate(vec)
    
    def polygons(self, npolygons, npoints, closed=True):
        pgs = []
        rr = np.linspace(0.1*self.radius, self.radius, npolygons)
        for r in rr:
            pg = _regularPolygon3D(npoints, self.center, r, 
                                   -self.normal, closed=closed)
            pgs.append(pg)
        return pgs
    
    def plot(self, axes_code=None, npolygons=10, *args, **kwargs):
        pgs = self.polygons(npolygons, 200, True)
        color = kwargs.get('color')
        if color is None:
            for pg in pgs: 
                color = plot_polygon_ex(pg, axes_code=axes_code, color=color, **kwargs)[0].get_color()
        else:         
            for pg in pgs: 
                plot_polygon_ex(pg, axes_code=axes_code, **kwargs)
                
    def intersect_with_segment(self, r0, r1):
        intersect_pt = self.main_plane.intersect_with_segment(r0, r1)
        if intersect_pt is None:
            return None
        
        dr = intersect_pt - self.center
        if vNorm(dr) > self.radius:
            return None
        return intersect_pt
    
    def gabarits(self):
        return self._gabarits
    
    def recalc_gabarits(self): 
        self._gabarits = calc_gabarits( _regularPolygon3D(100, self.center, self.radius, 
                                                          -self.normal, closed=True) )
    
class HollowCircle3D(Circle3D):
    def __init__(self, center, in_radius, out_radius, normal):
        self.center = deepcopy(center)
        self.normal = normalizedVector(normal) 
        self.in_radius = in_radius
        self.out_radius = out_radius
        self.main_plane = Plane3D(deepcopy(center), deepcopy(normal))
        self._gabarits = None
        self.recalc_gabarits()
    
    def scale_size(self, scale_coeff):
        self.in_radius *= scale_coeff    
        self.out_radius *= scale_coeff  
    
    def polygons(self, npolygons, npoints, closed=True):
        pgs = []
        rr = np.linspace(self.in_radius, self.out_radius, npolygons)
        for r in rr:
            pg = _regularPolygon3D(npoints, self.center, r, 
                                   -self.normal, closed=closed)
            pgs.append(pg)
        return pgs
    
    def intersect_with_segment(self, r0, r1):
        intersect_pt = self.main_plane.intersect_with_segment(r0, r1)
        if intersect_pt is None:
            return None
        
        dr = intersect_pt - self.center
        dr_norm = vNorm(dr)
        if (dr_norm < self.out_radius) and (dr_norm > self.in_radius):
            return intersect_pt
        return None
    
    def recalc_gabarits(self): 
        self._gabarits = calc_gabarits( _regularPolygon3D(100, self.center, self.out_radius, 
                                                          -self.normal, closed=True) )
    
#%% 
class PolygonWithHole3D(Polygon3D):
    def __init__(self, points, hole_pts):
        self._points = np.asarray(points)
        self._hole = hole_pts
        self._center = np.average(self._points, axis=0)
        self._normal = self.calc_normal()
        self._gabarits = None
        self.recalc_gabarits()
        
    def intersect_with_segment(self, pt0, pt1): 
        intersect_pt, t = _intersect_plane_segm_((self._center, self._normal), (pt0, pt1))
        if intersect_pt is None: 
            return None
        elif (t < 0.0) or (t > 1.0): 
            return None        
        elif _ptInPolygon3D_(self._points, intersect_pt): 
            if _ptInPolygon3D_(self._hole, intersect_pt):
                return None
            return intersect_pt
        else:
            return None

    def translate(self, vec): 
        self._points = [pt + vec for pt in self._points]
        self._hole   = [pt + vec for pt in self._hole]
        self._center += vec
        #self._normal = self._normal
        self.recalc_gabarits()
        return self

    def transform(self, mx): 
        """ doen't work for skew """
        self._points = [mx.dot(pt) for pt in self._points]
        self._hole   = [mx.dot(pt) for pt in self._hole]
        self._center = mx.dot(self._center)
        self._normal = self.calc_normal()  # mx.dot(self._normal) # ??? only for rotation !!! not for skew
        self.recalc_gabarits()
        return self
    
    def plot(self, axes_code=None, *args, **kwargs): 
        plot_polygon_ex(self._points, axes_code=axes_code, **kwargs)
        plot_polygon_ex(self._hole,   axes_code=axes_code, **kwargs)
        
#%%
class ColliderWithHole(AbstractCollider3D):
    def __init__(self, collider, hole):
        self._collider = collider
        self._hole = hole
        
    def gabarits(self):
        return self._collider.gabarits()
        
    def transform(self, mx):
        self._collider.transform(mx)
        self._hole.transform(mx)
        
    def translate(self, vec):
        self._collider.translate(vec)
        self._hole.translate(vec)
        
    def intersect_with_segment(self, pt0, pt1):
        intersect_pt = self._collider.intersect_with_segment(pt0, pt1)
        if intersect_pt is None: 
            return None
        if self._hole.intersect_with_segment(pt0, pt1) is None:
            return intersect_pt
        else:
            return None
    
    def plot(self, *args, **kwargs):
        self._collider.plot(*args, **kwargs)
        self._hole.plot(*args, **kwargs)

#%%
class CarcassWithHole(AbstractCarcass3D):
    def __init__(self, carcass, hole):
        self._carcass = carcass
        self._hole = hole
        
    def gabarits(self):
        return self._carcass.gabarits()
        
    def transform(self, mx):
        self._carcass.transform(mx)
        self._hole.transform(mx)
        
    def translate(self, vec):
        self._carcass.translate(vec)
        self._hole.translate(vec)
        
    def contains_pt(self, pt):
        if self._carcass.contains_pt(pt) and (not self._hole.contains_pt(pt)):
            return True 
        return False
    
    def plot(self, *args, **kwargs):
        self._carcass.plot(*args, **kwargs)
        self._hole.plot(*args, **kwargs)
        
#%%
class ColliderCutWithCarcass(ColliderWithHole):
    def intersect_with_segment(self, pt0, pt1):
        intersect_pt = self._collider.intersect_with_segment(pt0, pt1)
        if intersect_pt is None: 
            return None
        if not (self._hole.contains_pt(pt0) or self._hole.contains_pt(pt1)):
            return intersect_pt
        else:
            return None
        
    def plot(self, *args, **kwargs):
        self._collider.plot(*args, dont_plot_condition=self._hole.contains_pt, **kwargs)
        
#%%
Group3D.call_strategies['mask_on_points'] = PlusStrategy()