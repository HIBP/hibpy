# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:12:29 2023

@author: Eliseev_LG, Krohalev_OD

class AbstractAim
    intersect_with_segment(r0, r1)
    front_polygon
    basis
    hit(r0, r1)
    center
    deviation(r)
"""
#%% imports
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
import numpy as np
import math

from ..geom.prim import Polygon3D, Plane3D
from ..geom.group import Group3D
from ..geom.geom import transformMx, vec3D, normalizedVector
from ..phys.constants import deg

#%%
def calc_angles(vec, basis):
    pitch = vec.dot(basis[0])
    yaw   = vec.dot(basis[1])
    norm  = vec.dot(basis[2])
    pitch_angle = math.atan2(pitch, norm)
    yaw_angle   = math.atan2(yaw,   norm)
    return (pitch_angle, yaw_angle)

#%%
class AbstractAim(ABC):
    @abstractmethod
    def intersect_with_segment(self, pt0, pt1):
        """
        Parameters
        ----------
        r0, r1 : np.array of float with shape (3,)

        Returns
        -------
        None if no intersection
        else r_intersect : np.array of float with shape (3,)
        """
        
    @abstractproperty
    def front_polygon(self):
        """
        ???
        
        Returns
        -------
        None.

        """    
    @abstractproperty
    def basis(self):
        """
        basis vector 0 is in preffered obj direction if where is any - pitch
        basis vector 1 is in polygon plane normal to bv0             - yaw/lateral
        basis vector 2 is normal to polygon                          - normal

        Returns
        -------
        basis : np.array of float with shape (3,3)
        """
    
    @abstractproperty
    @basis.setter
    def basis(self, basis):
        """

        basis vector 0 is in preffered obj direction if where is any - pitch
        basis vector 1 is in polygon plane normal to bv0             - yaw/lateral
        basis vector 2 is normal to polygon                          - normal

        Returns
        -------
        None.

        """
        
    @abstractmethod
    def hit(self, pt0, pt1):
        """
        Parameters
        ----------
        r0, r1 : np.array of float with shape (3,)

        Returns
        -------
        tuple of (success, point) : 
            success : bool, True if hit aim 
        """
    
    @abstractproperty
    def center(self):
        """
        Returns
        -------
        center : np.array of float with shape (3,)
        """
        
    @abstractmethod
    def deviation(self, point, vec=None):
        """
        Parameters
        ----------
        point : np.array of float with shape (3,)
        vec   : np.array of float with shape (3,)

        Returns
        -------
        pitch, yaw : float
            deviation of r from aim center in aim basis vectors pitch and yaw
        """
        
    @abstractproperty 
    def plane(self):
        """
        
        Returns
        -------
        plane : Plane3D

        """
        
    @abstractproperty
    def transformable(self):
        """

        Returns
        -------
        Group3D of obj which will be transformed when Aim is transformed

        """
        
    @abstractproperty
    def mx(self):
        """
        Returns self._mx
        """
        
    @abstractmethod
    def recalc_mx(self):
        """
        recalcs self._mx

        Returns
        -------
        None.

        """
    
    def transform(self, mx):
        self.transformable.transform(mx)
        self.basis = np.asarray([mx.dot(vec) for vec in self.basis])
        self.recalc_mx()
        
    def translate(self, vec):
        self.transformable.translate(vec)
        
#%%
class DotAim(AbstractAim):
    def __init__(self, pt, basis, eps_pitch, eps_yaw):
        self._basis = []
        for vector in basis:
            self._basis.append(normalizedVector(vector))
        self._basis = np.array(self._basis)
        
        self._polygon = Polygon3D([pt + self._basis[0]*eps_pitch + self._basis[1]*eps_yaw,
                                   pt + self._basis[0]*eps_pitch - self._basis[1]*eps_yaw,
                                   pt - self._basis[0]*eps_pitch - self._basis[1]*eps_yaw,
                                   pt - self._basis[0]*eps_pitch + self._basis[1]*eps_yaw])
        self._target = pt
        
        self._plane = Plane3D(pt, deepcopy(self._basis[2]))
        self._transformable = self.plane
        self.eps = np.array([eps_pitch, eps_yaw])
        
        self._mx = None
        self.recalc_mx()
        
    def intersect_with_segment(self, pt0, pt1):
        return self._plane.intersect_with_segment(pt0, pt1)
    
    def hit(self, pt0, pt1):
        point = self._plane.intersect_with_segment(pt0, pt1)
        if point is None:
            return False
        else:
            # pt_loc = self._mx.dot(point - self.center)
            if all(abs(self.deviation(point, pt1 - pt0)) < self.eps):
                return True
            else:
                return False
            
    def deviation(self, point, vec):
        pt_loc = self._mx.dot(point - self.center)
        return pt_loc[:2]
    
    @property
    def mx(self):
        return self._mx
    
    @property
    def front_polygon(self):
        return self._polygon
    
    @property
    def center(self):
        return self._target
    
    @property
    def transformable(self):
        return self._transformable
    
    @property
    def plane(self):
        return self._plane
    
    @property
    def basis(self):
        return self._basis
    
    @basis.setter
    def basis(self, basis):
        self._basis = basis
        
    def recalc_mx(self):
        self._mx = transformMx([vec3D(1, 0, 0), vec3D(0, 1, 0), vec3D(0, 0, 1)], 
                               self._basis)
        
class DotAimAdaptiveOneAngleLinear(DotAim):
    def __init__(self, pt, basis, eps_pitch, eps_yaw, correction_pitch, correction_yaw,
                 reference_pitch_angle, reference_yaw_angle, max_correction_angles=[20*deg, 25*deg]):
        super().__init__(pt, basis, eps_pitch, eps_yaw)
        self.correction = [correction_pitch, correction_yaw]
        self.reference_angles = [reference_pitch_angle, reference_yaw_angle]
        self.max_correction_angles = max_correction_angles
        
    def deviation(self, point, vec):
        pitch_angle, yaw_angle = calc_angles(vec, self.basis)
        main_deviation = super().deviation(point, vec=vec)
        relative_angles = np.asarray([pitch_angle, yaw_angle]) - self.reference_angles
        for i in [0, 1]:
            if abs(relative_angles[i]) > self.max_correction_angles[i]:
                # main_deviation[i] += -self.correction[i]*np.sign(relative_angles[i])
                main_deviation[i] += self.correction[i]*np.sign(relative_angles[i])
            else:
                # main_deviation[i] += -self.correction[i]*relative_angles[i]/self.max_correction_angles[i]
                main_deviation[i] += self.correction[i]*relative_angles[i]/self.max_correction_angles[i]
        return main_deviation
    
class DotAimAdaptiveTwoAngleLinearCombo(DotAimAdaptiveOneAngleLinear):
    def __init__(self, pt, basis, eps_pitch, eps_yaw, correction_pitch, correction_yaw,
                 reference_pitch_angle, reference_yaw_angle, 
                 alpha_to_beta_correction_coefficient,
                 max_correction_angles=[20*deg, 25*deg]):
        super().__init__(pt, basis, eps_pitch, eps_yaw, correction_pitch, correction_yaw,
                     reference_pitch_angle, reference_yaw_angle, max_correction_angles=max_correction_angles)
        self.atb = alpha_to_beta_correction_coefficient
        
    def deviation(self, point, vec):
        pitch_angle, yaw_angle = calc_angles(vec, self.basis)
        relative_angles = np.asarray([pitch_angle, yaw_angle]) - self.reference_angles
        main_deviation = super().deviation(point, vec=vec)
        if abs(relative_angles[1]) > self.max_correction_angles[1]:
            main_deviation[0] += self.correction[0]*self.atb*np.sign(relative_angles[1])
        else:
            main_deviation[0] += self.correction[0]*self.atb*relative_angles[1]/self.max_correction_angles[1]
            
        return main_deviation
    
class PolygonCenterAim(DotAim):
    def __init__(self, pts, main_vec, eps_pitch, eps_yaw):
        self._polygon = Polygon3D(pts)
        self._normal = deepcopy(self._polygon._normal)
        self._pitch  = deepcopy(main_vec)
        self._yaw    = np.cross(self._normal, self._pitch)
        self._basis  = np.array([self._pitch, self._yaw, self._normal])
        _basis = []
        for vector in self._basis:
            _basis.append(normalizedVector(vector))
        self._basis = np.array(_basis)
        
        self._target = self._polygon._center
        self._plane = Plane3D(self._target, self._normal)
        self._transformable = Group3D([self._polygon, self._plane])
        self.eps = np.array([eps_pitch, eps_yaw])
        
        self._mx = None
        self.recalc_mx()
        
    def intersect_with_segment(self, pt0, pt1):
        return self._polygon.intersect_with_segment(pt0, pt1)
    