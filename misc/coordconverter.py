# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:07:18 2024

@author: Krokhalev, Ammosov
"""

#%%
import math
import numpy as np
from abc import ABC, abstractmethod

from ..geom.geom import rotateMx
from ..phys.constants import deg
from ..misc.interp import FastGridInterpolator
from ..misc.tokameq import TokameqFile

#%%
class AbstractCoordConverter(ABC):
    
    @abstractmethod
    def __call__(self, r):
        """
        
        Parameters
        ----------
        r : array like of floats with shape (3,)
            Point in cartesian coordinate system.

        Returns
        -------
        [rho, theta, phi] : list of floats with shape (3,)
            Point in flux coordinate system.

        """
        
def ray_segm_intersect_2d(ray, segm):
    ray_0, ray_1 = ray
    segm_0, segm_1 = segm
    ray_v = ray_1 - ray_0
    segm_v = segm_1 - segm_0
    segm_n = (segm_v[1], -segm_v[0])
    if abs(np.cross(ray_v, segm_v)) < 0.001:
        return None
        
    t_1 = -np.dot((ray_0 - segm_0), segm_n)/np.dot(ray_v, segm_n)
    
    if t_1 > 0:
        r = ray_0 + ray_v*t_1 - segm_0
        t_2 = np.dot(r, r)/np.dot(segm_v, segm_v)
        if t_2 <= 1.:    
            return 1/t_1
        else:
            return None
    else:
        return None

class CoordConverterDShapeSimple(AbstractCoordConverter):
    def __init__(self, R, separatrix):
        self.R = R
        self.sep = separatrix
    
    def __call__(self, r):
        phi = self.calc_phi(r)
        rho, theta = self.calc_rho_and_theta(r, phi)
        return [rho, theta, phi]
    
    def calc_R_vector(self, r):
        x, y, z = r
        R_v = np.array([x, 0, z])/np.sqrt(x**2+z**2)*self.R
        return R_v
    
    def calc_phi(self, r):
        R_v = self.calc_R_vector(r)
        phi = np.arcsin(R_v[2]/self.R)
        # if R_v[2] < 0:
        #     phi = phi - np.pi
        return phi
    
    def calc_rho_and_theta(self, r, phi):
        r = np.asarray(r)
        mx = rotateMx(np.array([0, 1, 0]), ang=-phi*deg) #!!! fixed from [1, 0, 0] 26.07.24 by Oleg
        r_eq = mx.dot(r)
        
        # r_eq[2] = 0. # it should be 0. after rotation, but to avoid computational mistakes we set it 0. additionally
        # deprivated 26.07.24 by Oleg
        
        r_eq = r_eq[0:2]
        r_center = [self.R, 0]
        r_local = r_eq - r_center
        if np.all(np.isclose(r_local, [0., 0.])):
            return 0., 0.
        theta = math.atan2(r_local[1], r_local[0])
        for segm_0, segm_1 in zip(self.sep[:-1], self.sep[1:]):
            rho = ray_segm_intersect_2d((r_center, r_eq), (segm_0, segm_1))
            if rho is not None:
                break
        else:
            return 0., 0.
        return rho, theta
    
class FluxCoordConverter(CoordConverterDShapeSimple): #!!! needs to be tested before use
    '''
    rho is calculated as square root of magnetic flux
    '''
    def __init__(self, points, flux_data_2D):
        self.flux = FastGridInterpolator(points, flux_data_2D, fill_value=None, method='linear') # what fill value needed?
        
        min_flux_index = np.argmin(self.flux.values)
        self.R = self.flux.xx[min_flux_index[0]]
    
    def calc_rho_and_theta(self, r, phi):
        r = np.array(r)
        mx = rotateMx(np.array([0, 1, 0]), ang=-phi*deg)
        r_eq = mx.dot(r)[0:2]
        
        local_flux = self.flux(r_eq)
        rho = math.sqrt(local_flux)
        
        r_center = [self.R, 0]
        r_local = r_eq - r_center
        theta = math.atan2(r_local[1], r_local[0])
        
        return rho, theta
    
    @classmethod
    def from_toqamek_file(cls, filename):
        file = TokameqFile(filename)
        points = [file.F.rr, file.F.zz[::-1]]
        data = file.F.values[::-1, :].T
        return cls(points, data)
    
    @classmethod
    def from_dina_file(cls, filename):
        with open(filename, 'rb'):
            points = []
            data = []
        
        return cls(points, data)

#%%
if __name__ == "__main__":
    
    separatrix = np.loadtxt("D:\py\programs\SynHIBP\devices\T-15MD\geometry\T15_sep.txt") / 1000
    T15_R = 1.5 # [m]
    T15_separatrix = []
    for sep in separatrix:
        T15_separatrix.append([sep[0]+T15_R, sep[1]])
    T15_separatrix = np.asarray(T15_separatrix)
        
    