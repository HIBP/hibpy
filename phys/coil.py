# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:31:13 2023

@author: Eliseev_LG, Krohalev_OD
"""
#%% imports
from abc import ABC, abstractmethod
import copy
import numpy as np
import matplotlib.pyplot as plt

from .ring import Ring
from ..geom.group import Group3D
from ..geom.groupstrat import JustCallStrategy
from ..geom.geom import normalized_vector, plot_polygon, get_coord_indexes, line_array
from ..misc.tokameq import TokameqFile
from ..misc.grid import _grid_to_raw_points, _field_from_raw_repr, func_on_points
from ..phys.constants import MAGFIELD_COEFF, SI_1MA

#%% 
class AbstractCoil(ABC):
    
    @abstractmethod
    def calcB(self, r):
        """
        Calc magnetic field in given point

        Parameters
        ----------
        r : np.array with shape (3,)
            radius-vector of point

        Returns
        -------
        B : np.array with shape (3,) 
            [Bx, By, Bz]

        """
        
    @abstractmethod
    def transform(self, mx):
        """
        Rotate object with given matrix

        Parameters
        ----------
        mx : hibpcalc.geometry.matrix Matrix
            rotation matrix

        Returns
        -------
        None.

        """
        
    @abstractmethod
    def translate(self, vec):
        """
        translate object on given vector

        Parameters
        ----------
        vec : hibpcalc.geometry.vector Vector

        Returns
        -------
        None.

        """
        
#%% 
def calc_Bpoint(r, IdL, r_elem, EPS_WIRE = 0.0005):
    """
    calc magnetic field from given array of quasi-point currents in point

    Parameters
    ----------
    r : array of float with shape (3,), [m]
    IdL : array of float with shape (n, 3), [A*m]
    r_elem : array of float with shape (n, 3), [m]
    EPS_WIRE : float, [m]
        if point is closer to current than EPS, B is set [0., 0., 0.]

    Returns
    -------
    s : np.array of float with shape (3,)
        total magnetic field in point, [T]

    """
    r2 = r - r_elem
    r25 = np.linalg.norm(r2, axis=1)
    # create a mask to exclude points close to wire
    mask = (r25 < EPS_WIRE)
    r25 = r25**3
    r3 = r2 / r25 [:, np.newaxis]

    cr = np.cross(IdL, r3)
    cr[mask] = [0., 0., 0.]

    # claculate sum of contributions from all current elements
    s = np.sum(cr, axis=0) * MAGFIELD_COEFF
    return s

class AnyShapeCoil(AbstractCoil):
    def __init__(self, inner_coil_array, outer_coil_array, z_width, n_R, n_tor, J):
        """

        Parameters
        ----------
        inner_coil_array : np.array of floats with shape (len, 3), [m]
            array containing points of inner coil side
        outer_coil_array : np.array of floats with shape (len, 3), [m]
            array containing points of outer coil side
        z_width : float, [m]
            size of coil in toroidal direction
        n_R : int
            number of filaments in coil along x (in radial direction)
        n_tor : int
            number of filaments in coil along z (in toroidal direction)
        J : float
            total current in coil, [A]

        Returns
        -------
        None.

        """
        self.inner_coil_array = inner_coil_array
        self.outer_coil_array = outer_coil_array
        self.z_width = z_width
        self.n_R = n_R
        self.n_tor = n_tor
        
        #create n_R*n_tor equally spaced filamentes inside coil
        tt = np.linspace(0, 1, n_R) if (n_R > 1) else [0.5]
        zz = np.linspace(-z_width/2, z_width/2, n_tor) if (n_tor > 1) else [0.]
        filaments = []
        for t in tt: 
            for z in zz: 
                coil_array = outer_coil_array - t*(outer_coil_array - inner_coil_array)
                coil_array[:, 2] = z 
                filaments.append(coil_array)
        
        self.filaments = filaments 
        self.current = J
    
    @property
    def current(self):
        return self.J
    
    @current.setter
    def current(self, J):
        self.J = J
        self.filament_currents = [J / len(self.filaments) for filament in self.filaments]
    
    @classmethod
    def from_txt(cls, path, J=1.):
        """
        Loads coil from txt file. 
        file mast have following structure:
        
            # first comment
            coeff   z_width     n_R     n_tor
            # second comment
            x_in    y_in        x_out   y_out
            
        all coords and sizes must be given in same measure units
        all comments must begin with #
        last point of coil must be equal to first one
        coeff   - coefficient to transform coordinates in [m]
        z_width - coil size in toroidal direction
        n_R     - number of filaments along radial direction
        n_tor   - number of filaments along toroidal direction

        Parameters
        ----------
        path : str
            full path to txt with coil data
        J : float
            total current in coil, [A]. Default is 1.

        Returns
        -------
        coil : AnyShapeCoil

        """

        data = np.loadtxt(path) # [m]
        coeff, z_width, n_R, n_tor = data[0]
        z_width = z_width*coeff
        n_R = int(round(n_R, 0))
        n_tor = int(round(n_tor, 0))
        data = data[1:]*coeff
        N = data.shape[0]
        zz = np.zeros(N)

        # data has only x and y columns, add 0. in z column
        outer_coil_array = np.vstack( (data[:, [2, 3]].T, zz) ).T
        inner_coil_array = np.vstack( (data[:, [0, 1]].T, zz) ).T
        
        coil = AnyShapeCoil(inner_coil_array, outer_coil_array, z_width, n_R, n_tor, J)
        return coil
    
    @classmethod
    def from_single_path(cls, points, J):
        points = np.array(points)
        result = cls(points, points, 1.0, 1, 1, J)
        result.filaments = [points]
        return result
    
    def calcB(self, r):
        B = np.array([0., 0., 0.])
        for filament, filament_current in zip(self.filaments, self.filament_currents): 
            arr0 = filament[0:-1, :]
            arr1 = filament[1:  , :]
            rr = (arr0 + arr1) * 0.5
            dL = arr1 - arr0
            IdL = dL * filament_current
            B += calc_Bpoint(r, IdL, rr)
        return B
    
    def discretize(self, discret_len):
        for fil_n, filament in enumerate(self.filaments):
            new_filament = []
            for r1, r2 in zip(filament[:-1], filament[1:]):
                dr = np.linalg.norm(r2-r1)
                n = int(np.ceil(dr/discret_len))
                for i in range(n):
                    new_filament.append(r1 + (r2 - r1)*i/n)
            new_filament.append(filament[-1])
            new_filament = np.array(new_filament)
            self.filaments[fil_n] = new_filament
            
    
    def transform(self, mx):
        for filament in self.filaments:         
            for i in range(filament.shape[0]): 
                filament[i] = mx.dot( filament[i] )
        
    def translate(self, vec):
        for filament in self.filaments: 
            filament += vec
        
    def plot(self, axes_code='XY', *args, **kwargs):
        """

        Parameters
        ----------
        ax : matplotlib.Axis, optional
            if not given, new one will be created. The default is None.
        axes : str, optional
            pair of axes in which coil is to be plotted. The default is 'XY'.

        Returns
        -------
        None.

        """

        X, Y = get_coord_indexes(axes_code)

        for filament in self.filaments: 
            plt.plot(filament[:, X], filament[:, Y], *args,**kwargs)
            
#%%
class RoundCoil(AbstractCoil):
    def __init__(self, center, radius, width, height, n_R, n_y, J, 
                 normal=np.array([0., 1., 0.]), name=None):
        """
        
        Parameters
        ----------
        center : np.array of float with shape (3,) [m]
        radius : float [m]
        width : float [m]
            size along x
        height : float [m]
            size along y
        n_x : int
            number of filaments in coil along x
        n_y : int
            number of filaments in coil along y
        J : float
            total current in coil
        normal : array of float with shape (3,), optional
             The default is [0., 1., 0.].

        Returns
        -------
        None.

        """
        self.center = center
        self.radius = radius
        self.width = width
        self.height = height
        self.n_R = n_R
        self.n_y = n_y
        self.normal = normalized_vector(normal)
        self.J = J
        self.name = name
        
        #create n_R*n_y equally spaced filamentes inside coil
        filaments = []
        fil_J = self.J/(self.n_R*self.n_y)
        xx = np.linspace(-width/2, width/2, n_R) if (n_R > 1) else [0.]
        yy = np.linspace(-height/2, height/2, n_y) if (n_y > 1) else [0.]
        for x in xx:
            for y in yy:
                fil_center = self.center + self.normal*y
                fil_radius = self.radius + x
                fil_width =  width/(2*(n_R-1)) if (n_R > 1) else  width/2
                fil_height = height/(2*(n_R-1)) if (n_R > 1) else height/2
                fil_a = min(fil_height, fil_width)
                filament = Ring(fil_center, fil_radius, self.normal, fil_J, fil_a)
                filaments.append(filament)
        
        self.filaments = filaments
    
    @property
    def current(self):
        return self.J
    
    @current.setter
    def current(self, J):
        self.J = J
        for filament in self.filaments:
            filament.I = J/len(self.filaments)
    
    @classmethod
    def from_txt(cls, path, J):
        """
        Loads coil from txt file. 
        file mast have following structure:
        
            # first comment
            coeff       n_R         n_y
            # second comment
            x_center    y_center    width   height
            
        all coords and sizes must be given in same measure units
        all comments must begin with #
        last point of coil must be equal to first one
        coeff   - coefficient to transform coordinates in [m]
        n_R     - number of filaments along radial direction
        n_tor   - number of filaments along toroidal direction

        Parameters
        ----------
        path : str
            full path to txt with coil data
        J : float
            total current in coil, [A]. Default is 1.

        Returns
        -------
        coil : RoundCoil

        """
        #!!!
    
    def calcB(self, r):
        B = np.array([0., 0., 0.])
        for filament in self.filaments:
            B_fil = filament.calcB(r)
            B += B_fil
        return B
    
    def calcB_on_grid(self, grid):
        points = _grid_to_raw_points(grid)
        _B = np.zeros_like(points)
        for batch in self.batches():    
            _B += func_on_points(points, batch.calcB)
        B = _field_from_raw_repr(_B, grid.shape[1:])
        return B
        
    def transform(self, mx):
        """
        doesn't work for skew
        """
        self.center = mx.dot(self.center)
        self.normal = normalized_vector(  mx.dot(self.normal)  )
        for filament in self.filaments:
            filament.transform(mx)
        
    def translate(self, vec):
        self.center += vec
        for filament in self.filaments:
            filament.translate(vec)
    
    def plot(self, axes_code='XY', *args, **kwargs):
        for fil in self.filaments:
            pgn = fil.polygon(100)
            plot_polygon(pgn, axes_code=axes_code, *args, **kwargs)
            
    def batches(self, n=50, max_fil=None):
        i0 = 0
        while True: 
            i1 = i0 + n
            if i1 > len(self.filaments): 
                i1 = len(self.filaments)
            grp = Group3D(self.filaments[i0:i1]) 
            yield grp
            i0 = i1
            if i0 >= len(self.filaments): 
                break
            if (max_fil is not None) and (i0 >= max_fil): 
                break
        
#%%
class PlasmaTokamak(RoundCoil):
    def __init__(self, xx, yy, currents2D):
        """
        
        Parameters
        ----------
        xx : np.array of float [m]
        yy : np.array of float [m]
        currents2D : np.array of floats with shape(len(rr), len(zz))
            currents for each r and z in [A]

        Returns
        -------
        None.

        """
        
        self.xx = xx
        self.yy = yy
        self.currents2D = currents2D
        self.center = np.array([0., 0., 0.])
        self.normal = np.array([0., 1., 0.])
        
        dx = abs(xx[1] - xx[0])
        dy = abs(yy[1] - yy[0])
        fil_a = (dx + dy)/4.
        
        self.filaments = []
        for i, x in enumerate(xx):
            for j, y in enumerate(yy):
                # if (i % 4 != 0) or (j % 4 != 0):
                #     continue #!!!
                if self.currents2D[i, j] != 0.0:
                    fil_center = np.array([0., y, 0.])
                    fil_radius = x
                    filament = Ring(fil_center, fil_radius, self.normal, self.currents2D[i, j], fil_a)
                    self.filaments.append(filament)
        
    @classmethod
    def from_txt(cls, filename):
        tok = TokameqFile(filename)
        xx = tok.Jpl.rr
        yy = tok.Jpl.zz
        dx = abs(xx[1] - xx[0])
        dy = abs(yy[1] - yy[0])
        currents2D = tok.Jpl.values.T*dx*dy*SI_1MA
        plasma = PlasmaTokamak(xx, yy, currents2D)
        return plasma
    
#%%
class Busbar(AnyShapeCoil):
    def __init__(self, points, nn, J, name=None):
        '''
        generates busbar from 8 pts: 4 for beginning, 4 for end

        Parameters
        ----------
        points : array-like of shape (8, 3)
            start and end pts of busbar
        n : array-like of 2 int
            amount of simple currents along 1st and 2nd axis, each >= 2
        J : float
            total current through busbar
            
        Returns
        -------
        None.

        '''
        self.filaments = []

        pts = np.asarray(points)
        st_dn = line_array(pts[0], pts[1], nn[0])
        st_up = line_array(pts[3], pts[2], nn[0])
        fn_dn = line_array(pts[4], pts[5], nn[0])
        fn_up = line_array(pts[7], pts[6], nn[0])
        for sd, su, fd, fu in zip(st_dn, st_up, fn_dn, fn_up):
            start_points  = line_array(sd, su, nn[1])
            finish_points = line_array(fd, fu, nn[1])
            for start, finish in zip(start_points, finish_points):
                
                self.filaments.append(np.array([start, finish]))
        
        self.current = J
        self.name = name
            
Group3D.call_strategies['discretize']  = JustCallStrategy()