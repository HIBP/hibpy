# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:48:02 2024

@author: Ammosov, Krokhalev
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod, abstractproperty

from ..phys.constants import mRad, PI
from ..geom.prim import HollowCircle3D
from ..geom.geom import (plot_point, pt3D, vec3D, rotateMx, transformMx,
                             normalizedVector, vNorm)
#%% definitions
class AbstractLatticeMarkup(ABC):
    """
    Example commands:
        
        Creating lattice:
            lattice = CircleLatticeMarkup(d_beam, div_angle*mRad, n, m)
            
        Plotting:
            lattice.plot() - plot lattice
            
            lattice.plot3d() - plot 3D lattice with div_angle arrows.
            NOTICE: This method works after using "__call__" method of lattice
            or after fatbeam initialization (creating fatbeam object).
            
        Misc. commands:
            len(lattice) - get amount of filaments
            
    """
    
    def __call__(self, rv0):
        """
        Creates grid of fatbeam initial dots.
        
        Parameters
        ----------
        rv0 : np.array, shape==(6,)

        Returns
        -------
        rv0_list : list [np.array, shape==(6,)].

        """
        
        rv0_list = []
        r0, v0 = rv0[:3], rv0[3:]
        basis_dest = np.array([vec3D(1, 0, 0),
                               vec3D(0, 1, 0),
                               vec3D(0, 0, 1)])
        
        bx = normalizedVector(v0)
        by = np.cross(bx, basis_dest[2]) # [x_src X z_dest] = y_src
        bz = np.cross(bx, by)
        basis_src = np.array([bx,
                              by,
                              bz])
        
        transform_mx = transformMx(basis_src, basis_dest)
        
        for point in self.grid:
            pt = transform_mx.dot(point)
            pt += r0
            if not all(np.isclose(pt, r0)):
                rot_axis = np.cross(v0, pt - r0)
                mx = rotateMx(rot_axis, 
                              self.div_angle * vNorm(pt - r0) * 2. / self.d_beam)
                v = mx.dot(v0)
            else:
                v = v0
                
            rv = np.hstack((pt, v))
            rv0_list.append(rv)
        
        self._rv_list = rv0_list
        return rv0_list
    
    @abstractproperty
    def div_angle(self):
        """
        Divergency angle in radians.

        Returns
        -------
        div_angle : float.

        """
    
    @abstractproperty
    def grid(self):
        """
        List of points in default coordinate system.
        
        Returns
        -------
        grid : list [np.array, shape==(3,)].

        """
    
    @abstractproperty
    def rv_list(self):
        """
        List of points in new basis in rv0 location.
        
        Returns
        -------
        rv0_list : list [np.array, shape==(6,)].

        """
    
    @abstractmethod
    def plot(self, axes_code='ZY', **kwargs):
        """
        Plot lattice
        
        Returns
        -------
        None.

        """
    
    def get_weights(self, profile):
        """
        Provides weights of points according to given profile.
        
        Returns
        -------
        wights : list [float].

        """
        weights = np.array([profile(2 * np.linalg.norm(x) / self.d_beam) for x in self.grid])
        return weights / sum(weights)
    
    def plot3d(self, **kwargs):
        """
        Plot lattice in 3D
        
        Returns
        -------
        None.

        """
        
        if len(self._rv_list) == 0:
            print("Before plot3d use \"call\" method first.")
            return None
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        rv_list = np.asarray(self._rv_list)
        ax.scatter(rv_list[:, 0], rv_list[:, 1], rv_list[:, 2], color='red')
        ax.quiver(rv_list[:, 0], rv_list[:, 1], rv_list[:, 2],
                  rv_list[:, 3], rv_list[:, 4], rv_list[:, 5],
                  length=1e-9, arrow_length_ratio=0.2)
        
        # Set labels and show the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_aspect('equal')
        plt.show()
    
    def __len__(self):
        return len(self._grid)
    
    @abstractmethod
    def __repr__():
        """
        Prepresentation to console.
        
        Returns
        -------
        None.

        """
        
#%%
class SquaredLatticeMarkup(AbstractLatticeMarkup):
    def __init__(self, d_beam, div_angle, n):
        """
        
        Creates squared grid of initial points for fatbeam.
        
        Parameters
        ----------
        d_beam : float
            Grid diameter.
        n : integer
            Amount of dots along vertical/horizontal diameters of squared grid.
            Odd values are prefered.
        div_angle : float
            Divergency angle in radians.

        Returns
        -------
        None.

        """
        self.d_beam = d_beam
        self.n = n
        self._div_angle = div_angle
        self._grid = []
        self._rv_list = []
        self.circle = HollowCircle3D(pt3D(0, 0, 0), self.d_beam*0.5, self.d_beam*0.5, vec3D(1, 0, 0))
        
        # if only 1 central dot
        if n == 1:
            self._grid.append(pt3D(0, 0, 0))
            return None
        
        # create full rectangle grid of points
        r = self.d_beam * 0.5
        x = np.linspace(-r, r, int(n))
        X, Y = np.meshgrid(x, x)
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        # take points, which inside the circle with radius r
        for i in range(len(positions[0])):
            point = pt3D(0., positions[0][i], positions[1][i])
            if np.linalg.norm(point) <= r:
                self._grid.append(point)
    
    @property 
    def div_angle(self):
        return self._div_angle
    
    @property 
    def grid(self):
        return self._grid
    
    def rv_list(self):
        return self._rv_list
    
    def plot(self, axes_code='ZY', **kwargs):
        self.circle.plot(axes_code=axes_code, **kwargs)
        for pt in self._grid:
            plot_point(pt, axes_code=axes_code, **kwargs)
        plt.axis('equal')
        plt.show()
        
    def __repr__(self):
        repr_str = f"Type: SquaredLatticeMarkup\nd_beam = {self.d_beam}\ndiv_angle \
= {self.div_angle}\nn = {self.n}\nfilaments: {len(self.grid)}\n"
        return repr_str

class LineLatticeMarkup(AbstractLatticeMarkup):
    def __init__(self, d_beam, div_angle, n, angle=0):
        """
        
        Creates one straight line grid of initial points for fatbeam with given angle.
        DO NOT confuse div_angle with angle.
        
        Parameters
        ----------
        d_beam : float
            Grid diameter.
        n : integer
            Amount of dots along vertical/horizontal diameters of squared grid.
            Odd values are prefered.
        div_angle : float
            Divergency angle in radians.
        angle : float
            Angle for Line in radians. Rotates generated straight line on this angle.
            PI/2 by default which means Line is in Plane[x_src, z_dest],
            PI/2 [rad] is vertical Line and 0 [rad] is horizontal Line.
                
        Returns
        -------
        None.

        """
        self.d_beam = d_beam
        self.n = n
        self._div_angle = div_angle
        self._grid = []
        self._rv_list = []
        self.angle = angle
        self.circle = HollowCircle3D(pt3D(0, 0, 0), self.d_beam*0.5, self.d_beam*0.5, vec3D(1, 0, 0))
        
        # if only 1 central dot
        if n == 1:
            self._grid.append(pt3D(0, 0, 0))
            return None
        
        # create one strip/straight line of points
        r = self.d_beam * 0.5
        x = np.zeros(int(n))
        y = np.zeros(int(n))
        z = np.linspace(-r, r, int(n))
        positions = np.vstack((x, y, z,)).T
        
        # rotate Line by angle
        rot_axis = np.cross(vec3D(0, 0, 1), vec3D(0, 1, 0)) # [z X y]
        mx = rotateMx(rot_axis, self.angle)
        for point in positions:
            self._grid.append(mx.dot(point))
    
    @property 
    def div_angle(self):
        return self._div_angle
    
    @property 
    def grid(self):
        return self._grid
    
    def rv_list(self):
        return self._rv_list
    
    def plot(self, axes_code='ZY', **kwargs):
        self.circle.plot(axes_code=axes_code, **kwargs)
        for pt in self._grid:
            plot_point(pt, axes_code=axes_code, **kwargs)
        plt.axis('equal')
        plt.show()

    def __repr__(self):
        repr_str = f"Type: LineLatticeMarkup\nd_beam = {self.d_beam}\ndiv_angle \
= {self.div_angle}\nn = {self.n}\nfilaments: {len(self.grid)}\n"
        return repr_str
    
class CircleLatticeMarkup(AbstractLatticeMarkup):
    def __init__(self, d_beam, div_angle, n, m):
        """
        
        Creates one straight line grid of initial points for fatbeam with given angle.
        DO NOT confuse div_angle with angle.
        
        Parameters
        ----------
        d_beam : float
            Grid diameter.
        div_angle : float
            Divergency angle in radians.
        n : integer
            Amount of dots on external circle.
            Odd values are prefered.
        m : integer
            Amount of concentric circles.
            If m == 1: only one outer circle (with d_beam diameter) and central dot.
                
        Returns
        -------
        None.

        """
        self.d_beam = d_beam
        self.n = n
        self._div_angle = div_angle
        self._grid = []
        self._rv_list = []
        self.m = m
        
        # circle for printing
        self.circle = HollowCircle3D(pt3D(0, 0, 0), self.d_beam*0.5, self.d_beam*0.5, vec3D(1, 0, 0))
        
        # if only 1 central dot
        if n == 1:
            self._grid.append(pt3D(0, 0, 0))
            return None
        
        # create one strip/straight line of points
        r = self.d_beam * 0.5
        positions = []
        circle_radiuses = np.linspace(-r, 0, int(self.m), endpoint=False)
        
        for cr in circle_radiuses:
            j = round(-n * cr/r, 0)
            pos = self.create_circle(cr, j)
            for val in pos:
                positions.append(val)
        
        for point in positions:
            self._grid.append(point)
        self._grid.append(pt3D(0, 0, 0)) # add central dot
    
    def create_circle(self, r, n):
        x = []
        y = []
        z = []
        theta_zero = 360.0 / n
        for i in range(int(n)):
            x.append(0.0)
            z.append(r * np.cos(i * theta_zero / 180.0 * PI))
            y.append(r * np.sin(i * theta_zero / 180.0 * PI))
        positions = np.vstack((x, y, z,)).T
        return positions
    
    @property 
    def div_angle(self):
        return self._div_angle
    
    @property 
    def grid(self):
        return self._grid
    
    def rv_list(self):
        return self._rv_list
    
    def plot(self, axes_code='ZY', **kwargs):
        self.circle.plot(axes_code=axes_code, **kwargs)
        for pt in self._grid:
            plot_point(pt, axes_code=axes_code, **kwargs)
        plt.axis('equal')
        plt.show()

    def __repr__(self):
        repr_str = f"Type: CircleLatticeMarkup\nd_beam = {self.d_beam}\ndiv_angle \
= {self.div_angle}\nn = {self.n}\nm = {self.m}\nfilaments: {len(self.grid)}\n"
        return repr_str
#%% tests
if __name__=='__main__':
    
    def plot_rv_list_3d(rv_list):
        # Plot the 3D surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        rv_list = np.asarray(rv_list)
        ax.scatter(rv_list[:, 0], rv_list[:, 1], rv_list[:, 2], color='red')
        ax.quiver(rv_list[:, 0], rv_list[:, 1], rv_list[:, 2],
                  rv_list[:, 3], rv_list[:, 4], rv_list[:, 5],
                  length=0.01, arrow_length_ratio=0.1)
        
        # Set labels and show the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_aspect('equal')
        plt.show()

    def test_squared_lattice_markup():
        d_beam = 0.01
        n = 15
        div_angle = 10
        
        grid = SquaredLatticeMarkup(d_beam, div_angle*mRad, n)
        rv0 = np.array([1., 0., 0., 1.0, 0., 0])
        rv0_list = grid(rv0)
        return grid, rv0_list
    
    def test_line_lattice_markup():
        d_beam = 0.01
        n = 3
        div_angle = 0
        angle = PI * 0.5
        
        grid = LineLatticeMarkup(d_beam, div_angle*mRad, n, angle=angle)
        rv0 = np.array([1., 0., 0., 1.0, 0., 0])
        rv0_list = grid(rv0)
        return grid, rv0_list
    
    def test_circle_lattice_markup():
        d_beam = 0.01
        div_angle = 10
        n = 15
        m = 3
        
        grid = CircleLatticeMarkup(d_beam, div_angle*mRad, n, m)
        rv0 = np.array([1., 0., 0., 1.0, 0., 0])
        rv0_list = grid(rv0)
        return grid, rv0_list
    
    # Squared Lattice Test
    # grid, rv0_list = test_squared_lattice_markup()
    
    # Line Lattice Test
    # grid, rv0_list = test_line_lattice_markup()
    
    # Circle Lattice Test
    grid, rv0_list = test_circle_lattice_markup()
    
    # grid.plot(color='blue')
    # print(len(grid.grid), grid.grid)
    # plot_rv_list_3d(rv0_list)
    grid.plot3d()