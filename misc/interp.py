# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:47:47 2023

@author: Eliseev_LG, Krohalev_OD
"""
#%%
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

#%%
def array_to_value(array):
    result = array[0]
    if np.all( np.isclose(array, result, atol=1e-8) ): 
        return result
    else:     
        raise Exception()
        
def FastGridInterpolator(points, values, method="linear", fill_value=np.nan, **kwargs):
    points_dim = len(points)
    values = np.asarray(values)
    is_vector = (values.ndim > points_dim)
    val_dim = values.shape[0] if is_vector else 1
        
    if (points_dim == 2) and (val_dim == 1):
        return FastGridInterpolator2D(points, values, method=method, fill_value=fill_value, **kwargs)
    if (points_dim == 2) and (val_dim == 2):
        return RegularGridVectorInterpolator2D(points, values, method=method, fill_value=fill_value, **kwargs)
    if (points_dim == 3) and (val_dim == 1):
        return FastGridInterpolator3D(points, values, method=method, fill_value=fill_value, **kwargs)
    if (points_dim == 3) and (val_dim == 2):
        raise NotImplementedError()
    if (points_dim == 3) and (val_dim == 3):
        return RegularGridVectorInterpolator3D(points, values, method=method, fill_value=fill_value, **kwargs)
    raise NotImplementedError()
        
#%% scalar interpolators
class FastGridInterpolator2D:
    def __init__(self, points, values, method="linear", fill_value=np.nan, **kwargs): 
        self.ndim = len(points)
        self.xx, self.yy = points  
        self.values = values
        self.fill_value = fill_value 
        
        self.res = np.array([self.xx[1] - self.xx[0], 
                             self.yy[1] - self.yy[0]])
        
        self.res2 = self.res[0]*self.res[1]
    
        self.lower_corner =  np.array([ self.xx[ 0], self.yy[ 0]])
        self.upper_corner  = np.array([ self.xx[-1], self.yy[-1]])

        self._reginterp = RegularGridInterpolator(points, values, method=method, 
                                                  fill_value=fill_value, bounds_error=False, **kwargs)
         

    def __call__(self,  point): 
#       !!! RegularGridInterpolator is drastically faster in this case
        if hasattr(point[0], '__iter__'):   
            return self._reginterp(point)
        
        if any(point - self.lower_corner <= 0.0) or any(self.upper_corner - point <= 0.0):  # np.logical_and, np.logical_or
            return self.fill_value # [0]

        ii = (  (point - self.lower_corner)/self.res  ) # // 1
        ii = ii.astype(int) #   ii = np.int_(ii)  #   ii = ii.astype(int, copy=False)
        
        lower_pt  = [  self.xx[ii[0]  ], self.yy[ii[1]  ]  ]
        higher_pt = [  self.xx[ii[0]+1], self.yy[ii[1]+1]  ]

        delta_x = [ higher_pt[0] - point[0],   point[0] - lower_pt[0] ] 
        delta_y = [ higher_pt[1] - point[1],   point[1] - lower_pt[1] ]  

        number = 0
        weights = [0.]*4
        for i in range(2):
            for j in range (2):
                    weights[number] = delta_x[i]*delta_y[j]/self.res2
                    number += 1
            
        #finding interpolation
        v = 0.0
        vv = self.values

        for ij in range(4):
            i = ( ij >> 1 ) % 2
            j =   ij % 2 
      
            v += weights[ij]* vv[ii[0] + i, ii[1] + j]

        return v 
    
    def contour_plot(self, filled = False, **kwargs):
        if filled:
            plt.contourf(self.xx, self.yy, self.values.T, **kwargs)
        else:
            plt.contour(self.xx, self.yy, self.values.T, **kwargs)

class FastGridInterpolator3D:
    def __init__(self, points, values, method="linear", fill_value=np.nan, **kwargs): 
        self.ndim = len(points)
        self.xx, self.yy, self.zz = points  
        self.values = values
        self.fill_value = fill_value 
        
        self.res = np.array([self.xx[1] - self.xx[0], 
                             self.yy[1] - self.yy[0],
                             self.zz[1] - self.zz[0]])
        
        self.res3 = self.res[0]*self.res[1]*self.res[2]  
    
        self.lower_corner =  np.array([ self.xx[ 0], self.yy[ 0], self.zz[ 0] ])
        self.upper_corner  = np.array([ self.xx[-1], self.yy[-1], self.zz[-1] ])

        self._reginterp = RegularGridInterpolator(points, values, method=method, 
                                                  fill_value=fill_value, bounds_error=False, **kwargs)
         

    def __call__(self,  point): 
#       !!! RegularGridInterpolator is drastically faster in this case
        if hasattr(point[0], '__iter__'):   
            return self._reginterp(point)
        
        if any(point - self.lower_corner <= 0.0) or any(self.upper_corner - point <= 0.0):  # np.logical_and, np.logical_or
            return self.fill_value # [0]

        ii = (  (point - self.lower_corner)/self.res  ) # // 1
        ii = ii.astype(int) #   ii = np.int_(ii)  #   ii = ii.astype(int, copy=False)
        
        lower_pt  = [self.xx[ii[0]  ], self.yy[ii[1]  ], self.zz[ii[2]  ]  ]
        higher_pt = [self.xx[ii[0]+1], self.yy[ii[1]+1], self.zz[ii[2]+1]  ]

        delta_x = [higher_pt[0] - point[0],   point[0] - lower_pt[0] ] 
        delta_y = [higher_pt[1] - point[1],   point[1] - lower_pt[1] ]  
        delta_z = [higher_pt[2] - point[2],   point[2] - lower_pt[2] ] 

        number = 0
        weights = [0.]*8
        for i in range(2):
            for j in range (2):
                for k in range(2):
                    weights[number] = delta_x[i]*delta_y[j]*delta_z[k]/self.res3
                    number += 1

        #finding interpolation
        v = 0.0
        vv = self.values

        for ijk in range(8):
            i = ( ijk >> 2 ) % 2
            j = ( ijk >> 1 ) % 2
            k =   ijk % 2 
      
            v += weights[ijk]* vv[ii[0] + i, ii[1] + j, ii[2] + k]
            
        return v 
    
#%% vector interpolators
class EmptyVectorInterpolator3D():
    def __call__(self, r):
        return np.full((3,), 0.)

class RegularGridVectorInterpolator2D():
    '''
    Interpolates vector on 2d grid with equal steps
    '''
    
    def __init__(self, points, values, method="linear", fill_value=np.nan, **kwargs): 
        self.ndim = len(points)
        self.xx, self.yy = points  
        self.values = values
        self.fill_value = fill_value 
        
        self.res = np.array([self.xx[1] - self.xx[0], 
                             self.yy[1] - self.yy[0]])
        
        self.res2 = self.res[0]*self.res[1]
    
        self.lower_corner =  np.array([ self.xx[ 0], self.yy[ 0]])
        self.upper_corner  = np.array([ self.xx[-1], self.yy[-1]])

    def __call__(self,  point): 
        if any(point - self.lower_corner <= 0.0) or any(self.upper_corner - point <= 0.0):  # np.logical_and, np.logical_or
            return np.full((2,), self.fill_value) # [0]

        ii = (  (point - self.lower_corner)/self.res  ) # // 1
        ii = ii.astype(int) #   ii = np.int_(ii)  #   ii = ii.astype(int, copy=False)
        
        lower_pt  = [  self.xx[ii[0]  ], self.yy[ii[1]  ]  ]
        higher_pt = [  self.xx[ii[0]+1], self.yy[ii[1]+1]  ]

        delta_x = [ higher_pt[0] - point[0],   point[0] - lower_pt[0] ] 
        delta_y = [ higher_pt[1] - point[1],   point[1] - lower_pt[1] ]  

        number = 0
        weights = [0.]*4
        for i in range(2):
            for j in range (2):
                    weights[number] = delta_x[i]*delta_y[j]/self.res2
                    number += 1
            
        #finding interpolation
        vx = 0.0
        vy = 0.0
        vvx = self.values[0]
        vvy = self.values[1]

        for ij in range(4):
            i = ( ij >> 2 ) % 2
            j = ( ij >> 1 ) % 2
      
            vx += weights[ij]* vvx[ii[0] + i, ii[1] + j]
            vy += weights[ij]* vvy[ii[0] + i, ii[1] + j]
            
        return np.array([vx, vy]) 
    
    def quiver_plot(self, thin = 1, normalize = False, **kwargs):
        if normalize:
            norm = (np.asarray(self.values[0]).T[::thin, ::thin]**2+np.asarray(self.values[1]).T[::thin, ::thin]**2)**(1/2)
            norm[norm<1e-20] = 1e-20
            plt.quiver(self.xx[::thin], self.yy[::thin], np.asarray(self.values[0]).T[::thin, ::thin]/norm, np.asarray(self.values[1]).T[::thin, ::thin]/norm, **kwargs)
        else:
            plt.quiver(self.xx[::thin], self.yy[::thin], np.asarray(self.values[0]).T[::thin, ::thin], np.asarray(self.values[1]).T[::thin, ::thin], **kwargs)
            
    def stream_plot(self, density = 2.5, **kwargs):
        plt.streamplot(self.xx, self.yy, self.values[0].T, self.values[1].T, density = density, **kwargs)


class RegularGridVectorInterpolator2Dto3D():
    def __init__(self, points, values, method="linear", fill_value=np.nan, main_axis = None, **kwargs):
        if main_axis is None:
            main_axis = np.array([1., 0., 0.])
        self.main_axis = main_axis
        self.ndim = len(points)
        self.xx, self.yy = points  
        self.values = values
        self.fill_value = fill_value 
        
        self.res = np.array([self.xx[1] - self.xx[0], 
                             self.yy[1] - self.yy[0]])
        
        self.res2 = self.res[0]*self.res[1]
    
        self.lower_corner =  np.array([ self.xx[ 0], self.yy[ 0]])
        self.upper_corner  = np.array([ self.xx[-1], self.yy[-1]])
        
    def __call__(self,  point3D): 
        z = point3D.dot(self.main_axis)
        z_vec = self.main_axis*z
        r_vec = point3D - z_vec
        r = np.linalg.norm(r_vec)
        point = np.array([z, r])
        
        if any(point - self.lower_corner <= 0.0) or any(self.upper_corner - point <= 0.0):  # np.logical_and, np.logical_or
            # return self.fill_value # [0]
            return np.full((3,), self.fill_value) # [0]

        ii = (  (point - self.lower_corner)/self.res  ) # // 1
        ii = ii.astype(int) #   ii = np.int_(ii)  #   ii = ii.astype(int, copy=False)
        
        lower_pt  = [  self.xx[ii[0]  ], self.yy[ii[1]  ]  ]
        higher_pt = [  self.xx[ii[0]+1], self.yy[ii[1]+1]  ]

        delta_x = [ higher_pt[0] - point[0],   point[0] - lower_pt[0] ] 
        delta_y = [ higher_pt[1] - point[1],   point[1] - lower_pt[1] ]  

        number = 0
        weights = [0.]*4
        for i in range(2):
            for j in range (2):
                    weights[number] = delta_x[i]*delta_y[j]/self.res2
                    number += 1
            
        #finding interpolation
        vx = 0.0
        vy = 0.0
        vvx = self.values[0]
        vvy = self.values[1]

        for ij in range(4):
            i = ( ij >> 2 ) % 2
            j = ( ij >> 1 ) % 2
      
            vx += weights[ij]* vvx[ii[0] + i, ii[1] + j]
            vy += weights[ij]* vvy[ii[0] + i, ii[1] + j]
            
        vx = vx*self.main_axis
        if np.isclose(r, 0., atol=1e-9):
            vy = np.array([0., 0., 0.])
        else:
            vy = vy*r_vec/r
        return vx + vy

class UniformGridVectorInterpolator3D():
    '''
    Interpolates vector on 3d grid with equal steps
    '''
    
    def __init__(self, grid, list_Fx_Fy_Fz, default=np.nan):
        self.grid = grid
        self.res = self.grid[:, 1, 1, 1] - self.grid[:, 0, 0, 0]
        self.res = array_to_value(self.res)
        self.list_Fx_Fy_Fz = list_Fx_Fy_Fz
        self.lower_corner = self.grid[:, 0, 0, 0]
        self.upper_corner = self.grid[:, -1, -1, -1]
        self.volume_corner2 = self.upper_corner + self.res
        self.default_value = default
    
    def __call__(self, point):
        #if point is outside the volume - return array of np.nan-s
        if any(point - self.lower_corner <= 0.0) or any(self.upper_corner - point <= 0.0):
            # return np.full((1, 3), self.default_value) # [0]
            return np.full((3,), self.default_value) # [0]
        
        #finding indexes of left corner of volume with point
        indexes_float = (point - self.lower_corner)/self.res // 1
        indexes = [[0]*3]*8
        for i in range(3):
            indexes[0][i] = int(indexes_float[i])
        
        # finding weights for all dots close to point
        '''
        delta_x = [x2 - x, x - x1]
        point = [x, y, z]
        '''
        
        i00 = indexes[0][0]
        j01 = indexes[0][1]
        k02 = indexes[0][2]
        
        left_bottom = self.grid[:, i00,     j01,     k02]        
        right_top   = self.grid[:, i00 + 1, j01 + 1, k02 + 1]
        #delta = (right_top - point,   point - left_bottom)
        
        delta_x = [right_top[0] - point[0],   point[0] - left_bottom[0] ] 
        delta_y = [right_top[1] - point[1],   point[1] - left_bottom[1] ]  
        delta_z = [right_top[2] - point[2],   point[2] - left_bottom[2] ] 
        
        res_cubic = self.res**3
        number = 0
        weights = [0.]*8
        for i in range(2):
            for j in range (2):
                for k in range(2):
                    weights[number] = delta_x[i]*delta_y[j]*delta_z[k]/res_cubic
                    number += 1
        
        #finding interpolation
        Fx = 0.0
        Fy = 0.0
        Fz = 0.0
        
        _Fx = self.list_Fx_Fy_Fz[0]
        _Fy = self.list_Fx_Fy_Fz[1]
        _Fz = self.list_Fx_Fy_Fz[2]
        for ijk in range(8):
            i = ( ijk >> 2 ) % 2
            j = ( ijk >> 1 ) % 2
            k =   ijk % 2 
            
            Fx += weights[ijk]* _Fx[i00 + i, j01 + j, k02 + k]
            Fy += weights[ijk]* _Fy[i00 + i, j01 + j, k02 + k]
            Fz += weights[ijk]* _Fz[i00 + i, j01 + j, k02 + k]

        return np.array([Fx, Fy, Fz])

class RegularGridVectorInterpolator3D:
    def __init__(self, points, values, method="linear", fill_value=np.nan, **kwargs): 
        self.ndim = len(points)
        self.xx, self.yy, self.zz = points  
        self.values = values
        self.fill_value = fill_value 
        
        self.res = np.array([self.xx[1] - self.xx[0], 
                             self.yy[1] - self.yy[0],
                             self.zz[1] - self.zz[0]])
        
        self.res3 = self.res[0]*self.res[1]*self.res[2]  
    
        self.lower_corner =  np.array([ self.xx[ 0], self.yy[ 0], self.zz[ 0] ])
        self.upper_corner  = np.array([ self.xx[-1], self.yy[-1], self.zz[-1] ])

    def __call__(self,  point): 
        
        if any(point - self.lower_corner <= 0.0) or any(self.upper_corner - point <= 0.0):  # np.logical_and, np.logical_or
            # return np.full((1, 3), self.fill_value) # [0]
            return np.full((3,), self.fill_value) # [0]

        ii = (  (point - self.lower_corner)/self.res  ) # // 1
        # print(ii)
        ii = ii.astype(int) #   ii = np.int_(ii)  #   ii = ii.astype(int, copy=False)
        # print(ii)
        lower_pt  = [self.xx[ii[0]  ], self.yy[ii[1]  ], self.zz[ii[2]  ]  ]
        higher_pt = [self.xx[ii[0]+1], self.yy[ii[1]+1], self.zz[ii[2]+1]  ]

        delta_x = [higher_pt[0] - point[0],   point[0] - lower_pt[0] ] 
        delta_y = [higher_pt[1] - point[1],   point[1] - lower_pt[1] ]  
        delta_z = [higher_pt[2] - point[2],   point[2] - lower_pt[2] ] 

        number = 0
        weights = [0.]*8
        for i in range(2):
            for j in range (2):
                for k in range(2):
                    weights[number] = delta_x[i]*delta_y[j]*delta_z[k]/self.res3
                    number += 1

        #finding interpolation
        vx = 0.0
        vy = 0.0
        vz = 0.0
        vvx = self.values[0]
        vvy = self.values[1]
        vvz = self.values[2]

        for ijk in range(8):
            i = ( ijk >> 2 ) % 2
            j = ( ijk >> 1 ) % 2
            k =   ijk % 2 
      
            vx += weights[ijk]* vvx[ii[0] + i, ii[1] + j, ii[2] + k]
            vy += weights[ijk]* vvy[ii[0] + i, ii[1] + j, ii[2] + k]
            vz += weights[ijk]* vvz[ii[0] + i, ii[1] + j, ii[2] + k]
            
        return np.array([vx, vy, vz]) 
    
#%%
class RegularGridGradientInterpolator3D():
    def __init__(self, points, values, method="linear", fill_value=np.nan, volume_size=4, **kwargs):
        self.ndim = len(points)
        self.xx, self.yy, self.zz = points  
        self.values = values
        self.fill_value = fill_value 
        
        self.dn_size = volume_size//2 - 1
        self.up_size = volume_size//2
        
        self.res = np.array([self.xx[1] - self.xx[0], 
                             self.yy[1] - self.yy[0],
                             self.zz[1] - self.zz[0]])
        
        self.res3 = self.res[0]*self.res[1]*self.res[2]  
    
        self.lower_corner =  np.array([ self.xx[ 0], self.yy[ 0], self.zz[ 0] ])
        self.upper_corner  = np.array([ self.xx[-1], self.yy[-1], self.zz[-1] ])
        self.up_border = [len(self.xx), len(self.yy), len(self.zz)]
        
    def __call__(self, point):
        if any(point - self.lower_corner <= 0.0) or any(self.upper_corner - point <= 0.0):  # np.logical_and, np.logical_or
            # return np.full((1, 3), self.fill_value) # [0]
            return np.full((3,), self.fill_value) # [0]
        
        ii = (  (point - self.lower_corner)/self.res  ) # // 1
        ii = ii.astype(int) #   ii = np.int_(ii)  #   ii = ii.astype(int, copy=False)
        
        lower_pt  = [self.xx[ii[0]  ], self.yy[ii[1]  ], self.zz[ii[2]  ]  ]
        higher_pt = [self.xx[ii[0]+1], self.yy[ii[1]+1], self.zz[ii[2]+1]  ]

        delta_x = [higher_pt[0] - point[0],   point[0] - lower_pt[0] ] 
        delta_y = [higher_pt[1] - point[1],   point[1] - lower_pt[1] ]  
        delta_z = [higher_pt[2] - point[2],   point[2] - lower_pt[2] ] 
        
        i_x, start_x = self.min_ind(ii[0])
        i_y, start_y = self.min_ind(ii[1])
        i_z, start_z = self.min_ind(ii[2])
        stop_x, stop_y, stop_z = self.max_ind(ii[0], 0), self.max_ind(ii[1], 1), self.max_ind(ii[2], 2)
       
        ii_loc = [i_x, i_y, i_z]

        grad_volume = self.values[start_x:stop_x, start_y:stop_y, start_z:stop_z]
        E = np.gradient(grad_volume, self.res[0], self.res[1], self.res[2])

        number = 0
        weights = [0.]*8
        for i in range(2):
            for j in range (2):
                for k in range(2):
                    weights[number] = delta_x[i]*delta_y[j]*delta_z[k]/self.res3
                    number += 1

        #finding interpolation
        vx = 0.0
        vy = 0.0
        vz = 0.0
        vvx = E[0]
        vvy = E[1]
        vvz = E[2]

        for ijk in range(8):
            i = ( ijk >> 2 ) % 2
            j = ( ijk >> 1 ) % 2
            k =   ijk % 2 
            
            vx += weights[ijk]* vvx[ii_loc[0] + i, ii_loc[1] + j, ii_loc[2] + k]
            vy += weights[ijk]* vvy[ii_loc[0] + i, ii_loc[1] + j, ii_loc[2] + k]
            vz += weights[ijk]* vvz[ii_loc[0] + i, ii_loc[1] + j, ii_loc[2] + k]
            
        return np.array([vx, vy, vz]) 
        
    def min_ind(self, i):
        if i > self.dn_size:
            return self.dn_size, i - self.dn_size
        return i, 0
        
    def max_ind(self, i, axis):
        if self.up_border[axis] - i > self.up_size:
            return i + 1 + self.up_size
        return self.up_border[axis]
    