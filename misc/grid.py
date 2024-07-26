# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:13:31 2023

@author: reonid

'G-form'
Standard grid shape (3, Nx, Ny, Nz) - due to compatibility with 
  np.mgrid

'B-form'
Field3D has another shape (Nx, Ny, Nz, 3) - due to compatibility with 
  Philipp's magfieldPF*.npy files

Actually 'G-form' is more convenient for fields: in case of 'G-form'
  Bx, By, Bz = B[0], B[1], B[2]

Raw representation for both cases is array of points with shape (Nx*Ny*Nz, 3)

"""
#%%
import numpy as np
#import matplotlib.pyplot as plt

try:
    import multiprocessing as mp
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except:
    JOBLIB_AVAILABLE = False

#%%
def _mgrid(*xx_yy_etc): # _mgrid(xx, yy), _mgrid(xx, yy, zz)
    ndim = len(xx_yy_etc)
    xNd_yNd_etc = np.meshgrid(*xx_yy_etc, indexing='ij') # x2d_y2d, x2d_y3d_z3d
    
    gridshape = [ndim] + list( xNd_yNd_etc[0].shape )   #gridshape.insert(0, ndim)
    grid = np.empty( tuple(gridshape) )

    for i, coord_array in enumerate(xNd_yNd_etc): 
        #grid[i, :, :, :] = coord_array
        #grid[i, :, :] = coord_array
        grid[i, ...] = coord_array
        
    return grid

#%%
def _xx_yy_etc_from_grid(grid):
    if grid.shape[0] == 2:
        return [grid[0, :, 0], 
                grid[1, 0, :]]
    if grid.shape[0] == 3:
        return [grid[0, :, 0, 0], 
                grid[1, 0, :, 0], 
                grid[2, 0, 0, :]]
    
#%%    
def _grid_to_raw_points(grid):  # "G-form" grid with shape (3, Nx, Ny, Nz)
    #return np.vstack(map(np.ravel, grid)).T  # drastically slower
    pt_dim = grid.shape[0]
    N = np.prod(grid.shape[1:])
    return grid.reshape(pt_dim, N).swapaxes(0, 1)  # -> (Nx*Ny*Nz, 3) 

def _grid_from_raw_points(points, main_shape):
    pt_dim = points.shape[-1]
    return points.swapaxes(0, 1).reshape( (pt_dim,) + main_shape ) # -> (3, Nx, Ny, Nz) 

#%% Vector field
def _field_to_raw_repr(field): # "B-form" field with shape (Nx, Ny, Nz, 3)
    pt_dim = field.shape[-1]
    N = np.prod(field.shape[0:-1])
    return field.reshape(N, pt_dim) # -> (Nx*Ny*Nz, 3) 
        

def _field_from_raw_repr(raw_field, main_shape):  
    pt_dim = raw_field.shape[-1]     
    return raw_field.reshape(main_shape + (pt_dim,)) # -> (Nx, Ny, Nz, 3) 


def _field_to_G_repr(field): # "B-form" field with shape (Nx, Ny, Nz, 3)
    main_shape = field.shape[0:-1]
    raw_field = _field_to_raw_repr(field)    
    return _grid_from_raw_points(raw_field, main_shape) # -> (3, Nx, Ny, Nz) 

#%%
def vector_field_on_grid(grid, func):
    points = _grid_to_raw_points(grid)
    _vv = func_on_points(points, func)
    vv = _field_from_raw_repr(_vv, grid.shape[1:])
    return vv

def scalar_field_on_grid(grid, func):
    points = _grid_to_raw_points(grid)
    _ss = func_on_points(points, func)
    return _ss.reshape(grid.shape[1:])

def func_on_points(points, func):
    # multiprocessing
    n_workers = mp.cpu_count() - 1 
    vv = Parallel (n_jobs=n_workers) (delayed(func)(r) for r in points) 
    return np.array(vv)

#%% speed efficient functions
def gabarits2subidx(grid, gabarits): 
    # Используется Биеарный поиск O(ln(N)). 
    # В принципе можно сделать расчетным O(1), но нужно ли? 
    
    result = [slice(None, None)] # Первый элемент в G-form shape - 2D/3D, мы его добавляем заранее в виде [:] 
    xx_yy_etc = _xx_yy_etc_from_grid(grid) 
    for k in range( len(xx_yy_etc) ): # 2D/3D
        vv = xx_yy_etc[k]
        vmin = gabarits[0][k]
        vmax = gabarits[1][k]

        i0 = max(np.searchsorted(vv, vmin) - 1,   0       ) 
        i1 = min(np.searchsorted(vv, vmax) + 2,   len(vv) ) # ??? выход за границы игнорируется, проверка в принципе не нужна

        result.append(slice(i0, i1))
    return tuple(result)

def fast_scalar_field_on_grid(grid, func, subidx=None, gabarits=None, outvalue=0.0): 
    '''
    Applies func to grid. Allows speeding by calculating only for the part of the grid.

    Parameters
    ----------
    grid : np.meshgrid
    func : any callable with one argument - point
    subidx : indexing, optional
        Grid indexes in which calculation is needed. The default is None.
    gabarits : tuple of two points, optional
        Gabarits in which calculation is needed. The default is None.
    outvalue : same type as func(r) output, optional
        This value will fill all grid outside given gabarits or subidx. The default is 0.0.

    '''
    if subidx is None: 
        if gabarits is None: 
            return  scalar_field_on_grid(grid, func)
        else: 
            subidx = gabarits2subidx(grid, gabarits)

    res_shape = grid.shape[1:] # Первый элемент в G-form shape - 2D/3D, мы его отрезаем. 
                               # То, что называется main_shape (domain_shape???)
    result = np.full(res_shape, outvalue)  

    subgrid = grid[subidx] 
    subresult = scalar_field_on_grid(subgrid, func)
    
    res_subidx = subidx[1:]
    result[res_subidx] = subresult
    return result

def fast_vector_field_on_grid(grid, func, subidx=None, gabarits=None, outvalue=0.0): 
    '''
    Same as fast_scalar_field_on_grid, but func can return vectors
    '''
    if subidx is None: 
        if gabarits is None: 
            return  vector_field_on_grid(grid, func)
        else: 
            subidx = gabarits2subidx(grid, gabarits)

    #res_shape = grid.shape[1:] + (outvalue.shape[0], ) # B-form
    raw_result = np.zeros(  ( np.prod(grid.shape[1:]), outvalue.shape[0])  ) # raw form (R-form???)
    raw_result[:, :] = outvalue
    result = _field_from_raw_repr(raw_result, grid.shape[1:])

    subgrid = grid[subidx] 
    subresult = vector_field_on_grid(subgrid, func)
    
    res_subidx = subidx[1:] + (slice(None, None), ) # B-form 
    result[res_subidx] = subresult
    return result

#%%
    
class Grid: 
    def __init__(self, mgrid): 
        self.grid = mgrid

    def __getitem__(self, obj): 
        return self.grid[obj]
    
    @property 
    def shape(self): 
        return self.grid.shape
    
    def as_raw_points(self): 
        return _grid_to_raw_points(self.grid)
    
    @classmethod
    def from_indexing(cls, obj): 
        _grid = np.mgrid[obj]
        return cls(_grid)
 
    @classmethod
    def from_domain(cls, lower_corner, upper_corner, resolution): 
        # create grid of points
        lower_corner = np.asarray(lower_corner)
        upper_corner = np.asarray(upper_corner)
        # upper_corner += resolution
        upper_corner += resolution*0.001
        if lower_corner.shape[0] == 3:
            _grid = np.mgrid[lower_corner[0]:upper_corner[0]:resolution,
                             lower_corner[1]:upper_corner[1]:resolution,
                             lower_corner[2]:upper_corner[2]:resolution]
            
            # assert ( all(upper_corner == _grid[:, -1, -1, -1]) )
            
        if lower_corner.shape[0] == 2:
            _grid = np.mgrid[lower_corner[0]:upper_corner[0]:resolution,
                             lower_corner[1]:upper_corner[1]:resolution]
            
            # assert ( all(upper_corner == _grid[:, -1, -1]) )
        
        return cls(_grid)
    
    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            lower_corner = np.array([float(i) for i in f.readline().split()[0:3]])
            upper_corner = np.array([float(i) for i in f.readline().split()[0:3]])
            resolution = float(f.readline().split()[0])
        return cls.from_domain(lower_corner, upper_corner, resolution)
    
    #def from_raw_points(cls, points)

    def calc_raw_field(self, func, parallel=False): 
        points = self.as_raw_points()
        
        if parallel and JOBLIB_AVAILABLE: 
            n_workers = mp.cpu_count() - 1
            raw_field = Parallel (n_jobs=n_workers) (delayed(func)(r) for r in points)            
        else: 
            raw_field = [func(r) for r in points]

        return np.array(raw_field)

    def calc_vector_field(self, func, parallel=False): 
        raw_field = self.calc_raw_field(func, parallel)
        return _field_from_raw_repr(raw_field, self.grid.shape[1:])

    def calc_scalar_field(self, func, parallel=False): 
        raw_field = self.calc_raw_field(func, parallel)
        return raw_field.reshape( self.grid.shape[1:] )

    def as_points(self):
        if self.grid.shape[0] == 3:
            lower_corner = self.grid[:, 0, 0, 0]
            upper_corner = self.grid[:, -1, -1, -1]
            res3d = self.grid[:, 1, 1, 1] - self.grid[:, 0, 0, 0]
            x = np.arange(lower_corner[0], upper_corner[0] + res3d[0]*0.01, res3d[0])
            y = np.arange(lower_corner[1], upper_corner[1] + res3d[1]*0.01, res3d[1])
            z = np.arange(lower_corner[2], upper_corner[2] + res3d[2]*0.01, res3d[2])
            
            return (x, y, z)
        
        if self.grid.shape[0] == 2:
            lower_corner = self.grid[:, 0, 0]
            upper_corner = self.grid[:, -1, -1]
            res2d = self.grid[:, 1, 1] - self.grid[:, 0, 0]
            x = np.arange(lower_corner[0], upper_corner[0] + res2d[0]*0.01, res2d[0])
            y = np.arange(lower_corner[1], upper_corner[1] + res2d[1]*0.01, res2d[1])
            
            return (x, y)
        
#%%
def resolution(grid):
    res = None
    if len(grid.shape) == 2:
        res = abs(grid[0, 1] - grid[0, 0])
    if len(grid.shape) == 3:
        res  = abs(grid[0, 1, 0] - grid[0, 0, 0])    
        res1 = abs(grid[1, 0, 1] - grid[1, 0, 0])
        if not np.isclose(res, res1):
            raise Exception('grid must be regular')
    if len(grid.shape) == 4:
        res  = abs(grid[0, 1, 0, 0] - grid[0, 0, 0, 0])    
        res1 = abs(grid[1, 0, 1, 0] - grid[1, 0, 0, 0])
        res2 = abs(grid[2, 0, 0, 1] - grid[2, 0, 0, 0])
        if not (np.isclose(res, res1) and np.isclose(res, res2)):
            raise Exception('grid must be regular')
    return res

