# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:55:08 2024

@author: reonid

Optimization of calculation of xxx_field_on_grid
using information about gabarits

"""

import numpy as np

from .grid import scalar_field_on_grid, vector_field_on_grid, _xx_yy_etc_from_grid, _field_from_raw_repr

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



def scalar_field_on_grid_ex(grid, func, subidx=None, gabarits=None, outvalue=0.0): 
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


def vector_field_on_grid_ex(grid, func, subidx=None, gabarits=None, outvalue=0.0): 
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


