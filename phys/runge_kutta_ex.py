# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:41:33 2024

@author: reonid
"""

import numpy as np
#import numpy.random as rand
import matplotlib.pyplot as plt


#%%
def trivial_decorator(): 
    def decorator(func): 
        return func
    return decorator
    

#_numba_njit = numba.njit
_numba_njit = trivial_decorator

#%%
#q_m = q/m
def f(rv, q_m, E, B):
    v = rv[3:]
    r = rv[:3]
    dv = q_m*(E(r) + np.cross(v, B(r)))
    dr = v
    #return np.vstack((dr, dv))
    return np.hstack((dr, dv))


@_numba_njit()
def runge_kutta4(rv, q_m, dt, E, B):


#    if np.any(np.isnan(B)): 
#        print('NaN!!  B = ', B); print('   RV = ', RV)
#    
#    if np.any(np.isnan(E)): 
#        print('NaN!!  E = ', E); print('   RV = ', RV)

    dt2 = dt*0.5

    k1 = f(rv,          q_m, E, B)
    # print(k1)
    k2 = f(rv + dt2*k1, q_m, E, B)
    k3 = f(rv + dt2*k2, q_m, E, B)
    k4 = f(rv +  dt*k3, q_m, E, B)

    _rv = rv + dt/6.0 * (k1 + 2.0 * (k2 + k3) + k4)
    return _rv


@_numba_njit()
def runge_kutta5(rv, q_m, dt, E, B):

#    if np.any(np.isnan(B)): 
#        print('NaN!!  B = ', B); print('   RV = ', RV)
#    
#    if np.any(np.isnan(E)): 
#        print('NaN!!  E = ', E); print('   RV = ', RV)

    _1_2  = dt*0.5
    _1_4  = dt*0.25
    _1_8  = dt*0.125
    _1_16 = dt*0.0625
    _1_7 =  dt/7.0
    
    _3_16 = 3.0*_1_16
    _9_16 = 9.0*_1_16

    _2_7  = 2.0 *_1_7
    _3_7  = 3.0 *_1_7
    _8_7  = 8.0 *_1_7
    _12_7 = 12.0*_1_7

    k1 = f(rv,          q_m, E, B)
    k2 = f(rv + _1_4*k1, q_m, E, B)
    
    k3 = f(rv + _1_8*(k1 + k2), q_m, E, B)
    k4 = f(rv - _1_2*k2 +  dt*k3, q_m, E, B)

    k5 = f(rv + _3_16*k1 + _9_16*k4, q_m, E, B)
    k6 = f(rv - _3_7 *k1 + _2_7 *k2 + _12_7*(k3 - k4) + _8_7*k5, q_m, E, B)

    _rv = rv + dt/90.0 * (7.0*(k1 + k6) + 32.0*(k3 + k5) + 12.0*k4)
    return _rv


