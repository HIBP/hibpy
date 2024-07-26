# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:38:37 2023

@author: Krohalev_OD
"""
#%%
import numba
import numpy as np

#%%
def trivial_decorator(): 
    def decorator(func): 
        return func
    return decorator
    

_numba_njit = numba.njit
# _numba_njit = trivial_decorator

#%%
@_numba_njit()
def f(k, E, V, B):
    return k*(E + np.cross(V, B))

@_numba_njit()
def runge_kutta(k, RV, dt, E, B):
    '''
    Calculate one step using Runge-Kutta algorithm

    V' = k(E + [VxB]) == K(E + np.cross(V,B)) == f
    r' = V == g

    V[n+1] = V[n] + (h/6)(m1 + 2m2 + 2m3 + m4)
    r[n+1] = r[n] + (h/6)(k1 + 2k2 + 2k3 + k4)
    m[1] = f(t[n], V[n], r[n])
    k[1] = g(t[n], V[n], r[n])
    m[2] = f(t[n] + (h/2), V[n] + (h/2)m[1], r[n] + (h/2)k[1])
    k[2] = g(t[n] + (h/2), V[n] + (h/2)m[1], r[n] + (h/2)k[1])
    m[3] = f(t[n] + (h/2), V[n] + (h/2)m[2], r[n] + (h/2)k[2])
    k[3] = g(t[n] + (h/2), V[n] + (h/2)m[2], r[n] + (h/2)k[2])
    m[4] = f(t[n] + h, V[n] + h*m[3], r[n] + h*k[3])
    k[4] = g(t[n] + h, V[n] + h*m[3], r[n] + h*k[3])

    Parameters
    ----------
    k : float
        particle charge [Co] / particle mass [kg]
    RV : np.array([[x, y, z, Vx, Vy, Vz]])
        coordinates and velocities array [m], [m/s]
    dt : float
        timestep [s]
    E : np.array([Ex, Ey, Ez])
        values of electric field at current point [V/m]
    B : np.array([Bx, By, Bz])
        values of magnetic field at current point [T]

    Returns
    -------
    RV : np.array([[x, y, z, Vx, Vy, Vz]])
        new coordinates and velocities

    '''
    if np.any(np.isnan(B)): 
        print('NaN!!  B = ', B)
        print('   RV = ', RV)
    
    if np.any(np.isnan(E)): 
        print('NaN!!  E = ', E)
        print('   RV = ', RV)
    
    r = RV[:3]
    V = RV[3:]

    m1 = f(k, E, V, B)
    k1 = V #g(V)

    fV2 = V + (dt / 2.) * m1
    gV2 = V + (dt / 2.) * m1
    m2 = f(k, E, fV2, B)
    k2 = gV2 # g(gV2)

    fV3 = V + (dt / 2.) * m2
    gV3 = V + (dt / 2.) * m2
    m3 = f(k, E, fV3, B)
    k3 = gV3 # g(gV3)

    fV4 = V + dt * m3
    gV4 = V + dt * m3
    m4 = f(k, E, fV4, B)
    k4 = gV4 # g(gV4)

    V = V + (dt / 6.) * (m1 + (2. * m2) + (2. * m3) + m4)
    r = r + (dt / 6.) * (k1 + (2. * k2) + (2. * k3) + k4)

    result = np.zeros((6,))
    result[:3] = r
    result[3:] = V
    return result