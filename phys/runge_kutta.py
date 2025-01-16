# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:41:33 2024

@author: reonid
"""

import numpy as np
import numba

#%%
def trivial_decorator():
    def decorator(func):
        return func
    return decorator

# _numba_njit = numba.njit
_numba_njit = trivial_decorator

#%%

# dr/dt = v
# dv/dt =  q/m*(  E(r) + [v x B(r)]  )  # q_m = q/m

def f(rv, q_m, E, B):
    v = rv[3:]
    r = rv[:3]
    dv = q_m*(E(r) + np.cross(v, B(r)))
    dr = v
    return np.hstack((dr, dv))


@_numba_njit()
def runge_kutta4(rv, q_m, dt, E, B):
    dt2 = dt*0.5

    k1 = f(rv,          q_m, E, B)
    k2 = f(rv + dt2*k1, q_m, E, B)
    k3 = f(rv + dt2*k2, q_m, E, B)
    k4 = f(rv +  dt*k3, q_m, E, B)

    return rv + dt/6.0 * (k1 + 2.0 * (k2 + k3) + k4)


@_numba_njit()
def runge_kutta5(rv, q_m, dt, E, B):

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

    k1 = f(rv,          q_m, E, B)              # t
    k2 = f(rv + _1_4*k1, q_m, E, B)             # t + 0.25*dt
    k3 = f(rv + _1_8*(k1 + k2), q_m, E, B)      # t + 0.25*dt
    k4 = f(rv - _1_2*k2 +  dt*k3, q_m, E, B)    # t + 0.5*dt
    k5 = f(rv + _3_16*k1 + _9_16*k4, q_m, E, B) # t + 0.75*dt
    k6 = f(rv - _3_7 *k1 + _2_7 *k2 + _12_7*(k3 - k4) + _8_7*k5, q_m, E, B)  # t + dt

    return rv + dt/90.0 * (7.0*(k1 + k6) + 32.0*(k3 + k5) + 12.0*k4)

#%%
def common_runge_kutta4(t, dt, vector, f):  # dvector = f(t, vector)

    dt2 = dt*0.5
    k1 = f(t,       vector,        )  # t
    k2 = f(t + dt2, vector + dt2*k1)  # t + 0.5*dt
    k3 = f(t + dt2, vector + dt2*k2)  # t + 0.5*dt
    k4 = f(t + dt , vector + dt *k3)  # t + dt

    return vector + dt/6.0 * (k1 + 2.0 * (k2 + k3) + k4)


def common_runge_kutta5(t, dt, vector, f): # dvector = f(t, vector)

    _1_2  = dt*0.5
    _1_4  = dt*0.25
    _3_4  = dt*0.75
    _1_8  = dt*0.125
    _1_16 = dt*0.0625
    _1_7  = dt/7.0
    _3_16 = 3.0*_1_16
    _9_16 = 9.0*_1_16
    _2_7  = 2.0 *_1_7
    _3_7  = 3.0 *_1_7
    _8_7  = 8.0 *_1_7
    _12_7 = 12.0*_1_7

    k1 = f(t,        vector           )             # t
    k2 = f(t + _1_4, vector + _1_4*k1 )             # t + 0.25*dt
    k3 = f(t + _1_4, vector + _1_8*(k1 + k2)      ) # t + 0.25*dt
    k4 = f(t + _1_2, vector - _1_2*k2 +  dt*k3    ) # t + 0.5*dt
    k5 = f(t + _3_4, vector + _3_16*k1 + _9_16*k4 ) # t + 0.75*dt
    k6 = f(t +   dt, vector - _3_7 *k1 + _2_7 *k2 + _12_7*(k3 - k4) + _8_7*k5 )  # t + dt

    return vector + dt/90.0 * (7.0*(k1 + k6) + 32.0*(k3 + k5) + 12.0*k4)
