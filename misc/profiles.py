# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:39:38 2024

File contains different profiles e.g. bell, gauss functions.

@author: Krokhalev, Ammosov
"""
import numpy as np
import matplotlib.pyplot as plt
#%%
def gauss(a, s):
    def g(rho_loc, width=2.0):
        if abs(rho_loc) > width/2:
            return 0.
        else:
            return a * np.exp(-(rho_loc**2. / (2. * s**2.)))
    return g

def parabolic(rho_loc, width=2.0):
    if abs(rho_loc) > width/2:
        return 0.
    else:
        return (1 - 4*rho_loc**2/width**2)

def bell(rho_loc, width=2.0):
    if abs(rho_loc) > width/2:
        return 0.
    else:
        return (1 - 4*(abs(rho_loc)/width)**2.5)**5.5

def bell_wide(rho_loc, width=2.0):
    if abs(rho_loc) > width/2:
        return 0.
    else:
        return 1/(1 + 4*(abs(rho_loc)/width)**2)**(4./3.)

def bell_peaked(rho_loc, width=2.0):
    if abs(rho_loc) > width/2:
        return 0.
    else:
        return 1/(1 + 4*(0.8*abs(rho_loc)/width)**2)**6.0
    
def f_with_base(f, v_edge):
    def g(rho, width=2.0):        
        if abs(rho) > width/2:
            return 0.
        else:
            v0 = f(0.0, width)
            y = f(rho, width)
            y = v0 -  (v0 - y)*(v0 - v_edge)
            return y
    return np.vectorize(g)
#%% testing
if __name__ == "__main__":
    # plot profile
    profile = gauss(1., 1./3.)
    fig, ax = plt.subplots()
    x = np.linspace(-1, 1, 50)
    y = [profile(i) for i in x]
    ax.plot(x, y, "-o")