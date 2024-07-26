# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:20:23 2023
"""
#%%
import numpy as np

#%% Rad
PI = 3.1415926535897932384626433832795
deg = PI/180.0
mRad = 0.001

#%% SI
SI_AEM     = 1.6604e-27       #     {kg}
SI_e       = 1.6021e-19       #     {C}
SI_Me      = 9.1091e-31       #     {kg}      // mass of electron
SI_Mp      = 1.6725e-27       #     {kg}      // mass of proton
SI_M_Tl    = 204*SI_AEM       #     {kg}      // mass of Talium ion
SI_M_Cs    = 133*SI_AEM       #     {kg}      // mass of Cesium ion
SI_c       = 2.9979e8         #     {m/sec}   // velocity of light
SI_1eV     = SI_e             #     {J}
SI_1keV    = SI_1eV*1000.0    #     {J}
SI_1MA     = 1e6              #     {A}
SI_mu0     = 4.0*np.pi*1e-7   #     {H/A}
SI_eps0    = 8.8e-12          #     {F/m}
SI_m       = 1.0              #     {m}
SI_cm      = 1e-2             #     {m}
SI_mm      = 1e-3             #     {m}
SI_V       = 1.0              #     {V}
SI_kV      = 1e3              #     {V}

#%%
SI_mm_eps0 = SI_eps0*SI_mm    #     {F/mm}
SI_mm_c    = SI_c*SI_mm       #     {mm/sec}

#%%
MAGFIELD_COEFF = SI_mu0/(4.0*np.pi)
