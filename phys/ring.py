# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:15:57 2023

@author: reonid
"""
#%%
#import sys
#hibplib_path = 'D:\\reonid\\myPy\\reonid-packages'
#if hibplib_path not in sys.path: sys.path.append(hibplib_path)

import numpy as np
import scipy.special as sp

import matplotlib.pyplot as plt
import math as math
#import types as types

from ..geom.geom import (invMx, stdRotateMx, vNorm, vec3D, mxScaleCoeff, 
                         _regularPolygon3D, plot_polygon, normalizedVector) #, line_array,

#%%
normalized_vector = normalizedVector

SI_mu0 = 4*np.pi*1e-7

sin = math.sin #np.sin
cos = math.cos #np.cos  # ??? math
atan2 = math.atan2


def _magneticFdOfRing(a, J, r, z): # !!! Gaussian 
    '''
    Magnetic field of infinitely thin ring
    a - radius [cm]
    (r, phi, z) - cillindrical coordinates
    J current [A]
    B [Gs] 
    Landau, Lifschitz VIII, chapter IV (p.164, edition 1982)
    '''
    aa = a*a # a**2
    zz = z*z # z**2
    #if r == 0.0 : # ???  if np.isclose(r, 0.0): 
    if  r < 1e-8:
        return [0.0, 0.0, J*np.pi/5*aa*(aa + zz)**-1.5]

    rr = r*r # r**2
    aplus = (a + r)**2 + zz
    aminus = (a - r)**2 + zz
    rrzz = rr + zz
    sqrtaplus = aplus**0.5
    arg = 4.0*a*r/aplus
    K = sp.ellipk(arg)
    E = sp.ellipe(arg)
    E_aminus = E/aminus
    J1 = J*0.2
    Hr = J1*z*((aa + rrzz)*E_aminus-K)/(r*sqrtaplus)      
    Hz = J1*(K+(aa - rrzz)*E_aminus)/sqrtaplus             
    return [Hr, 0.0, Hz]   

def _magneticPsiOfRing(a, J, r, z):
    '''
    Poloidal magnetic flux of infinitely thin ring of radius a, z=0
    through ring with r, z
    
    a - radius [cm]
    (r, phi, z) - cillindrical coordinates
    J current [A]
    '''
    aa = a*a # a**2
    zz = z*z # z**2
    #if r == 0.0 : # ???  if np.isclose(r, 0.0): 
    if  r < 1e-8:
        return (np.pi*r**2)*J*np.pi/5*aa*(aa + zz)**-1.5

    aplus = (a + r)**2 + zz
    arg = 4.0*a*r/aplus
    K = sp.ellipk(arg)
    E = sp.ellipe(arg)
    J1 = J*0.2
    psi = 4*np.pi*J1*np.sqrt(a*r/arg)*((1 - 0.5*arg)*K - E)
    return psi

def _magneticFdOfThickRing(a, J, r, z, wire_radius): # !!! Gaussian 
    '''
    Magnetic field of thick ring 
    a - radius [cm]
    (r, phi, z) - cillindrical coordinates
    J current [A]
    wire_radius - thickness [cm]
    B [Gs] 
    Modified version of the [ Landau, Lifschitz VIII, chapter IV ]
    '''
    
    aa = a*a # a**2
    zz = z*z # z**2
    if  r < 1e-8:     # on-axis
        return [0.0, 0.0, J*np.pi/5*aa*(aa + zz)**-1.5]

    #aplus = (a + r)**2 + zz
    aminus = (a - r)**2 + zz
    wr2 = wire_radius*wire_radius

    # if aminus < wr2*0.000001: # !!!
    #     #return [0.0, 0.0, 0.0]
    #     Hr0, _, Hz0 = _magneticFdOfRing(a, J, 0.0, 0.0)

    #     return [0.6*Hr0, 0.0, 0.6*Hz0]
    # elif aminus < wr2: 

    #     dist = aminus**0.5
    #     k = wire_radius/dist
    #     z_w = z*k
    #     r_w = (r - a)*k + a
    #     Hr, _, Hz = _magneticFdOfRing(a, J, r_w, z_w)
    #     Hr /= k
    #     Hz /= k
    #     return [Hr, 0.0, Hz]
    

    if aminus < wr2: 

        dist = aminus**0.5 # from the center of wire 
        if dist < wire_radius*0.01: # !!! recursion
              Hr, _, _  = _magneticFdOfThickRing(a, J, r + wire_radius*0.03, z, wire_radius)
              _,  _, Hz = _magneticFdOfThickRing(a, J, r, z - wire_radius*0.03, wire_radius)
              return Hr, 0.0, Hz 

        k = wire_radius/dist
        z_ = -z*k
        r_ = (r - a)*k + a
        #*r_ = (a - r)*k + a

        _z = - z_
        _r = 2.0*a - r_
        Hr_, _, Hz_ = _magneticFdOfRing(a, J,  r_, z_)
        _Hr, _, _Hz = _magneticFdOfRing(a, J, _r, _z )
        
        t = (wire_radius - dist)/wire_radius*0.5
        
        Hr = Hr_ + (_Hr - Hr_)*t
        Hz = Hz_ + (_Hz - Hz_)*t
 
        return -Hr, 0.0, Hz # ??? 
        #*return Hr, 0.0, Hz # ??? 


    rr = r*r # r**2
    aplus = (a + r)**2 + zz
    #aminus = (a - r)**2 + zz

    rrzz = rr + zz
    sqrtaplus = aplus**0.5
    #sqrtaminus = aminus**0.5  # !!! distance from wire 
    arg = 4.0*a*r/aplus
    K = sp.ellipk(arg)
    E = sp.ellipe(arg)
    E_aminus = E/aminus
    J1 = J*0.2
    Hr = J1*z*((aa + rrzz)*E_aminus-K)/(r*sqrtaplus)      
    Hz = J1*  ((aa - rrzz)*E_aminus+K)/sqrtaplus

    return Hr, 0.0, Hz   


def SI_magneticFdOfRing3D(pt, J, center, radius, normal): # !!! SI
    mx = stdRotateMx(normal)
    i_mx = invMx(mx)
    pt0 = pt - center
    pt0 = mx.dot(pt0)  

    r = vNorm(pt0[0:2])  #r = (pt0[0]**2 + pt0[1]**2)**0.5  # math.hypot(X, Y) 
    z = pt0[2]
    
    phi = atan2(pt0[1], pt0[0])   #phi = math.atan2(Y, X) 
    Hr, _, Hz = _magneticFdOfRing(radius, J, r, z)

    result = vec3D(Hr*np.cos(phi), Hr*np.sin(phi), Hz)
    result = i_mx.dot(result) # g3.transformPt(result, mx.I)
    result += center          # g3.translatePt(result, center)
    
    return 1e-6*result  # Gaussian -> SI : *1e-6 (cm, Gs) 

def SI_magneticFdOfRing3D_opt(pt, J, center, radius, mx, i_mx): # !!! SI
    pt0 = pt - center
    pt0 = mx.dot(pt0)  

    r = vNorm(pt0[0:2])  #r = (pt0[0]**2 + pt0[1]**2)**0.5  # math.hypot(X, Y) 
    z = pt0[2]
    
    phi = atan2(pt0[1], pt0[0])   #phi = math.atan2(Y, X) 
    Hr, _, Hz = _magneticFdOfRing(radius, J, r, z)

    result = vec3D(Hr*np.cos(phi), Hr*np.sin(phi), Hz)
    result = i_mx.dot(result) # g3.transformPt(result, mx.I)
    
    return 1e-6*result  # Gaussian -> SI : *1e-6 (cm, Gs) 

def SI_magneticFdOfThickRing3D(pt, J, center, radius, mx, mx_inv, wire_radius): # SI
    #mx = g3.stdRotateMx(normal)
    pt0 = pt - center       #pt0 = g3.translatePt(pt, -center)
    pt0 = mx.dot(pt0)       #pt0 = g3.transformPt(pt0, mx)
    
    #r = vNorm(pt0[0:2])     #r = (pt0[0]**2 + pt0[1]**2)**0.5  # math.hypot(pt[0], pt[1])
    r = math.hypot(pt0[0], pt0[1]) 
    z = pt0[2]
    
    phi = atan2(pt0[1], pt0[0]) # phi = math.atan2(Y, X) 
    Hr, _, Hz = _magneticFdOfThickRing(radius, J, r, z, wire_radius)

    result = np.array([  Hr*cos(phi), Hr*sin(phi), Hz  ])
    result = mx_inv.dot(result) # result = g3.transformPt(result, mx.I)
    
    return 1e-6*result  # Gaussian -> SI : *1e-6 (cm, Gs) 

def SI_magneticFdOnRingCenter(J, r): # for test
    return J/2.0/r * SI_mu0


class Ring:   
    def __init__(self, center, radius, normal, I, wire_radius): 
        self.I = I
        self.center = center
        self.radius = radius
        self.normal = normalized_vector(normal) #normal
        self.wire_radius = wire_radius
        
        self.std_mx = stdRotateMx(self.normal)
        self.std_mx_inv = invMx(self.std_mx)
        
        self.std_mx = np.array(self.std_mx)
        self.std_mx_inv = np.array(self.std_mx_inv)

    def calcB(self, r): 
        return SI_magneticFdOfThickRing3D(r, self.I, self.center, self.radius, 
            self.std_mx, self.std_mx_inv, self.wire_radius)  # 
    
    def distance(self, r): 
        if all(np.isclose(self.center, r, 1e-6)): # ???
            return self.radius
        
        norm = normalized_vector(self.normal)   
        r_c = r - self.center       
        auxilliary_plane_n = normalized_vector( np.cross(r_c, norm) ) 
        dir_on_r_on_ring_plane = np.cross(norm, auxilliary_plane_n)
        h = r_c.dot(norm) 
        y = r_c.dot(dir_on_r_on_ring_plane) - self.radius
        d = (h*h + y*y)**0.5
        return d
    
    def polygon(self, npoints, closed=True):
        pts = _regularPolygon3D(npoints, self.center, self.radius, -self.normal)
        if closed: 
            pts.append(pts[0])
        return pts
        
    def plot(self, *args, **kwargs): 
        pgn = self.polygon(100, True)
        plot_polygon(pgn, *args, **kwargs)
    
    def translate(self, vec): 
        self.center += vec
        #self.std_mx_inv = np.array(self.std_mx_inv)

    def transform(self, mx): 
        coeff = mxScaleCoeff(mx)

        self.center = mx.dot(self.center)
        self.radius = coeff*self.radius
        self.normal = normalized_vector(  mx.dot(self.normal)  ) # ??? normal : doesn't work for skew
        self.wire_radius = coeff*self.wire_radius
        self.std_mx = mx.dot(self.std_mx)
        self.std_mx_inv = invMx(self.std_mx)
        self.std_mx_inv = np.array(self.std_mx_inv)


class XZRing:  
    def __init__(self, y0, radius, I, wire_radius):
        self.y0 = y0
        self.center = np.array([0.0, y0, 0.0])
        self.normal = np.array([0.0, 1.0, 0.0])
        self.radius = radius
        self.I = I
        self.wire_radius = wire_radius
    
    def calcB(self, r): 
        pt0 = r - self.center
        Hr, _, Hz = _magneticFdOfThickRing(self.radius, self.I, math.hypot(pt0[0], pt0[2]), pt0[1], self.wire_radius)
        phi = atan2(pt0[2], pt0[0]) # phi = math.atan2(Y, X) 
        return 1e-6*np.array([Hr*cos(phi), Hz, Hr*sin(phi)])

    def polygon(self, npoints, closed=True):
        pts = _regularPolygon3D(npoints, self.center, self.radius, self.normal)
        if closed: 
            pts.append(pts[0])
        return pts    
        
    def plot(self, *args, **kwargs): 
        pgn = self.polygon(100, True)
        plot_polygon(pgn, *args, **kwargs)
    
class ThinXZRing(XZRing):
    def __init__(self, y0, radius, I):
        self.y0 = y0
        self.center = np.array([0.0, y0, 0.0])
        self.normal = np.array([0.0, 1.0, 0.0])
        self.radius = radius
        self.I = I

    def calcB(self, r): 
        pt0 = r - self.center
        Hr, _, Hz = _magneticFdOfRing(self.radius, self.I, math.hypot(pt0[0], pt0[2]), pt0[1])
        phi = atan2(pt0[2], pt0[0]) # phi = math.atan2(Y, X) 
        return 1e-6*np.array([Hr*cos(phi), Hz, Hr*sin(phi)])
    
    def calcPsi(self, r):
        pt0 = r - self.center
        psi = _magneticPsiOfRing(self.radius, self.I, math.hypot(pt0[0], pt0[2]), pt0[1])
        return 1e-6*psi
    
#if __name__ == "__main__": 
#    probing_line = line_array(pt3D(-2.0, 0.0, 0.0), pt3D(2.0, 0.0, 0.0), 1000)
#    ring = Ring(center=pt3D(0, 0, 0), radius=1.0, normal=vec3D(0, 0, 1), I=1.0, wire_radius=0.05)
#    bb = np.array( [ ring.calcB(r) for r in probing_line ] )
#    plt.plot( probing_line[:, 0], bb[:, 2] )





