# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:11:59 2024

@author: Sarancha_GA
"""

# Imports
import sys
import numpy as np
hibpy_path = 'D:\\py\\hibp-packages\\'
if hibpy_path not in sys.path: sys.path.append(hibpy_path)
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d as interp1d
from scipy.signal import argrelextrema
from hibpy.phys.constants import SI_eps0





class Neutralizer:
    """Neutralizer is an object for beam current analytical calculation.
    In general, solves a system of ordinary differential equations based on 
    kinetics of charge exchange-recombination reactions:
    
    .. math::

            \\frac{dI_{α}}{dz} = \sum_j n_{j}I_{α}σ_{jα}
    
    BUT CURRENTLY EVALUATES ONLY beam current for exponential solution
    
    .. math::

            I = I_{0}\\exp[-nσ(z-z_{0})]
    
    Coordinate system: cylindrical (z, r, θ). No dependence on r, θ."""
    
    
    def __init__(self, isNeutralizerTrue = False, z_Neutralizer_origin = np.nan, 
                                                  z_Neutralizer_ending = np.nan, 
                                                      ne = 2e19, sigma = 6e-19):
        '''
        
        Parameters
        ----------
        isNeutralizerTrue : bool, optional
            the flag if calculate space-dependent current. The default is False.
        z_Neutralizer_origin : float, optional
            the neutralizer origin z-coordinate [m]. The default is np.nan.
        z_Neutralizer_ending : float, optional
            the neutralizer ending z-coordinate [m]. The default is np.nan.
        ne : float, optional
            electron density in neutralizer. The default is 2e19 [m^-3].
        sigma : float, optional
            reaction cross-section. The default is 6e-19 [m^-2].

        '''
                
        self.flag     = isNeutralizerTrue
        self.z_origin = z_Neutralizer_origin
        self.z_ending = z_Neutralizer_ending
        self.ne       = ne
        self.sigma    = sigma
        
    def I_beam(self, I_0):
        '''

        Parameters
        ----------
        I_0 : float
            Initial beam current [A].

        Returns
        -------
        I_beam : Interp1d
            The beam current interpolator along the z-axis.

        '''
        
        z = np.linspace(0.5*self.z_origin, 1.5*self.z_ending, 1001)
        I = np.zeros_like(z)
        
        if self.flag:
          for i in range(len(I)):  
            if z[i] < self.z_origin:
                I[i] = I_0
            elif z[i] >= self.z_origin and z[i] < self.z_ending:
                I[i] = I_0*np.exp(-self.ne*self.sigma*(z[i]-self.z_origin))
            else:
                I[i] = I_0*np.exp(-self.ne*self.sigma*(self.z_ending-self.z_origin))
            return interp1d(z, I, bounds_error = False, fill_value = (I_0, I_0*np.exp(-self.n_e*self.sigma*(self.z_ending-self.z_origin))))
        else:
            I.fill(I_0)
            return interp1d(z, I, bounds_error = False, fill_value = I_0)
        
    
#%% Charged particle beam

class Envelope:
    """Particle beam envelope is a trajectory of such a probing charged particle,
    that no beam's particle exist outside this trajectory.
    
    Coordinate system: cylindrical (z, r, θ). No dependence on θ.
    
    
    According to Gauss` law force on such a particle is
    
    .. math::

        m\ddot{r} = qE_{r} + \\frac{qI}{2\pi\\varepsilon_{0}r\dot{z}}
    
    .. math::

        m\ddot{z} = qE_{z}

    
    """

    
    def __init__(self, q, m, I_0, z_0, r_0, dr_0, K_0, neutralizer = Neutralizer()):
        '''
        Parameters
        ----------
        q : int
            particle charge [Co].
        m : int
            particle mass [kg].
        I_0 : float
            current at the inlet [A].
        z_0 : float
            z-coordinate of the inlet [m].
        r_0 : float
            r-coordinate of the inlet [m].
        dr_0 : float
            divergence at the inlet [rad]
            divergence =def= (radial velocity)/(axial velocity).
        K_0 : float
            kinetic energy at the inlet [J].
        neutralizer : envelope.Neutralizer object, optional
            neutralizer, which goes (if goes) after injector. The default is Neutralizer().

        '''

        self.q   = q
        self.m   = m
        self.k   = self.q/self.m
        self.mu  = 0.5/np.pi/SI_eps0 
        self.I   = neutralizer.I_beam(I_0)
        
        self.t   = [0.0]
        self.z   = [z_0]
        self.r   = [r_0]
        self.vz  = [np.sqrt(2*K_0/self.m)*np.cos(dr_0)]
        self.vr  = [np.sqrt(2*K_0/self.m)*np.sin(dr_0)]
        self.F_z = None
        self.F_r = None
           
    @property
    def focus(self):
        '''
        Focus is a position on the trajectory where the local minimum of r(z) is reached.

        Returns
        -------
        None.

        '''
               
        idx_min  = argrelextrema(self.r, np.less)
        self.F_z = np.array(self.z[idx_min])
        self.F_r = np.array(self.r[idx_min])
        
    
    def size_on_coordinates(self, wires):
        '''
        Beam size on list of z-coordinates. Especially used for wire sensors.

        Parameters
        ----------
        wires : np.array, shape==(1,)
            z-positions of wires [m].

        Returns
        -------
        r_on_wires : np.array, shape==(1,)
            r-positions of beam [m].

        '''
        
        r = interp1d(self.z, self.r, bounds_error = True)
        r_wire = []
        for z_wire in wires:
            r_wire.extend([r(z_wire)])
        return np.array(r_wire)
    
            
    def __rarefy(self, N = 100):
        '''
        Rarefy beam properties after calculation for further saving

        Parameters
        ----------
        N : int, optional
            Save every Nth point. The default is 100.

        Returns
        -------
        None.

        '''
        
        self.t      = np.array(self.t [::N])
        self.z      = np.array(self.z [::N])
        self.r      = np.array(self.r [::N])
        self.vz     = np.array(self.vz[::N])
        self.vr     = np.array(self.vr[::N])
        self.I      = np.array(self.I(self.z))
    
        
    @property
    def k_energy(self):
        '''
        Calculates beam kinetic energy

        Returns
        -------
        k_energy : np.array, shape==(1, len(Envelope.t))
            Array of beam energies [scale] at each stored time step.

        '''
       
        return np.array(self.m*(self.vr**2 + self.vz**2)/2.)
    
    @property
    def divergence(self):
        '''
        Calculates beam divergence

        Returns
        -------
        divergence : np.array, shape==(1, len(Envelope.t))
            array of beam divergences at each stored time step [rad].

        '''
        
        return np.arctan(self.vr/self.vz)
    
    
    def acceleration(self, E_field):
        '''
        Calculates beam acceleration by components

        Parameters
        ----------
        E_field : RegularGridInterpolator2D
            electric field interpolator (Ez: [V/m], Er: [V/m]).

        Returns
        -------
        a_Ez : np.array, shape==(1, len(Envelope.t))
            array of electric field z-acceleration at each stored time step [m/s^2].
        a_Er : np.array, shape==(1, len(Envelope.t))
            array of electric field r-acceleration at each stored time step [m/s^2].
        a_beam : np.array, shape==(1, len(Envelope.t))
            array of beam self-field r-acceleration at each stored time step [m/s^2].

        '''
       
        a_Ez   = np.array(self.k*E_field([self.z, self.r])[1])
        a_Er   = np.array(self.k*E_field([self.z, self.r])[0])
        a_beam = np.array(self.k*self.mu*self.I(self.z)/self.r/self.vz)
        
        return a_Ez, a_Er, a_beam
    
    
    def __f(self, rv, E_field):
        '''
        f for Runge-Kutta solver
        
        Parameters
        ----------
        rv : np.array, shape==(1, 4)
            array of coordinates and velocities z [m], r [m], vz [m/s], vr [m/s].
        Returns
        -------
        result : np.array, shape==(1, 4)
            array of corrected coordinates and velocities z [m], r [m], vz [m/s], vr [m/s].
        
        '''
        z, r, vz, vr = rv
        result    = np.empty(4)
        result[0] = vz
        result[1] = vr
        result[2] = self.k*E_field([z, r])[0]
        result[3] = self.k*E_field([z, r])[1] + self.k*self.mu*self.I(z)/(r*vz)
        
        return result
            
    
    def runge_kutta4(self, E_field, h):
        '''
        4th order Runge-Kurtta iterational solver

        Parameters
        ----------
        E_field : RegularGridInterpolator2D
            electric field interpolator (Ez: [V/m], Er: [V/m]).
        h : float
            time step for solver, default 0.1 [ns].

        Returns
        -------
        None. 
            Rewrites attributes of object.

        '''
                        
        rv = [self.z[-1], self.r[-1], self.vz[-1], self.vr[-1]]
        k1 = h*self.__f(rv         , E_field)
        k2 = h*self.__f(rv + 0.5*k1, E_field)
        k3 = h*self.__f(rv + 0.5*k2, E_field)
        k4 = h*self.__f(rv +     k3, E_field)
        rv_new = rv + (k1 + 2.0*(k2 + k3) + k4)/6.0
        
        self.z.extend([rv_new[0]])
        self.r.extend([rv_new[1]])
        self.vz.extend([rv_new[2]])
        self.vr.extend([rv_new[3]])
        self.t.extend([self.t[-1]+h])
        
    
    def euler(self, E_field, h):
        '''
        1st order Euler iterational solver

        Parameters
        ----------
        E_field : RegularGridInterpolator2D
            electric field interpolator (Ez: [V/m], Er: [V/m]).
        h : float
            time step for solver, default 0.1 [ns].

        Returns
        -------
        None.
            Rewrites attributes of object.
            
        '''
                
        a_Ez   = self.k*E_field([self.z[-1], self.r[-1]])[0]
        a_Er   = self.k*E_field([self.z[-1], self.r[-1]])[1]
        a_beam = self.k*self.mu*self.I(self.z[-1])/self.r[-1]/self.vz[-1]
        
        self.vr.extend([self.vr[-1] + (a_Er + a_beam)*h])
        self.vz.extend([self.vz[-1] + (a_Ez         )*h])
        self.r.extend( [self.r[-1]  + self.vr[-1]*h])
        self.z.extend( [self.z[-1]  + self.vz[-1]*h])
        self.t.extend( [self.t[-1]  + h])
    
        
    def beam_tracing(self, method, E_field, h = 1e-10, z_max = None, r_max = None):
        '''
        Beam envelope tracing iterator

        Parameters
        ----------
        method : callable
            iterator function: Euler or Runge-Kutta.
        E_field : RegularGridInterpolator2D
            electric field interpolator (Ez: [V/m], Er: [V/m]).
        h : float, optional
            time step for solver. The default is 1e-10.
        z_max : float, optional
            maximum beam z-coordinate limitation [m]. The default is max of E_field grid.
        r_max : float, optional
            maximum beam z-coordinate limitation [m]. The default is max of E_field grid.

        Returns
        -------
        None.

        '''
        
        if r_max is None:
            r_max = max(E_field.yy)
        if z_max is None:    
            z_max = max(E_field.xx)
            
        while self.t[-1] < 1e-3:
            if (np.abs(self.r[-1]) > r_max and self.z[-1] < 1.):
                print('Beam has flown out through the side of the injector')
                break
            elif (self.z[-1] > z_max):
                print('Beam has flown far away along the injector axis')
                break
            elif (self.vz[-1] < 0.):
                print('Beam is reflected')
                break
            else:
                if np.abs(self.r[-1]) < 0.5e-4:
                    """finer time step if extremely near z-axis"""
                    dt = h/100.
                elif np.abs(self.r[-1]) < 2e-4:
                    """fine time step if near z-axis"""
                    dt = h/10.
                else: 
                    """normal time step"""
                    dt = h
                
                method(E_field, dt)
                
        self.__rarefy()
        
        
    def radius_plot(self, ax = None, rscale = 1e3, focus = False):
        '''
        Plotter of the r(z) function. Typically used as a stand-alone chart

        Parameters
        ----------
        ax : matplotlib.axes, optional
            subplot of figure for plotting. The default is None.
        rscale : float, optional
            units of radius scale. The default is 1e3 (m --> mm).
        focus : bool, optional
            if plot focus point on scatter plot. The default is False.

        Returns
        -------
        None.

        '''
        
        if ax is None:
            plt.figure()
            ax = plt.gca()
            
        ax.plot(self.z, rscale*self.r)
        if focus:
            focus()
            for i in range(len(self.F_z)):
                ax.scatter(self.F_z[i], rscale*self.F_r[i])
        
    
    def envelope_plot(self, ax = None):
        '''
        Plotter of the envelope shape. Usually used in the same figure with the injector

        Parameters
        ----------
        ax : matplotlib.axes, optional
            subplot of figure for plotting. The default is None.

        Returns
        -------
        None.

        '''
        
        if ax is None:
            plt.figure()
            ax = plt.gca()
            
        ax.fill_between(self.z, self.r, -self.r, color = 'red', alpha = 0.45)
        ax.plot(self.z, self.r, color = 'red', linewidth = 1)
        ax.plot(self.z, -self.r, color = 'red', linewidth = 1)
   


#%% Gaussian packet profile            
        
class Gaussian:
    '''Gaussian profile of particle beam packet (assume that the entire profile lies in 3σ):
           
    .. math::

        \Gamma = \Gamma_{0}\\frac{3\\exp{[-\\frac{9r^{2}}{2(r_{0}+r'_{0}\\cdot(z-z_{0}))^{2})}]}}{\\sqrt{2\pi}(r_{0}+r'_{0}\\cdot(z-z_{0}))}
    
    Coordinate system: cylindrical (z, r, θ). No dependence on θ.
    '''
    
    
    def __init__(self, z_0, r_0, dr_0, dI_0, q):
        '''
        
        Parameters
        ----------
        z_0 : float
            z-coorfinate of packet origin [m].
        r_0 : float
            half width of packet in r-direction in the origin [m].
        dr_0 : float
            divergence of packet in r-direction in the origin [rad]
            divergence =def= (radial velocity)/(axial velocity).
        dI_0 : float
            current of particle beam packet [A].
        q : float
            particle charge [Co].

        '''
        
        self.z_0     = z_0
        self.r_0     = r_0
        self.dr_0    = dr_0
        self.dFlux_0 = dI_0/q

        
    def profile(self, z, r_minmax, N = 1001):
        '''
        Calculation of the evolving profile packet

        Parameters
        ----------
        z : float
            coordinate on the z-axis, where the profile is calculated [m].
        r_minmax : float
            half-width boundary within which the profile is calculated [m].
        N : int, optional
            points on profile. The default is 1001.

        Returns
        -------
        j : np.array, shape==(1,)
            intensity distribution of particle beam at z-coordinate [pcs].

        '''
        
        r = np.linspace(-r_minmax, r_minmax, N)
        if z < self.z_0:
            """Profile has not been born yet"""
            return np.zeros_like(r)
        
        else:
            j = (self.dFlux_0*
                 (3./np.sqrt(2.*np.pi*(self.r_0 + self.dr_0*(z - self.z_0))**2)*
                  np.exp(-9.*r**2/2./(self.r_0 + self.dr_0*(z - self.z_0))**2))/
                  np.sum(3./np.sqrt(2.*np.pi*(self.r_0 + self.dr_0*(z - self.z_0))**2)*
                  np.exp(-9.*r**2/2./(self.r_0 + self.dr_0*(z - self.z_0))**2)))
            return j