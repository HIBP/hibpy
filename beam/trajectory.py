# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:24:55 2023

@authors: Eliseev_LG, Krohalev_OD
"""

#%% imports
import math
import numpy as np
import matplotlib.pyplot as plt

from ..beam.slits import SlitPolygon
from ..phys.constants import SI_1keV
from ..phys.runge_kutta import runge_kutta4
from ..geom.geom import (vNorm, get_coord_indexes, vec3D, rotation_mx_by_angles)

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception as e:
    JOBLIB_AVAILABLE = False
    print('WARNING: joblib import failed', e)

#%%
def calc_vector(length, alpha, beta):
    '''
    calculate vector based on its length and angles
    alpha is the angle with XZ plane
    beta is the angle of rotation around Y axis
    '''
    vec = vec3D(length, 0, 0)
    mx = rotation_mx_by_angles(alpha, beta, 0.0)
    vec = mx.dot(vec)
    return vec

#%%
TRAJ_BLOCK_LEN = 10000

class Trajectory():
    def __init__(self, q, m, Ebeam, rv0, U=None):
        '''
        Parameters
        ----------
        q : float
            particle charge [Co]
        m : float
            particle mass [kg]
        Ebeam : float
            beam energy [keV]
        rv0 : np.array, shape==(6,)
            initial point and velocity of the trajectory [m]
        U : dict
            dict of voltages in [kV] keys=[A1 B1 A2 B2 A3 B3 A4 an]
            optional
        Returns
        -------
        None.
        '''
        self.q = q
        self.m = m
        self.Ebeam = Ebeam
        self.rrvv = rv0.reshape(1, 6)
        self.__fan = []
        self.partial_dense_fan = []
        self.secondary = None
        self.obstacle = None
        self.prim_index = 0
        self.sec_index = None
        self.lambdas = {}
        self.attens = {}
        if U is None:
            U = {}
        self.U = U
        self.log = []

    @classmethod
    def from_injector(cls, q, m, Ebeam, r0, alpha, beta, U=None):
        Ebeam_J = Ebeam
        v_abs = math.sqrt(2.0 * Ebeam_J / m)
        v0 = calc_vector(-v_abs, alpha, beta)
        rv0 = np.hstack( (r0, v0) )
        return cls(q, m, Ebeam, rv0, U)

    @property
    def fan(self):
        return self.__fan

    @fan.setter
    def fan(self, fan):
        if type(fan) != list:
            raise ValueError('fan must be list of Trajectories, %s is given' % type(fan))
        if any([type(tr) != Trajectory for tr in fan]):
            raise ValueError('fan must be list of Trajectories, list of %s is given' % type(fan[0]))
        self.__fan = fan

    @property
    def points(self):
        return self.rrvv

    @property
    def initial_data(self):
        return self.q, self.m, self.Ebeam, self.rrvv[0]

    @property
    def slit_bins(self):
        """
        Sort full trajectory fan to slit bins.

        Parameters
        ----------
        traj_list : list[Trajectory]
            List of Trajectory class objects; full fan of one trajectory.

        Returns
        -------
        slit_bins : dict {int : list[Trajectory, ...]}
            Dictionary: keys - slit numbers : int; values - fans to slits : list[Trajectory, ...].

        """
        traj_list = self.partial_dense_fan
        if len(traj_list) > 0:
            slit_bins = {}
            tr_to_slits = [tr for tr in traj_list if isinstance(tr.obstacle, SlitPolygon)]
            for tr in tr_to_slits:
                if tr.obstacle.number in slit_bins.keys():
                    slit_bins[tr.obstacle.number].append(tr)
                else:
                    slit_bins[tr.obstacle.number] = [tr]
            return slit_bins

    def print_log(self, s):
        self.log.append(s)
        print(s)

    def find(self, r):
        dd = self.rrvv[:, 0:3] - r
        idx = np.argmin(vNorm(dd, axis=1))
        if all(  np.isclose(self.rrvv[idx, 0:3], r, 1e-8)  ):
            return idx
        return None

    @property
    def segments(self):
        yield from zip(self.rrvv[0:-1, 0:3], self.rrvv[1:, 0:3])

    def intersect_with_object(self, obj):
        for i, r0, r1 in enumerate(self.segments()):
            r_intersect = obj.intersect_with_segment(r0, r1)
            if r_intersect is not None:
                return i, r_intersect
        return None, None

    def cut_with_plane(self, plane):
        index, r = self.intersect_with_object(plane)
        if r is not None:
            self.rrvv = self.rrvv[:index+2]
            self.rrvv[index+1, 0:3] = r

    def run(self, E, B, stopper, dt):
        rv_old = self.rrvv[-1]  # initial position
        q_m = self.q / self.m

        _curr = self.rrvv.shape[0]
        _rrvv = np.vstack((  self.rrvv, np.full( (TRAJ_BLOCK_LEN, 6), np.nan)  ))

        while True:
            rv_new = runge_kutta4(rv_old, q_m, dt, E, B)
            if np.isnan(rv_new).any():
                self.obstacle = 'B_is_None'
                break

            _rrvv[_curr] = rv_new; _curr += 1
            if _curr >= _rrvv.shape[0]:
                 _rrvv = np.vstack((   _rrvv, np.full( (TRAJ_BLOCK_LEN, 6), np.nan)  ))

            self.rrvv = _rrvv[0:_curr]
            if stopper(self):
                # self.obstacle = stopper.obstacle
                break

            rv_old = rv_new

        self.rrvv = self.rrvv.copy()

    def plot(self, axes_code='XY', *args, **kwargs):
        i1, i2 = get_coord_indexes(axes_code)
        xx = self.rrvv[:, i1]
        yy = self.rrvv[:, i2]
        plt.plot(xx, yy, *args, **kwargs)

    def transform(self, mx):
        for tr in self.fan:
            tr.transform(mx)
        if self.secondary is not None:
            self.secondary.transform(mx)
        for i in range(self.rrvv.shape[0]):
            rv = self.rrvv[i]
            r = rv[:3]
            v = rv[3:]
            self.rrvv[i, :3] = mx.dot(r)
            self.rrvv[i, 3:] = mx.dot(v)

    def translate(self, vec):
        for tr in self.fan:
            tr.translate(vec)
        if self.secondary is not None:
            self.secondary.translate(vec)
        for i in range(self.rrvv.shape[0]):
            self.rrvv[i, :3] += vec

    def __repr__(self):
        Ebeam = f"Ebeam:{round(self.Ebeam/SI_1keV, 1)}, "
        U = ", ".join([f"{key}:{round(self.U[key], 2)}" for key in self.U.keys()])
        return Ebeam + U
#%%
def run_with_return(tr, *args):
    tr.run(*args)
    return tr

def pass_fan(trajectory, E, B, point_is_in_plasma, fan_stopper, dt_sec,
             fan_density=100, parallel=True, verbose=0):
    prim_rrvv = trajectory.points
    q, m, Ebeam, prim_rv0 = trajectory.initial_data
    fan = []
    for i, rv in enumerate(prim_rrvv):
        if point_is_in_plasma(rv[0:3]) and (i%fan_density == 0):
            fan.append(Trajectory(q*2, m, Ebeam, rv))
            fan[-1].prim_index = i
    if parallel and JOBLIB_AVAILABLE:
        fan = Parallel(n_jobs=-2, verbose=verbose) (delayed(run_with_return)(tr, E, B, fan_stopper, dt_sec) for tr in fan)
    else:
        for tr in fan:
            tr.run(E, B, fan_stopper, dt_sec)
    trajectory.fan = fan
