# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:59:48 2024

@author: Krohalev_OD
"""

#%%
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np


import matplotlib.pyplot as plt

from .trajectory import pass_fan, Trajectory
from .stopper import CollisionStopper, CountStopper
from .aim import AbstractAim, calc_angles
from ..geom.geom import intersect_line_segment_2D, plot_point
from ..geom.group import Group3D
from ..phys.constants import SI_1keV

try:
    import multiprocessing as mp
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except:
    JOBLIB_AVAILABLE = False

#%%
class AbstractOptimizer(ABC):
    @abstractmethod
    def __call__(self, trajectory):
        """

        Parameters
        ----------
        trajectory : hibpy.beam.trajectory Trajectory

        Returns
        -------
        success : bool, True if optimization successfull
        trajectory : None if failed to optimize, else hibpy.beam.trajectory Trajectory

        """
        
#%%
def stopped_at_aim(tr):
    return isinstance(tr.obstacle, AbstractAim)

def intersect_horizontal_line(pts, line_lvl = 0.):
    for i, (pt0, pt1) in enumerate(zip(pts[:-1], pts[1:])):
        # print(i, pt0, pt1)
        intersect_pt, t = intersect_line_segment_2D((np.array([0., line_lvl]), np.array([1., 0.])), (pt0, pt1))
        if t is None:
            pass
        elif 0. <= t <= 1.:
            # print(pt0, pt1, intersect_pt, t)
            return i, intersect_pt
        
    return None, None

def intersect_vertical_line(pts, line_lvl=0.):
    for i, (pt0, pt1) in enumerate(zip(pts[:-1], pts[1:])):
        intersect_pt, t = intersect_line_segment_2D((np.array([line_lvl, 0.]), np.array([0., 1.])), (pt0, pt1))
        if t is None:
            pass
        elif 0. <= t <= 1.:
            # print(pt0, pt1, intersect_pt, t)
            return i, intersect_pt
        
    return None, None

def filtred_trace(trajectory, aim, filter_condition):
    '''
    trajectory must have fan
    '''
    # trace = [aim.mx.dot(tr.rrvv[-1, :3] - aim.center) if stopped_at_aim(tr)
    #          else None
    #          for tr in trajectory.fan]
    trace = [aim.mx.dot(tr.obstacle.intersect_with_segment(tr.rrvv[-2, :3], tr.rrvv[-1, :3]) - aim.center) if stopped_at_aim(tr)
             else None
             for tr in trajectory.fan]
    prim_indexes = [tr.prim_index if stopped_at_aim(tr)
                    else None
                    for tr in trajectory.fan]
    angles = [calc_angles(tr.rrvv[-1, 3:], aim.basis) for tr in trajectory.fan]
    
    filtred_trace  = []
    filtred_indxs  = []
    filtred_angles = []
    trace.append(None) # to catch last point of trace without separate if 
    for j, (i, pt0, pt1) in enumerate(zip(prim_indexes, trace[:-1], trace[1:])):
        if (pt0 is not None):
            if (pt1 is not None):
                if filter_condition(pt0, pt1):
                    filtred_trace.append(pt0[:2])
                    filtred_indxs.append(i)
                    filtred_angles.append(angles[j])
            elif (j > 0):
                if (trace[j-1] is not None):
                    if filter_condition(trace[j-1], pt0):
                        filtred_trace.append(pt0[:2])
                        filtred_indxs.append(i)
                        filtred_angles.append(angles[j])
            
    return filtred_indxs, filtred_trace, filtred_angles

def EmptyCondition(pt0, pt1):
    return True

def find_secondary_index(i, intersect_pt, trace, indxs, dens):
    pt0, pt1 = trace[i], trace[i+1]
    dist0 = np.linalg.norm(pt0 - intersect_pt)
    dist_full = np.linalg.norm(pt0 - pt1)
    sec_ind = indxs[i] + int(round((dist0/dist_full)*dens, 0))
    return sec_ind

def find_secondary_traj(prim_traj, E, B, aim, prim_dt, sec_dt, 
                        secondary_stopper, point_is_in_plasma, fan_density, 
                        timestep_divider=10, trace_filter_condition=EmptyCondition,
                        main_aim_axis=0):
    '''
    trajectory must have fan
    '''
    intersect_line = [intersect_horizontal_line, intersect_vertical_line][main_aim_axis]
    
    # find approximate position of secondary
    indxs_rough, trace_rough, angles = filtred_trace(prim_traj, aim, trace_filter_condition)
    if len(trace_rough) == 0:
        return None
    
    i, intersect_pt = intersect_line(trace_rough)
    if (i is None) or (intersect_pt is None):
        return None
    
    # pass tmp traj with presize step
    q, m, Ebeam, rv_prim = prim_traj.initial_data
    rv0 = prim_traj.rrvv[indxs_rough[i]]
    tmp_traj = Trajectory(q, m, Ebeam, rv0)
    stopper = CountStopper(timestep_divider*2*fan_density)
    tmp_traj.run(E, B, stopper, prim_dt/timestep_divider)
    
    # pass presize fan, find position of secondary
    pass_fan(tmp_traj, E, B, point_is_in_plasma, secondary_stopper, sec_dt, fan_density=1)
    
    indxs_presize, trace_presize, angles = filtred_trace(tmp_traj, aim, trace_filter_condition)
    if len(trace_presize) == 0:
        return None

    i, intersect_pt = intersect_line(trace_presize)
    if (i is None) or (intersect_pt is None):
        return None
    
    sec_index_presize = find_secondary_index(i, intersect_pt, trace_presize, indxs_presize, 1)
    rv0_sec = tmp_traj.rrvv[sec_index_presize]
    
    # pass secondary
    secondary = Trajectory(2*q, m, Ebeam, rv0_sec)
    secondary.run(E, B, secondary_stopper, sec_dt)
    
    # correct vertical position of secondary if possible #!!! shitty solution
    stop_point = secondary.obstacle.intersect_with_segment(secondary.rrvv[-2, :3], secondary.rrvv[-1, :3])
    line_lvl=-aim.deviation(stop_point, vec=secondary.rrvv[-1, 3:])[1-main_aim_axis]
    
    i, intersect_pt = intersect_line(trace_rough, line_lvl=line_lvl)
    if (i is not None) or (intersect_pt is not None):
        rv0 = prim_traj.rrvv[indxs_rough[i]]
        tmp_traj = Trajectory(q, m, Ebeam, rv0)
        stopper = CountStopper(timestep_divider*2*fan_density)
        tmp_traj.run(E, B, stopper, prim_dt/timestep_divider)
        
        # pass presize fan, find position of secondary
        pass_fan(tmp_traj, E, B, point_is_in_plasma, secondary_stopper, sec_dt, fan_density=1)
        
        indxs_presize, trace_presize, angles = filtred_trace(tmp_traj, aim, trace_filter_condition)
        if len(trace_presize) != 0:
            i, intersect_pt = intersect_line(trace_presize, line_lvl=line_lvl)
            if (i is not None) or (intersect_pt is not None):
                sec_index_presize = find_secondary_index(i, intersect_pt, trace_presize, indxs_presize, 1)
                rv0_sec = tmp_traj.rrvv[sec_index_presize]
                secondary = Trajectory(2*q, m, Ebeam, rv0_sec)
                secondary.run(E, B, secondary_stopper, sec_dt)
    
    # insert begining of secondary in primary
    closest_index_in_full_traj = np.nanargmin(np.linalg.norm(prim_traj.rrvv[:, :3] -
                                                             rv0_sec[:3], axis=1))

    previous_pt = prim_traj.rrvv[closest_index_in_full_traj - 1, :3]
    closest_pt  = prim_traj.rrvv[closest_index_in_full_traj,     :3]
    
    if np.linalg.norm(previous_pt - rv0_sec[:3]) <= np.linalg.norm(previous_pt - closest_pt):
        index_to_insert = closest_index_in_full_traj
    else:
        index_to_insert = closest_index_in_full_traj + 1
        
    prim_traj.rrvv = np.insert(prim_traj.rrvv, index_to_insert, rv0_sec, axis=0)
    prim_traj.secondary_index = index_to_insert
    
    return secondary

#%% 
class B2ZoneAOptimizer(AbstractOptimizer):
    def __init__(self, B2_plates, aim, primary_beamline, prim_stopper, sec_stopper, 
                 prim_dt, sec_dt, B, point_in_plasma_condition, max_steps=10, fan_density=400):
        self.plates = B2_plates
        self.aim = aim
        self.plane = aim.plane
        self.beamline = primary_beamline
        self.prim_stopper = prim_stopper
        self.sec_stopper = sec_stopper
        self.prim_dt = prim_dt
        self.sec_dt = sec_dt
        self.B = B
        self.point_in_plasma_condition = point_in_plasma_condition
        self.max_steps = max_steps
        self.fan_density = fan_density 
        
    def __call__(self, trajectory):
        U0 = self.plates.base_U
        U1 = self.plates.base_U + 1.
        
        # print('first shot')
        trajectory_0, dev_0 = self.single_shot(trajectory, U0)
        
        if dev_0 is None:
            print('Optimization failed: ', self.exception)
            return False, trajectory_0
        
        # print('second shot')
        trajectory_1, dev_1 = self.single_shot(trajectory, U1)
        
        if dev_1 is None:
            print('Optimization failed: ', self.exception)
            return False, trajectory_1
        
        step_count = 0
        
        #!!! 
        UU = [U0, U1]
        dd = [dev_0, dev_1]
        
        while True:
            step_count += 1
            print('\rOptimization steps: ' + str(step_count) + '   ', end='', flush=True)
            if step_count > self.max_steps:
                self.exception = 'max amount of steps exceeded'
                print('\nOptimization failed: ', self.exception)
                return False, None
            
            tr_res, U_res, dev_res = self.make_U_step(trajectory, U0, dev_0, U1, dev_1)
            
            #!!!
            UU.append(U_res)
            dd.append(dev_res)
            
            if dev_res is None:
                print('\nOptimization failed: ', self.exception)
                return False, tr_res
            
            # if hit target
            if abs(dev_res) < self.aim.eps[0]:
                print('\nOptimization successfull.', end='', flush=True)
                tr_res.U[self.plates.name] = U_res
                return True, tr_res
            
            U0, dev_0, U1, dev_1 = U1, dev_1, U_res, dev_res
        
    def make_U_step(self, trajectory, U1, dev_1, U2, dev_2):

        dev_to_U_coeff = (U2 - U1)/(dev_2 - dev_1)
        U_res = U2 - dev_to_U_coeff*dev_2
        trajectory_res, dev_res = self.single_shot(trajectory, U_res) 
        return trajectory_res, U_res, dev_res
        
    def single_shot(self, trajectory, U):
        # calculate fan
        self.plates.set_U(U)
        trajectory_1 = deepcopy(trajectory)
        trajectory_1.run(self.beamline.E, self.B, self.prim_stopper, self.prim_dt)
        pass_fan(trajectory_1, self.beamline.E, self.B, self.point_in_plasma_condition, 
                 self.sec_stopper, self.sec_dt, fan_density=self.fan_density)
        
        if len(trajectory_1.fan) == 0:
            self.exception = "primary didn't enter plasma"
            return trajectory_1, None
        
        secondary = find_secondary_traj(trajectory_1, self.beamline.E, self.B, self.aim, 
                                        self.prim_dt, self.sec_dt, self.sec_stopper, 
                                        self.point_in_plasma_condition, self.fan_density,
                                        trace_filter_condition=self.filter_condition,
                                        main_aim_axis=0)
        if secondary is None:
            self.exception = 'Failed to find secondary. All secondaries higher or lower than aim or trace is empty.'
            return trajectory_1, None
        
        trajectory_1.secondary = secondary
        dev = self.aim.deviation(secondary.obstacle.intersect_with_segment(secondary.rrvv[-2, :3], secondary.rrvv[-1, :3]), 
                                 secondary.rrvv[-1, 3:])[0]
        return trajectory_1, dev
    
    def filter_condition(self, pt0, pt1):
        return pt1[1] - pt0[1] <= 0
    
class B2ZoneCOptimizer(B2ZoneAOptimizer):
    def filter_condition(self, pt0, pt1):
        return pt1[1] - pt0[1] >  0
    
#%% not actually optimizers, for debagging
class TestB2ZoneAOptimizer(B2ZoneAOptimizer):
    def __call__(self, trajectory):
        U0 = self.plates.base_U
        trajectory_0, dev_0 = self.single_shot(trajectory, U0)
        return trajectory_0, dev_0
    
class TestB2ZoneCOptimizer(B2ZoneCOptimizer):
    def __call__(self, trajectory):
        U0 = self.plates.base_U
        trajectory_0, dev_0 = self.single_shot(trajectory, U0)
        return trajectory_0, dev_0
    
#%% sec bl optimizers
def set_U_and_run(traj, plates, *args, plot=False):
    plates.set_U(traj.U[plates.name])
    traj.run(*args)
    if plot:
        traj.plot(axes_code='XZ')
        plt.pause(0.01)
    return traj
    
class SecondaryBeamlineOnePlatesOptimizer(AbstractOptimizer):
    def __init__(self, A3_plates, U_limits, aim, secondary_beamline, sec_dt, B, 
                 max_steps=10, max_gradent_steps=50, U_steps=10, main_aim_axis=0, 
                 parallel=True, silent=True):
        self.main_aim_axis = main_aim_axis
        self.intersect_line = [intersect_vertical_line, intersect_horizontal_line][main_aim_axis]
        self.plates = A3_plates
        self.aim = aim
        self.beamline = secondary_beamline
        self.dt = sec_dt
        self.B = B
        self.U_limits = U_limits
        self.U_steps = U_steps
        self.max_steps = max_steps
        self.max_gradent_steps = max_gradent_steps
        self.beamline_collider = Group3D([plates.collider for plates in self.beamline])
        self.stopper = CollisionStopper(Group3D([self.aim, self.beamline_collider]))
        self.parallel = parallel
        self.silent = silent
        self.exception = None
    
    def __call__(self, trajectory):
        print(f'{self.plates.name} optimization', end='', flush=True)
        step_count = 0
        U_limits, U_steps = self.U_limits, self.U_steps
        self.exception = None
        self.log_trace = []
        
        if self.plates.name not in trajectory.U.keys():
            trajectory.U[self.plates.name] = self.plates.base_U
        
        success, traj = self.try_sec(trajectory, trajectory.U[self.plates.name])
        if success:
            trajectory.secondary = traj
            print(f'\r{self.plates.name} optimization successful')
            return True, trajectory
        
        if not any([self.plates._domain_box.contains_pt(r) for r in traj.rrvv[:, :3]]):
            self.exception = f"secondary didn't enter {self.plates.name}"
            print(f'\r{self.plates.name} optimization failed: ', self.exception)
            return False, trajectory
        
        while True:
            step_count += 1
            if step_count > self.max_steps:
                self.exception = 'too many steps'
                # print(f'\r{self.plates.name} optimization failed: ', self.exception)
                break
                # return False, trajectory
            
            sec_list, U_array = self.shot(trajectory, U_limits, U_steps)
                
            trace = [self.aim.mx.dot(tr.obstacle.intersect_with_segment(tr.rrvv[-2, :3], tr.rrvv[-1, :3]) - self.aim.center)[:2] for tr in sec_list]
            
            self.log_trace.append(trace)
            
            i, intersect_pt = self.intersect_line(trace)
            
            if (i is None) or (intersect_pt is None):
                self.exception = 'U out of limits'
                print(f'\r{self.plates.name} optimization failed: ', self.exception)
                trajectory.U[self.plates.name] = self.plates.U
                return False, trajectory
            
            U = self.find_U(trace, U_array, i, intersect_pt)
            success, traj = self.try_sec(trajectory, U)
            if success:
                trajectory.U[self.plates.name] = U
                trajectory.secondary = traj
                print(f'\r{self.plates.name} optimization successful')
                return True, trajectory
            else:
                U_limits, U_steps = [U_array[i], U_array[i+1]], self.U_steps
                
        if self.exception == 'too many steps':
            success, traj, U = self.dumb_down(trajectory, (self.U_limits[1] - self.U_limits[0])/U_steps, U)
            trajectory.U[self.plates.name] = U
            trajectory.secondary = traj
            if success:
                print(f'\r{self.plates.name} optimization successful')
                return True, trajectory
            else:
                print(f'\r{self.plates.name} optimization failed: ', self.exception)
                return False, trajectory
            
    def dumb_down(self, trajectory, step_size, start_U):
        U = start_U
        step_count = 0
        deviation = None
        
        while True:
            success, traj = self.try_sec(trajectory, U)
            if success:
                return success, traj, U
            
            step_count += 1
            if step_count > self.max_gradent_steps:
                self.exception = 'too many steps in dumb down'
                return False, traj, U
            
            intersect_pt = traj.obstacle.intersect_with_segment(traj.rrvv[-2, :3], traj.rrvv[-1, :3])
            prev_deviation = deviation if deviation is not None else self.aim.deviation(intersect_pt, traj.rrvv[-1, 3:])[self.main_aim_axis]
            deviation = self.aim.deviation(intersect_pt, traj.rrvv[-1, 3:])[self.main_aim_axis]
            if deviation*prev_deviation < 0:
                step_size *= 0.2

            U += deviation*step_size
            if (U > self.U_limits[1]) or (U < self.U_limits[0]):
                self.exception = 'U out of limits in dumb down'
                return False, traj, U
        
    def shot(self, trajectory, U_limits, U_steps):
        U_array = np.linspace(U_limits[0], U_limits[1], U_steps)
        q, m, Ebeam, prim_rv0 = trajectory.initial_data
        rv = trajectory.rrvv[trajectory.secondary_index]
        traj_list = []
        for U in U_array:
            traj = Trajectory(q*2, m, Ebeam, rv)
            traj.prim_index = trajectory.secondary_index
            traj.U[self.plates.name] = U
            traj_list.append(traj)
            
        if self.parallel and JOBLIB_AVAILABLE:
            n_workers = mp.cpu_count() - 1 
            traj_list = Parallel (n_jobs=n_workers) (delayed(set_U_and_run)(tr, self.plates, self.beamline.E, self.B, self.stopper, self.dt) for tr in traj_list) 
            
        else:
            for tr in traj_list:
                set_U_and_run(tr, self.plates, self.beamline.E, self.B, self.stopper, self.dt, plot=not self.silent)
   
        return traj_list, U_array
    
    def find_U(self, trace, U_array, i, intersect_pt):
        pt0, pt1 = trace[i], trace[i+1]
        dist0 = np.linalg.norm(pt0 - intersect_pt)
        dist_full = np.linalg.norm(pt0 - pt1)
        U = U_array[i] + (dist0/dist_full)*(U_array[i+1] - U_array[i])
        return U
    
    def try_sec(self, trajectory, U):
        q, m, Ebeam, prim_rv0 = trajectory.initial_data
        rv = trajectory.rrvv[trajectory.secondary_index]
        traj = Trajectory(q*2, m, Ebeam, rv)
        self.plates.set_U(U)
        traj.run(self.beamline.E, self.B, self.stopper, self.dt)
        
        deviation = self.aim.deviation(traj.obstacle.intersect_with_segment(traj.rrvv[-2, :3], traj.rrvv[-1, :3]), traj.rrvv[-1, 3:])
        if abs(deviation[self.main_aim_axis]) < self.aim.eps[self.main_aim_axis]:
            return True, traj
        else:
            return False, traj
        
class EmptyOptimizer(AbstractOptimizer):
    def __call__(self, trajectory):
        return False, trajectory
    
    @property
    def exception(self):
        return(None)
    
#%%
def optimize_secondary_beamline(first_wave_optimizers, second_wave_optimizers, 
                                reserve_optimizers, main_aim, trajectory_list, 
                                max_optimization_steps=3):
    while len(reserve_optimizers) < len(second_wave_optimizers):
        reserve_optimizers.append(EmptyOptimizer())
        
    presize_optimizers = deepcopy(second_wave_optimizers)
    presize_optimizers.extend(reserve_optimizers)
    
    plates_list = [opt.plates for opt in second_wave_optimizers]
    optimized_list = []
    for trajectory in trajectory_list:
        
        exception = None
        traj = trajectory
        
        print(f'Optimizing secondary trajectory, E = {round(traj.Ebeam/SI_1keV, 0)} keV, U_A2 = {traj.U["A2"]} kV')
        traj_success = False
        
        for plates in plates_list:
            plates.U = plates.base_U
            
        print('Rough optimization')
        for rough_optimizer in first_wave_optimizers:
            success, traj = rough_optimizer(traj)
            # traj.secondary.plot()
            # plt.pause(0.01)
            
        print('Presize optimization')
        for i in range(max_optimization_steps):
            if traj.secondary is not None:
                if main_aim.hit(traj.secondary.rrvv[-1, :3], traj.secondary.rrvv[-2, :3]):
                    print('trajectory optimized and saved\n')
                    traj_success = True
                    optimized_list.append(traj)
                    break
            
            print(f'optimization attempt {i+1}')
            success_fine = []
            for optimizer, reserve in zip(second_wave_optimizers, reserve_optimizers):
                success, traj = optimizer(traj)
                if not success:
                    if not isinstance(reserve, EmptyOptimizer):
                        print('reserve ', end='', flush=True)
                    success, traj = reserve(traj)
                success_fine.append(success)
                
            if all(success_fine):
                if main_aim.hit(traj.secondary.rrvv[-1, :3], traj.secondary.rrvv[-2, :3]):
                    optimized_list.append(traj)
                    print('trajectory optimized and saved\n')
                    traj_success = True
                    break
                
            if all([opt.exception == 'U out of limits' for opt in presize_optimizers]):
                exception = 'U out of limits'
                break
            
        if not traj_success:
            if exception is None:
                exception = 'too many steps'
            print('optimization failed: ', exception, '\n')
    
    return optimized_list