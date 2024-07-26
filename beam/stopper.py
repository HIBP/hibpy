# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:34:59 2023

@author: Eliseev_LG, Krohalev_OD
"""
#%% imports
from abc import ABC, abstractmethod
import numpy as np

#%% 
class AbstractStopper(ABC):
    
    @abstractmethod
    def __call__(self, trajectory):
        """
        Must set trajectory.obstacle in case if where is any
        
        Parameters
        ----------
        trajectory : hibpcalc.trajectory.Trajectory

        Returns
        -------
        stop : bool
            True if trajectory should stop running 

        """

#%%
def as_obstacle(obj):
    if hasattr(obj, '_decoy'):
        return obj._decoy()
    return obj

def get_obstacle(obj):
    if hasattr(obj, "_active_elem"):
        return get_obstacle(obj._active_elem)
    return as_obstacle(obj)

class OneStepStopper(AbstractStopper):
    def __init__(self):
        pass
    
    def __call__(self, trajectory):
        return True
    
class CollisionStopper(AbstractStopper):
    def __init__(self, obj):
        self.collider = obj
        
    def __call__(self, trajectory):
        r0, r1 = trajectory.rrvv[-2, :3], trajectory.rrvv[-1, :3]
        r_collision = self.collider.intersect_with_segment(r0, r1)
        if r_collision is not None:
            trajectory.obstacle = get_obstacle(self.collider)
            return True
        return False
    
class SpeedStopper(AbstractStopper):
    def __init__(self, limit):
        self.limit = limit
        
    def __call__(self, trajectory):
        v = trajectory.rrvv[-1, 3:]
        return np.linalg.norm(v) < self.limit
            
class CountStopper(AbstractStopper):
    def __init__(self, limit):
        self.limit = limit
        
    def __call__(self, trajectory):
        return (trajectory.rrvv.shape[0] > self.limit)