# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:11:19 2024

@author: Ammosov

traj_list tools for friendly operation with traj_list.

"""
import numpy as np
from itertools import product
from abc import ABC, abstractmethod

from ..geom.group import Group3D
from ..phys.constants import SI_1keV
from ..beam.trajectory import Trajectory
from ..geom.groupstrat import ListStrategy
#%% functions
def print_traj(tr):
    """
    Prints Ebeam and UA2 values of trajectory to console.

    Parameters
    ----------
    tr : Trajectory
        Object of Trajectory class.

    Returns
    -------
    None.
    """
    if (tr is not None) and isinstance(tr, Trajectory):
        print(f"E={int(tr.Ebeam/SI_1keV)} UA2={int(tr.U['A2'])}")

def print_traj_list(traj_list):
    """
    Prints index, Ebeam and UA2 values for each trajectory of traj_list to console.

    Parameters
    ----------
    traj_list : list[Trajectory]
        List of Trajectory class objects.

    Returns
    -------
    None.
    """
    if len(traj_list) > 0:
        i = 0
        for tr in traj_list:
            print(f"[{i}] E={round(tr.Ebeam/SI_1keV)} UA2={round(tr.U['A2'])}")
            i += 1
            
def get_traj(traj_list, Ebeam, UA2):
    """
    Returns trajectory OR None if there is no such trajectory.

    Parameters
    ----------
    traj_list : list[Trajectory]
        List of Trajectory class objects.
    Ebeam : float
        Beam Energy value in keV.
    UA2 : float
        UA2 Voltage in kV.

    Returns
    -------
    tr : Trajectory
        Object of Trajectory class.

    """
    if len(traj_list) > 0:
        i = 0
        for tr in traj_list:
            if np.isclose(tr.Ebeam/SI_1keV, Ebeam) and tr.U['A2'] == UA2:
                return tr
            i += 1
        return None
#%% masks
class AbstractMask(ABC):
    @abstractmethod
    def apply(self, traj_list):
        """
        Applies mask to list of Trajectories.

        Parameters
        ----------
        traj_list : list[Trajectory]
            List of Trajectory class objects. Each trajectory mast have Ebeam and U['A2'].

        Returns
        -------
        traj_list : list[Trajectory]
            Masked list of Trajectory class objects.
        """
    
    def print(self, traj_list):
        print_traj_list(self.apply(traj_list))
    
class ConditionMask(AbstractMask):
    def __init__(self, condition):
        self.condition = condition
    
    def apply(self, traj_list):
        if len(traj_list) > 0:
            traj_list_slice = []
            for tr in traj_list:
                if self.condition(tr):
                    traj_list_slice.append(tr)
            return traj_list_slice
        
    def check_nomask(self):
        if len(self.values) == 0:
            self.condition = lambda tr: True
        
class EScanMask(ConditionMask):
    """
    for example:
        mask = EScanMask(180)
        mask = EScanMask(180, 220, 230, 270)
        mask = EScanMask() # no mask
        ---------------
        traj_list_slice = mask.apply(traj_list)
    """
    def __init__(self, *args):
        self.values = [*args]
        self.check_nomask()

    def condition(self, traj):
        return any([np.isclose(traj.Ebeam/SI_1keV, value) for value in self.values])
        
class UScanMask(EScanMask):
    """
    for example:
        mask = UScanMask(-22)
        traj_list_slice = mask.apply(traj_list)
    """
    def __init__(self, *args, scan_plates='A2'):
        super().__init__(*args)
        self.scan_plates = scan_plates
        
    def condition(self, traj):
        return any([traj.U[self.scan_plates] == value for value in self.values])

class ListMask(ConditionMask):
    """
    for example:
        mask = ListMask((180, -22), (220, -15))
        traj_list_slice = mask.apply(traj_list)
    """
    def __init__(self, *args, scan_plates='A2'):
        self.values = [*args]
        self.check_nomask()
        self.scan_plates = scan_plates
        
    def condition(self, traj):
        return any([(np.isclose(traj.Ebeam/SI_1keV, value[0]) and traj.U[self.scan_plates] == value[1]) for value in self.values])
        
class ExcludeMask(ListMask):
    """
    for example:
        mask = ExcludeMask((180, -22), (220, -15))
        traj_list_slice = mask.apply(traj_list)
    """
    
    def condition(self, traj):
        return all([not (np.isclose(traj.Ebeam/SI_1keV, value[0]) and traj.U[self.scan_plates] == value[1]) for value in self.values])

class IncludeRangeMask(ListMask):
    """
    for example:
        mask = IncludeRangeMask(
                                (range(200, 280+1, 20), range(-20, -5)),
                                (range(200, 280+1, 20), range(-20, -5)),
                                (500, -9), 
                                (500, -9), 
                                (500, -8)
                                )
        traj_list_slice = mask.apply(traj_list)
    """
    
    def __init__(self, *args, scan_plates='A2'):
        self.values = []
        self.scan_plates = scan_plates
        
        if len([*args]) == 0:
            self.condition = lambda tr: True
        else:
            for value in [*args]:
                
                if isinstance(value, int) or isinstance(value, float):
                    self.values += [value]
                    continue
                
                ebeams = value[0]
                ua2s = value[1]
                
                if not (isinstance(value[0], range) or isinstance(value[0], list)):
                    ebeams = [value[0]]
                
                if not (isinstance(value[1], range) or isinstance(value[1], list)):
                    ua2s = [value[1]]
                
                self.values += product(ebeams, ua2s)
                
class ExcludeRangeMask(ExcludeMask):
    """
    for example:
        mask = ExcludeRangeMask(
                                (range(200, 280+1, 20), range(-20, -5)),
                                (range(200, 280+1, 20), range(-20, -5)),
                                (500, -9), 
                                (500, -9), 
                                (500, -8)
                                )
        traj_list_slice = mask.apply(traj_list)
    """
    def __init__(self, *args, scan_plates='A2'):
        self.values = []
        self.scan_plates = scan_plates
        
        if len([*args]) == 0:
            self.condition = lambda tr: True
        else:
            for value in [*args]:
                
                if isinstance(value, int) or isinstance(value, float):
                    self.values += [value]
                    continue
                
                ebeams = value[0]
                ua2s = value[1]
                
                if not (isinstance(value[0], range) or isinstance(value[0], list)):
                    ebeams = [value[0]]
                
                if not (isinstance(value[1], range) or isinstance(value[1], list)):
                    ua2s = [value[1]]
                
                self.values += product(ebeams, ua2s)

Group3D.call_strategies['apply'] = ListStrategy() # allow grouping masks
#%% testing block
if __name__ == '__main__':
    
    Btor = 1.5
    Jpl  = 1.5
    zone = 'A'
    line_name = 'A'
    folder = r'detector_grids\\for_presentation\\'
    
    traj_list = np.load(folder+f"sec B {Btor}T, I {Jpl}MA, line {line_name}, zone {zone} adaptive_aim True.npy", allow_pickle=True)
    
    mask2 = ListMask((280, -15), (300, -22))
    mask22 = ListMask()
    
    mask3 = EScanMask(400, 420)
    mask33 = EScanMask()
    
    mask4 = UScanMask(-12)
    
    mask5 = ExcludeMask((280, -15), (300, -22))
    mask55 = ExcludeMask()
    
    mask6 = IncludeRangeMask(
        (range(200, 280+1, 20), range(-20, -5)),
        (range(200, 280+1, 20), range(-20, -5)),
        (500, -9), 
        (500, -9), 
        (500, -8)
        )
    
    mask7 = ExcludeRangeMask(
        (range(200, 280+1, 20), range(-20, -5)),
        (range(200, 280+1, 20), range(-20, -5)),
        (500, -9), 
        (500, -9), 
        (500, -8)
        )
    
    mask = mask7
    traj_list_slice = mask.apply(traj_list)
    
    # test group
    # new_mask = Group3D([mask2, mask2])
    # new_traj_list = new_mask.apply(traj_list)
    
    # prints
    print_traj_list(traj_list)
    print('-----------------------')
    print_traj_list(traj_list_slice)
    # print_traj_list(new_traj_list)