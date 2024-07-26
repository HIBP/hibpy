# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:03:52 2024

@author: Krohalev_OD
"""
#%%
import numpy as np
from itertools import islice
from joblib import Parallel

from .batch_gen import BatchGenGen

#%%
# def batched(a_list, n):
#     L = len(a_list)
#     i = 0
#     j = i + n
#     while j < L+n: 
#         yield a_list[i:min(j, L)]
#         i += n
#         j += n
    
def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def batched_gen(gen, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1: n = 1
    tmp_arr = list(gen)
    L = len(tmp_arr)
    i = 0
    while i < L:     
        yield tmp_arr[i:min(i+n, L)]
        i += n

        
#%%
# fan = Parallel (n_jobs=n_workers) (delayed(run_with_copy)(tr, E, B, fan_stopper, dt_sec) for tr in fan)
class _Parallel:
    def __init__(self, package_size, *args, **kwargs):
        self.package_size = package_size
        self._parallel = Parallel(*args, **kwargs)
        
    def __call__(self, iterable):
        result = []        
        if self.package_size is None:
            return self._parallel(iterable)
            
        for batch in BatchGenGen(iterable, n=self.package_size):
            result.extend(self._parallel(batch))
            
        return result
    
    def __truediv__(self, arg): 
        self.package_size = arg
        return self

#%%
if __name__ == '__main__':
    a = np.linspace(0, 10, 11)
    #b = batched(a, n=3)
    
    gen = (x for x in a)
    # for i, x in enumerate(gen):
    #     print(x)
    #     if i > 2:
    #         gen.close()
    
    # print(gen)
    # for i, x in enumerate(gen):
    #     print(x)
    
    for g in batched_gen(gen, 3):
        print('__')
        for x in g: 
            print(x)