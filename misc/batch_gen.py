# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:35:49 2024

@author: reonid
"""

import numpy as np


class BatchGen: 
    def __init__(self, gen, n, batchgengen): 
        self.gen = gen
        self.n = n
        self.i = 0
        self.batchgengen = batchgengen
    
    def __iter__(self): 
        return self
    
    def __next__(self):
        self.i += 1
        if self.i > self.n: 
            raise StopIteration  # source generator can continue
        
        try: 
            result = next(self.gen)

        except StopIteration:    # source generator is ended
            self.batchgengen.exceed = True
            raise # StopIteration

        return result
    
class BatchGenGen: 
    def __init__(self, gen, n): 
        self.gen = gen
        self.n = n  
        self.i = 0
        self.exceed = False

    def __iter__(self): 
        return self
    
    def __next__(self):
        self.i += 1
        result = BatchGen(self.gen, self.n, self)
        if self.exceed: 
            raise StopIteration

#        if self.i > 10000: # stop infinite loop
#            raise StopIteration
        
        return result

if __name__ == '__main__':
    #a = np.linspace(0, 10, 11)
    a = np.linspace(0, 11, 12)
    #a = np.linspace(0, 9, 10)
    
    gen = (x for x in a)
    
    bb = BatchGenGen(gen, 3)
    
    for b in bb: # b can be empty generator
        print('---')
        for x in b: 
            print(x)

