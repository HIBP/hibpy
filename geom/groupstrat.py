# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:26:52 2023

@author: reonid



"""

from abc import abstractmethod, ABC

#%%
class AbstractGroupMethodStrategy(ABC): 
    def __init__(self):
        pass

    def prefix(self, group, method_name):
        pass #
        
    def postfix(self, group, method_name): 
        pass #
    
    @abstractmethod
    def join(self, result, addition, group, elem): # accumulate
        pass # return result + addition, stop

    def stop(self, result): 
        return False   # default

    def set_active_elem(self, group, elem, method_name): 
        pass # default: do nothing
    


#%%

def group_method(group, call_strategy, method_name): 
    #strategy = group.__class__.strategies[name]

    def method(*args, **kwargs): 
        result = None
        call_strategy.prefix(group, method_name)
        call_strategy.set_active_elem(group, None, method_name)
        
        for elem in group: 
            elem_result = getattr(elem, method_name)(*args, **kwargs)
            
            if result is None: 
                result = elem_result
            else:
                result = call_strategy.join(result, elem_result, group, elem)

            if call_strategy.stop(result): 
                call_strategy.set_active_elem(group, elem, method_name)
                break

        call_strategy.postfix(group, method_name)
        return result
    
    #def generator(*args, **kwargs): 
    #    pass
    
    return method

#%%
class DefaultStrategy(AbstractGroupMethodStrategy): 
    def set_active_elem(self, group, elem, method_name): 
        print("""
-------------------------------------------------------------------------------
WARNING: 
Methodics of calling arbitrary methods for Group objects if following: 

1) All the elements of the group shoup support this method

2) Group class should know the way how results should be accumulated
For this purposes a strategy should be specified for every method name 

Default strategy just raise exception and show this message

Example: 
    Group3D.call_strategies['plot'] = groupstrat.JustCallStrategy()
    Group3D.call_strategies['calcB'] = groupstrat.PlusStrategy()
-------------------------------------------------------------------------------
""")
        raise NotImplementedError("%s doesn't have a strategy for calling method '%s'" % (group.__class__.__name__, method_name))
        
    def join(self, result, addition, group, elem): 
        raise NotImplementedError("Call strategy for method ") # never called

#%%
class PlusStrategy(AbstractGroupMethodStrategy): 
    def join(self, result, addition, group, elem): 
        return result + addition

#%%
class ListStrategy(AbstractGroupMethodStrategy):
    def join(self, result, addition, group, elem): 
        return result + addition

#%%
class JustCallStrategy(AbstractGroupMethodStrategy): 
    def join(self, result, addition, group, elem): 
        return None
    
#%%
#class ChainCallStrategy(AbstractGroupMethodStrategy): 
#    def join(self, result, addition): 
#        ???return None

#%%
#class GeneratorStrategy(AbstractGroupMethodStrategy): 
#    def join(self, result, addition): 
#        return ???
        
#%%
class UntilStrategy(AbstractGroupMethodStrategy): 
    def set_active_elem(self, group, elem, method_name): 
        if hasattr(group, '_active_elem'):
            group._active_elem = elem

    def join(self, result, addition, group, elem): 
        return result or addition
    
    def stop(self, result): 
        return result

