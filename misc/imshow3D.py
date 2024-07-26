# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:36:34 2024

@author: Krohalev_OD
"""
#%%
import sys 
hibpy_path = 'D:\\py\\hibp-packages\\'
if hibpy_path not in sys.path: sys.path.append(hibpy_path) 

#%%
import matplotlib.pyplot as plt
import numpy as np

from hibpy.geom.geom import get_ortho_index
from hibpy.misc.cienptas import indexing

#%%
def onWhealEvent(event):
    if event.name == 'scroll_event': 
        ax = plt.gca()
        cmin, cmax = ax.images[0].get_clim()

        if event.button == 'down':  k = 1.1
        elif event.button == 'up':  k = 0.9

        cmax *= k
        if not abs(cmin)*10 < abs(cmax): cmin *= k
        
        ax.images[0].set_clim((cmin, cmax))        
        plt.draw()

def indexator(i, dim):
    res = [indexing[i, :, :], indexing[:, i, :], indexing[:, :, i]][dim]
    # res[dim] = i
    return res
    
class ImageKeyHandler():
    def __init__(self, image, array3D, axes_code, start_number=0):
        self.current = start_number
        self.array3D = array3D
        self.image = image
        self.axes_code = axes_code
        self.i = get_ortho_index(axes_code)
        
    def __call__(self, event):
        if event.name == 'key_press_event': 
            d = 1
            
            if 'shift' in event.key:
                d = 10
                
            if event.key in ['down', 'right', 'shift+right', 'shift+up']:
                self.current = min(self.array3D.shape[self.i] - 1, self.current + d)
                    
            if event.key in ['up', 'left', 'shift+left', 'shift+down']:
                self.current = max(0, self.current - d)
                
            self.image.set_array(self.array3D[indexator(self.current, self.i)].T)
            # title = plt.gca().get_title()
            # plt.gca().set_title(str(indexator(self.current, self.i)))
            plt.draw()
            # print(event.key)
                    
def imshow3D(data, axes_code='XZ'):
    i = get_ortho_index(axes_code)
    ind = indexator(0, i)
    
    fig = plt.figure()
    im = plt.imshow(data[ind].T)
    fig.canvas.mpl_connect('scroll_event', onWhealEvent)
    fig.canvas.mpl_connect('key_press_event', ImageKeyHandler(im, data, axes_code))
    plt.colorbar(im)
    
#%%
if __name__ == '__main__':
    path = r'D:\py\programs\SynHIBP\devices\T-15MD\electric_field\precalculated\U_A3_B_up_new.npy'
    data = np.load(path)
    i = get_ortho_index('XY')
    ind = indexator(0, i)
    
    fig = plt.figure()
    im = plt.imshow(data[ind].T)
    fig.canvas.mpl_connect('scroll_event', onWhealEvent)
    fig.canvas.mpl_connect('key_press_event', ImageKeyHandler(im, data, 'XY'))
    plt.colorbar(im)