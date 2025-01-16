# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:36:14 2019

"100 Pesetas"
Miscellaneous utils

@author: reonid

Exports:
    update_default_kwargs
    func_from_file
    iter_ranges
    find_ranges
    expand_mask
    narrow_mask
    ispair
    x_op_y
    func_x_y

"""
#%%
import time
from copy import deepcopy
from inspect import getfullargspec
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

#%%
class StopWatch:
    def __init__(self, title=''):
        self.title = title
        self.t0 = None
        self.t1 = None

    def time_passed(self):
        if self.t1 is None:
            return time.time() - self.t0 # inside
        else:
            return self.t1 - self.t0 # outside

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exp_type, exp_value, traceback):
        self.t1 = time.time()
        if exp_type is None:
            print(self.title + ': dt = %.2f' % (self.t1 - self.t0))
        else:
            print(self.title + ': dt = %.2f  ERROR' % (self.t1 - self.t0))

        # return True
        return False # !!! don't suppress exception

#%%
class Indexing:
    def __getitem__(self, arg):
        return arg

indexing = Indexing()

#%%
def get_args_kwargs(*args, **kwargs):
    return args, kwargs

#%%
def acc_vstack(acc, arr):
    if acc is None:
        return arr
    else:
        return np.vstack((acc, arr))

def acc_hstack(acc, arr):
    if acc is None:
        return arr
    else:
        return np.hstack((acc, arr))

#%%
def update_default_kwargs(func_or_argspec, kwargs_dict, new_defaults):
    '''
    Update default values for arguments in kwargs
    if they are not given explicitly and if they are relevant for the function
    Also remove non-relevant arguments from kwargs
    Returns new dictionary

    Example:
    kwargs = update_default_kwargs(mlab.psd, kwargs, {'detrend': 'linear'})
    mlab.psd(sig, **kwargs)

    It can be used just for select relevalt arguments:
    kw = update_default_kwargs(np.genfromtxt, kwargs, {})
    data = np.genfromtxt(filename, **kw)
    '''
    argspec = getfullargspec(func_or_argspec) if callable(func_or_argspec) else func_or_argspec
    result = {argname:value for argname, value in kwargs_dict.items() if argname in argspec.args}

    for argname in new_defaults:
        new_def_val = new_defaults[argname]
        if argname in argspec.args:
            if not argname in result:
                result[argname] = new_def_val

    return result


def func_from_xy(xarr, yarr, **kwargs):
    '''
    Just wrapper on interpolate.interp1d
    '''

    kw = update_default_kwargs(interpolate.interp1d, kwargs,
        {'bounds_error':False, 'fill_value':'extrapolate'})

    if kw['fill_value'] == 'const':
        kw['fill_value'] = (yarr[0], yarr[-1])

    interp = interpolate.interp1d(xarr, yarr, **kw)
    return interp


def func_from_file(filename, xcol=0, ycol=1, **kwargs):
    '''
    Load functional dependency y(x) from file
    with help of scipy.interpolate.interp1d

    fill_value == 'const' : constant extrapolation with edge values
    other kwargs : as in numpy.genfromtxt + scipy.interpolate.interp1d

    Example:
    f = func_from_file('func.txt', 0, 1, fill_value='extrapolate', skip_header=2)
    '''

    kw = update_default_kwargs(np.genfromtxt, kwargs, {})

    data = np.genfromtxt(filename, **kw)
    xx = data[:, xcol]
    yy = data[:, ycol]

    return func_from_xy(xx, yy, **kw)

def iter_ranges(seq, condition):
    '''
    Divides sequence on ranges with the same value of condition
    |True|False|True|False|...
    or
    |False|True|False|True|...

    for start, fin, cond_value, isfirst, islast in iter_ranges(sec, condition):
        ...

    '''
    inside = False
    fin = 0
    start, last = None, None
    for i, value in enumerate(seq):
        last = i
        ok = condition(value)
        if (not inside)and(ok):
            inside = True
            start = i
            if start > fin:
                isfirst, islast = (fin == 0), False
                yield fin, i, False, isfirst, islast
            fin = None

        if (inside)and(not ok):
            inside = False
            fin = i
            isfirst, islast = (start == 0), False
            yield start, fin, True, isfirst, islast
            start = None

    # Last range: condition is the same till the end

    if last is None:
        pass # empty seq
    elif start is not None:
        isfirst, islast = (start==0), True
        yield start, last+1, True, isfirst, islast
    elif fin is not None:
        isfirst, islast = (fin==0), True
        yield fin, last+1, False, isfirst, islast


def find_ranges(seq, condition):
    return [(i0, i1) for i0, i1, ok, *_ in iter_ranges(seq, condition) if ok]


def expand_mask(mask, left, right):
    '''
    Expand mask as
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
                     |
                     V
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    '''
    L = len(mask)
    result = deepcopy(mask)
    for i0, i1, ok, isfirst, islast in iter_ranges(mask, lambda x: x == 1):
        if not ok:
            if not isfirst:
                result[i0 : min(L, i0 + right)] = True
            if not islast:
                result[max(0, i1 - left) : i1] = True

    return result

def narrow_mask(mask, left, right):
    '''
    Narrow mask as
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
                     |
                     V
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    '''
    L = len(mask)
    result = deepcopy(mask)
    for i0, i1, ok, isfirst, islast in iter_ranges(mask, lambda x: x):
        if ok:
            if not isfirst:
                result[i0 : min(L, i0 + left)] = False
            if not islast:
                result[max(0, i1 - right) : i1] = False

    return result

def ispair(arg, dtype=None):
    '''
    Check if arg is tupple of two elements of required type (if specified)

    Exapmpes:
    ispair((1, 2))
    ispair((1, 2), dtype=int)
    ispair((1, 2), dtype=numbers.Number)
    ispair((1, 2.0), dtype=(float, int))
    ispair((None, 2), dtype=(int, type(None)))

    '''
    if isinstance(arg, tuple):
        if len(arg) == 2:
            if (dtype is None): # type ignored
                return True
            else:
                a, b = arg
                return isinstance(a, dtype)and isinstance(b, dtype)
    return False


def x_op_y(x, op, y):
    if y is None:
        x, y = y, x

    if x is None:
        if   op == '':    return y
        elif op == '+':   return +y
        elif op == '-':   return -y
        elif op == '~':   return ~y
        elif op == 'not': return not y
        elif op == 'T':   return y.T
        elif op == 'I':   return y.I
        #elif op == 'str': return str(y)
        else: return op(y)

    if ispair(y):  # if ispair(x): ??? not supported
        x0, x1 = y
        if   op == 'inside':  return (x >= x0)&(x <  x1) # [..)
        elif op == '(..)':    return (x >  x0)&(x <  x1)
        elif op == '[..]':    return (x >= x0)&(x <= x1)
        elif op == '[..)':    return (x >= x0)&(x <  x1)
        elif op == '(..]':    return (x >  x0)&(x <= x1)

        elif op == 'outside': return (x >  x0)|(x <= x1) # .)[.
        elif op == '.)(.':    return (x <  x0)|(x >  x1)
        elif op == '.][.':    return (x <= x0)|(x >= x1)
        elif op == '.)[.':    return (x <  x0)|(x >= x1)
        elif op == '.](.':    return (x <= x0)|(x >  x1)
        else: pass

    if   op == '<':  return x <  y
    elif op == '<=': return x <= y
    elif op == '==': return x == y
    elif op == '!=': return x != y
    elif op == '>':  return x >  y
    elif op == '>=': return x >= y

    if   op == '+':  return x +  y
    elif op == '-':  return x -  y
    elif op == '*':  return x *  y
    elif op == '/':  return x /  y
    elif op == '//': return x // y
    elif op == '%':  return x %  y
    #elif op == 'divmod': return divmod(x, y)

    elif op == '**': return x ** y
    elif op == '<<': return x << y
    elif op == '>>': return x >> y
    elif op == '&':  return x &  y
    elif op == '^':  return x ^  y
    elif op == '|':  return x |  y
    elif op == '@':  return x.dot(y) # x @ y  - old versions don't support @ operator
    elif op == 'dot':   return x.dot(y)
    elif op == 'cross': return x.cross(y) # >< x X #

    elif op == '[]':  return x[y]
    elif op == '()':  return x(y)

    elif op == ',':   return (x,y)
    elif op == '(,)': return (x,y)
    elif op == '[,]': return [x,y]

    elif op == 'in':     return x in y
    elif op == 'not in': return (x not in y)
    elif op == 'is':     return x is y
    elif op == 'is not': return (x is not y)

    elif op == 'and': return x and y
    elif op == 'or':  return x or y
    elif op == 'xor': return x ^ y

    elif op == 'x':   return x
    elif op == 'y':   return y

    else: return op(x, y) #raise Exception('x_op_y: invalid operation symbol %s' % op)

def func_x_y(func, x, y=None):
    return x_op_y(x, func, y)

def savelisttotext(lst, filename):
    with open(filename, "wt") as f:
        for s in lst:
            f.write(str(s) + '\n')

def loadlistfromtext(filename):
    with open(filename, "rt") as f:
        #return f.readlines()
        return [s.strip() for s in f.readlines()]

def stop_loop_by_any_key():
    def handle_kbd(event):
        if event.name == 'key_press_event':
            plt.gcf()._stop_ = True

    fig = plt.gcf()
    if hasattr(fig, '_stop_'):
        return fig._stop_
    else:
        fig._stop_ = False
        fig.canvas.mpl_connect('key_press_event', handle_kbd)

def save_array(filename, array):
    with open(filename, 'wb') as file:
        np.save(file, array)

def save_arrays(filename, **kwargs):
    with open(filename, 'wb') as file:
        np.savez(file, **kwargs)

def cyclic_zip(a, closed=True):
    for a0, a1 in zip(a[:-1], a[1:]):
        yield a0, a1
    if closed:
        yield a[-1], a[0]

#%%
if __name__ == '__main__':
    pass
