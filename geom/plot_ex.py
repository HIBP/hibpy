# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:19:18 2024

@author: Krohalev_OD
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

from .geom import pt2D, vec2D, vec3D, plot_polygon, get_coord_indexes, line_array

#%%
def filter_points_nans(points, skip_func):
    if skip_func is None:
        return points

    def _pt(pt):
        if skip_func(pt):
            return np.full_like(pt, np.nan)
        else:
            return pt

    return [_pt(pt) for pt in points]

def filter_points_nans_interp(points, skip_func, accuracy=20):

    _points = []

    prev_pt = points[-1]
    prev_skip = skip_func(prev_pt)

    for curr_pt in points:
        curr_skip = skip_func(curr_pt)

        if not prev_skip and not curr_skip: # both points are OK
            _points.append(curr_pt)
        elif prev_skip ^ curr_skip:
            segm_pts = line_array(prev_pt, curr_pt, accuracy) # 20
            _segm_pts = filter_points_nans(segm_pts, skip_func)

            if prev_skip:
                _points.append( vec3D(np.nan, np.nan, np.nan) )

            _points.extend(_segm_pts)

            if curr_skip:
                _points.append( vec3D(np.nan, np.nan, np.nan) )
        else:
            pass # both points are out of plot

        prev_pt = curr_pt
        prev_skip = curr_skip
    return _points


def plot_polygon_ex(points, axes_code=None, dont_plot_condition=None, color=None, **kwargs):
    if dont_plot_condition is None:
        _points = points
    else:
        _points = filter_points_nans_interp(points, dont_plot_condition)

    if len(_points) > 0:
        return plot_polygon(_points, axes_code=axes_code, color=color, **kwargs)
    else:
        return plt.plot([], [], color=color, **kwargs)


#%%
def extract_polygon_2D(points, axes_code=None):
    X, Y = get_coord_indexes(axes_code)

    xx = [pt[X] for pt in points]
    xx.append(points[0][X])

    yy = [pt[Y] for pt in points]
    yy.append(points[0][Y])

    return xx, yy # np.asarray


def points2D_with_min_max_projection(polygon, center, perp):
    max_proj = -np.inf
    min_proj = np.inf
    max_pt = polygon[0]
    min_pt = polygon[0]

    xx, yy = polygon
    for x, y in zip(xx, yy):
        pt = pt2D(x, y)
        proj = perp.dot(pt - center)
        if proj > max_proj:
            max_proj = proj
            max_pt = pt

        if proj < min_proj:
            min_proj = proj
            min_pt = pt

    return min_pt, max_pt

def plot_cyllinder(pgn1, pgn2, axes_code=None, accuracy=50, **kwargs):

    base1_xxyy2d = extract_polygon_2D(pgn1, axes_code)
    base2_xxyy2d = extract_polygon_2D(pgn2, axes_code)

    base1_center2d = np.nanmean(base1_xxyy2d, axis=1)
    base2_center2d = np.nanmean(base2_xxyy2d, axis=1)

    axis2d = base2_center2d - base1_center2d
    perp2d = vec2D(axis2d[1], -axis2d[0])

    minpt1, maxpt1 = points2D_with_min_max_projection(base1_xxyy2d, base1_center2d, perp2d)
    minpt2, maxpt2 = points2D_with_min_max_projection(base2_xxyy2d, base2_center2d, perp2d)

    color = plt.plot(base1_xxyy2d[0], base1_xxyy2d[1], **kwargs)[0].get_color()
    kwargs['color'] = color
    plt.plot(base2_xxyy2d[0], base2_xxyy2d[1], **kwargs)

    plt.plot([minpt1[0], minpt2[0]], [minpt1[1], minpt2[1]], **kwargs)
    return plt.plot([maxpt1[0], maxpt2[0]], [maxpt1[1], maxpt2[1]], **kwargs)

#%%

def points3D_with_min_max_projection(polygon, center, perp):
    max_proj = -np.inf
    min_proj = np.inf
    max_pt = polygon[0]
    min_pt = polygon[0]

    for pt in polygon:
        proj = perp.dot(pt - center)
        if proj > max_proj:
            max_proj = proj
            max_pt = pt

        if proj < min_proj:
            min_proj = proj
            min_pt = pt

    return min_pt, max_pt

def calc_perp3d(axis3d, axes_code):
    X, Y = get_coord_indexes(axes_code)
    axis2d = vec2D(axis3d[X], axis3d[Y])
    perp2d = vec2D(axis2d[1], -axis2d[0])
    perp3d = vec3D(0, 0, 0)
    perp3d[X] = perp2d[0]
    perp3d[Y] = perp2d[1]
    return perp3d

def get_cyllinder_shell_polygon(pgn1, pgn2, axes_code=None, accuracy=50):
    X, Y = get_coord_indexes(axes_code)

    center1 = np.nanmean(np.asarray(pgn1), axis=0)
    center2 = np.nanmean(np.asarray(pgn2), axis=0)

    axis = center2 - center1
    perp3d = calc_perp3d(axis, axes_code)

    minpt1, maxpt1 = points3D_with_min_max_projection(pgn1, center1, perp3d)
    minpt2, maxpt2 = points3D_with_min_max_projection(pgn2, center2, perp3d)

    return [minpt1, minpt2, vec3D(np.nan, np.nan, np.nan), maxpt1, maxpt2, vec3D(np.nan, np.nan, np.nan)]

def plot_cyllinder_ex(pgn1, pgn2, axes_code=None, dont_plot_condition=None, accuracy=30, **kwargs):
    if dont_plot_condition is None:
        return plot_cyllinder(pgn1, pgn2, axes_code=axes_code, accuracy=accuracy, **kwargs)

    color = plot_polygon_ex(pgn1, axes_code=axes_code, dont_plot_condition=dont_plot_condition, **kwargs)[0].get_color()
    kwargs['color'] = color
    plot_polygon_ex(pgn2, axes_code=axes_code, dont_plot_condition=dont_plot_condition, **kwargs)

    shell_pgn = get_cyllinder_shell_polygon(pgn1, pgn2, axes_code=axes_code, accuracy=accuracy)

    return plot_polygon_ex(shell_pgn, axes_code=axes_code, dont_plot_condition=dont_plot_condition, **kwargs)
