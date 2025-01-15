# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:54:51 2023

@author: reonid
"""

#%% from reometry
import numba
import numpy as np
import math
import matplotlib.pyplot as plt

#%%
def trivial_decorator():
    def decorator(func):
        return func
    return decorator


_numba_njit = numba.njit
# _numba_njit = trivial_decorator

#%% trigonometry
def sin(x):
    if x - math.pi == 0.0: return 0.0
    if x + math.pi == 0.0: return 0.0
    return math.sin(x)

def cos(x):
    if x - math.pi*0.5 == 0.0: return 0.0
    if x + math.pi*0.5 == 0.0: return 0.0
    return math.cos(x)

#%% vector
def vec3D(x, y, z):
    return np.array([x, y, z], dtype=np.float64)

pt3D = vec3D
size3D = vec3D
vNorm = np.linalg.norm

def vec2D(x, y):
    return np.array([x, y], dtype=np.float64)

pt2D = vec2D

def normalizedVector(v):
    return v/vNorm(v)

normalized_vector = normalizedVector

def transformPt(pt, mx):   # 2D, 3D
    return mx.dot(pt)

def transformPts(pts, mx):
    return mx.dot(pts.T).T


main_basis = np.array([vec3D(1, 0, 0),
                       vec3D(0, 1, 0),
                       vec3D(0, 0, 1)])

#%% matrix
def mxScaleCoeff(mx):
    ks = [vNorm(col) for col in mx.T]
    ds = abs(ks - ks[0])
    if all(ds < 1E-6):
        return ks[0]
    else:
        #return None
        raise Exception('mxScaleCoeff: matrix cannot be decomposed on scale and rotation')

def invMx(mx):
    return np.matrix(mx).I.A

def identMx():
    return np.array([[ 1.0,  0.0,  0.0],
                     [ 0.0,  1.0,  0.0],
                     [ 0.0,  0.0,  1.0]],   dtype=np.float64)

def identMx2D():
    return np.array([[ 1.0,  0.0],
                     [ 0.0,  1.0]],   dtype=np.float64)

def mx2Dto3D(mx2D):
    mx3D = np.vstack((mx2D, [0., 0.]))
    mx3D = np.hstack((mx3D, [[0.], [0.], [1.]]))
    return mx3D

def scaleMx(k):
    return np.array([[ k,    0.0,  0.0],
                     [ 0.0,  k,    0.0],
                     [ 0.0,  0.0,  k  ]],   dtype=np.float64)

def xScaleMx(k):
    return np.array([[ k,    0.0,  0.0],
                     [ 0.0,  1.0,  0.0],
                     [ 0.0,  0.0,  1.0]],   dtype=np.float64)

def xySkewMx(ang): #  skewMx(ang, 'XY')
    tg = np.tan(ang)
    return np.array([[ 1.0,  tg,   0.0],    # [0, 1]
                     [ 0.0,  1.0,  0.0],
                     [ 0.0,  0.0,  1.0]],   dtype=np.float64)

def yxSkewMx(ang): #  skewMx(ang, 'YX')
    tg = np.tan(ang)
    return np.array([[ 1.0,  0.0,  0.0],
                     [ tg,   1.0,  0.0],   # [1, 0]
                     [ 0.0,  0.0,  1.0]],   dtype=np.float64)

def skewMx(ang, plane_code): # 'XY', 'YX', ...
    X, Y = get_coord_indexes(plane_code)
    tg = np.tan(ang)
    mx = identMx()
    mx[X, Y] = tg
    return mx

def rotateMx(axis, ang):
    res = identMx()
    s = sin(ang)
    c = cos(ang)
    x, y, z = -axis/np.linalg.norm(axis)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z

    res[0, 0] = xx + (1 - xx)*c
    res[0, 1] = xy*(1 - c) + z*s
    res[0, 2] = xz*(1 - c) - y*s

    res[1, 0] = xy*(1 - c) - z*s
    res[1, 1] = yy + (1 - yy)*c
    res[1, 2] = yz*(1 - c) + x*s

    res[2, 0] = xz*(1 - c) + y*s
    res[2, 1] = yz*(1 - c) - x*s
    res[2, 2] = zz + (1 - zz)*c

    return res

def xRotateMx(ang):
    s = sin(ang)
    c = cos(ang)
    return np.array([[ 1.0,  0.0,  0.0],
                     [ 0.0,    c,   -s],
                     [ 0.0,    s,    c]],   dtype=np.float64)

def yRotateMx(ang):
    s = sin(ang)
    c = cos(ang)
    return np.array([[ c,    0.0,    s],
                     [ 0.0,  1.0,  0.0],
                     [-s,    0.0,    c]],   dtype=np.float64)

def zRotateMx(ang):
    s = sin(ang)
    c = cos(ang)
    return np.array([[ c,   -s,    0.0],
                     [ s,    c,    0.0],
                     [ 0.0,  0.0,  1.0]],   dtype=np.float64)

def rotateMx2D(ang):
    s = sin(ang)
    c = cos(ang)
    return np.array([[ c,   -s],
                     [ s,    c]],   dtype=np.float64)

def zwardRotateMx(vec):    # vec -> (0, 0, L)
# to transform any plane to XY plane
    v = vec
    a1 = np.arctan2(v[0], v[2])
    mx1 = yRotateMx(-a1)

    v = transformPt(v, mx1)
    a2 = np.arctan2(v[1], v[2])
    mx2 = xRotateMx(a2)

    return mx2.dot(mx1)

stdRotateMx = zwardRotateMx

def xwardRotateMx(vec):    # vec -> (L, 0, 0)
# to transform any plane to XY plane
    v = vec
    a1 = np.arctan2(v[1], v[0])
    mx1 = zRotateMx(-a1)

    v = transformPt(v, mx1)
    a2 = np.arctan2(v[2], v[0])
    mx2 = yRotateMx(a2)

    return mx2.dot(mx1)

def transformMx(basis_src, basis_dest):
# to transform obj from one coord system to another
    global_to_src  = np.asarray(basis_src)
    global_to_dest = np.asarray(basis_dest)
    src_to_global = invMx(global_to_src)
    full = global_to_dest.dot(src_to_global)
    return full

def rotation_mx_by_angles(alpha, beta, gamma):
    basis = np.array([vec3D(1, 0, 0),
                      vec3D(0, 1, 0),
                      vec3D(0, 0, 1)])
    mx = identMx()

    alpha_mx = rotateMx(basis[2], alpha)
    basis = np.asarray([alpha_mx.dot(vec) for vec in basis])
    mx = alpha_mx.dot(mx)

    beta_mx  = rotateMx(basis[1], beta)
    basis = np.asarray([beta_mx.dot(vec) for vec in basis])
    mx = beta_mx.dot(mx)

    gamma_mx = rotateMx(basis[0], gamma)
    mx = gamma_mx.dot(mx)

    return mx

def proj_coord_mx(rot_mx: np.ndarray, trans_vec: tuple[float, float, float]) -> np.ndarray:
    """
    Create a 4x4 transformation matrix from a 3x3 rotation matrix and a 3D translation vector.
    It is not in cartesian coordinates. It is in homogeneous coordinates.

    Args:
    - rot_mx (numpy.ndarray): 3x3 rotation matrix.
    - trans_vec (tuple): A tuple of three values (x, y, z) representing translation.

    Returns:
    - numpy.ndarray: The 4x4 transformation matrix.
    """
    # Ensure the rotation matrix is 3x3 and translation vector has 3 elements
    if rot_mx.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    if len(trans_vec) != 3:
        raise ValueError("Translation vector must have 3 elements.")

    # Create a 4x4 identity matrix
    transformation_matrix = np.eye(4)

    # Fill the rotation part (top-left 3x3) and translation part (top-right 3x1)
    transformation_matrix[:3, :3] = rot_mx
    transformation_matrix[:3, 3] = trans_vec

    return transformation_matrix

#%% line
def line_array(pt0, pt1, n):
    ndim = len(pt0)
    result = np.zeros( (n, ndim) )
    #line = g3.Line3D(pt0, pt1 - pt0)
    for i in range(n):
        result[i, :] = pt0 + (pt1 - pt0)*i/(n-1)
    return result

def intersect_line_segment_2D(line, segm, eps=1e-9):
    line_pt, line_vec = line
    line_norm = np.asarray([-line_vec[1], line_vec[0]])
    return _intersect_plane_segm_((line_pt, line_norm), segm, eps=eps)


#%% plane
@_numba_njit()
def _intersect_plane_segm_(plane, segm, eps=1e-9):
    '''
    function returns intersection between plane and line (segment)
    '''
    segm_pt0, segm_pt1 = segm
    plane_point, plane_normal = plane

    segm_vector = segm_pt1 - segm_pt0
    plane_c = - plane_normal.dot(plane_point)

    up = plane_normal.dot(segm_pt0) + plane_c
    dn = plane_normal.dot(segm_vector)
    if abs(dn) < eps:
        # return np.full_like(segmPoint0, np.nan)
        return None, None

    t = - up/dn
    intersect_line_plane = segm_vector*t + segm_pt0

    return intersect_line_plane, t  # 0 <= t <= 1

#%% polygon
# @_numba_njit()
def ptInPolygon2D(P, pt, X=0, Y=1): # fastest variant

    # intersect = 0    # the crossing number counter
    # for i in range(len(P)):
    #     if ( (P[i][Y] <= pt[Y] and P[i-1][Y] >  pt[Y])or     # an upward crossing
    #          (P[i][Y] >  pt[Y] and P[i-1][Y] <= pt[Y])   ):  # a downward crossing

    #         t = (pt[Y] - P[i][Y]) / (P[i-1][Y] - P[i][Y])
    #         if pt[X]  <  P[i][X] + t*(P[i-1][X] - P[i][X]):  # pt[X] < x_intersect
    #             intersect += 1

    # return intersect % 2

    intersect = 0    # the crossing number counter
    for i in range(len(P)):
        if ( (P[i][Y] < pt[Y] and P[i-1][Y] > pt[Y])or     # an upward crossing
             (P[i][Y] > pt[Y] and P[i-1][Y] < pt[Y])   ):  # a downward crossing

            t = (pt[Y] - P[i][Y]) / (P[i-1][Y] - P[i][Y])
            if pt[X]  <  P[i][X] + t*(P[i-1][X] - P[i][X]):  # pt[X] < x_intersect
                intersect += 1

        elif (P[i][Y] == pt[Y] and P[i-1][Y] == pt[Y]):
            if (P[i][X] - pt[X])*(P[i-1][X]-pt[X]) < 0.:
                intersect += 1

    return intersect % 2

# @_numba_njit()
def _ptInPolygon3D_(P, pt): #  ???
    '''
    no check if point is in the plane of polygon
    This function is purposed only for the points that
    lie on the plane of the polygon
    '''
    l = len(P)//3
    normal = np.cross( P[l] - P[0], P[2*l] - P[0])
    # k = np.argmax(np.abs(normal)) #??? For numba
    k = np.argmax(abs(normal)) #??? Faster
    X, Y = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[k]
    return ptInPolygon2D(P, pt, X, Y)

# @_numba_njit()
def _regularPolygon3D(npoints, center, radius, normal, closed=False):
    i_std_mx = invMx( stdRotateMx(normal) )

    pts = []
    for i in range(npoints):
        ang = 2.0*np.pi*i/npoints
        pt = pt3D(radius*sin(ang), radius*cos(ang), 0.0)
        pt = i_std_mx.dot(pt) + center
        pts.append(pt)

    if closed:
        pts.append(pts[0])

    return pts

#%% cloud of points
def calc_gabarits_2D(points):
    xx = [pt[0] for pt in points]
    yy = [pt[1] for pt in points]

    xmin, xmax = np.min(xx), np.max(xx)
    ymin, ymax = np.min(yy), np.max(yy)

    return np.array([[xmin, ymin],
                     [xmax, ymax]])

def calc_gabarits(points):
    xx = [pt[0] for pt in points]
    yy = [pt[1] for pt in points]
    zz = [pt[2] for pt in points]

    xmin, xmax = np.min(xx), np.max(xx)
    ymin, ymax = np.min(yy), np.max(yy)
    zmin, zmax = np.min(zz), np.max(zz)

    return np.array([[xmin, ymin, zmin],
                     [xmax, ymax, zmax]])

def join_gabarits(gbr1, gbr2):
    if gbr1 is None:
        return gbr2
    elif gbr2 is None:
        return gbr1

    mins = np.min(   np.vstack((gbr1[0, :], gbr2[0, :])),  axis=0)
    maxs = np.max(   np.vstack((gbr1[1, :], gbr2[1, :])),  axis=0)
    return np.vstack( (mins, maxs) )

@_numba_njit()
def inside_gabarits(pt, gbr):
    return (  (pt[0] >= gbr[0, 0]) and (pt[1] >= gbr[0, 1]) and (pt[2] >= gbr[0, 2]) and
              (pt[0] <= gbr[1, 0]) and (pt[1] <= gbr[1, 1]) and (pt[2] <= gbr[1, 2])     )

@_numba_njit()
def outside_gabarits(pt, gbr):
    for i in range(len(pt)):
        if (pt[i] < gbr[0, i]) or (pt[i] > gbr[1, i]):
            return True
    return False

def testpoints(gbr, n, ndim=3):
    c = gbr[0]
    k = gbr[1] - gbr[0]
    return np.random.rand(n, ndim)*k + c

#%% plot
def get_coord_indexes(axes_code):
    axes_dict = {None: (0, 1), 'XY': (0, 1), 'XZ': (0, 2), 'ZY': (2, 1),
                               'YX': (1, 0),' ZX': (2, 0), 'YZ': (1, 2)}
    return axes_dict[axes_code]

def get_ortho_index(axes_code):
    axes_dict = {None: 2, 'XY': 2, 'XZ': 1, 'ZY': 0,
                          'YX': 2,' ZX': 1, 'YZ': 0}
    return axes_dict[axes_code]

def plot_point(pt, axes_code=None, **kwargs):
    X, Y = get_coord_indexes(axes_code)
    xx = [pt[X]]
    yy = [pt[Y]]
    return plt.plot(xx, yy, 'o', **kwargs) # ms, markersize

def plot_segm(p0, p1, axes_code=None, **kwargs):
    X, Y = get_coord_indexes(axes_code)
    return plt.plot([  p0[X], p1[X]  ],
                    [  p0[Y], p1[Y]  ], **kwargs)

def plot_polygon(points, axes_code=None, **kwargs):
    X, Y = get_coord_indexes(axes_code)

    xx = [pt[X] for pt in points]
    xx.append(points[0][X])

    yy = [pt[Y] for pt in points]
    yy.append(points[0][Y])

    return plt.plot(xx, yy, **kwargs)

#%%
def pt_in_ellipse_2D(R, a, elon):
    def f(pt):
        return ((pt[0] - R)**2 + (pt[1]/elon)**2) <= a
    return f

#%%
def direct_transform(obj, history):
    obj.transform(history._mx)
    obj.translate(history._vec)

def inverse_transform(obj, history):
    obj.translate(-history._vec)
    obj.transform(history._imx)
