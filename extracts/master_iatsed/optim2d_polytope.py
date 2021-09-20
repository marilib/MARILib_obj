#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 04 13:56 2021
@author: Nicolas Peteilh
"""

import numpy as np
import pandas as pd

# from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import random as rd

import extracts.master_iatsed.optim2d as opt2d
# from collections import Counter
# from matplotlib.tri import Triangulation, LinearTriInterpolator

import scipy.interpolate as sci_int


# def proj(pt, seg):
#     """
#
#     :param pt: is a list containing the coordinates of the point to be projected on the segment line
#     :param seg: is a list containing the 2 points defining the line on which we want to project pt
#     :return: a list containing the coordinates of the point projected on the segment line
#     """
#     A = np.array([seg[1][0]-seg[0][0], seg[1][1]-seg[0][1]])
#     b = np.array([    pt[0]-seg[0][0],     pt[1]-seg[0][1]])
#
#     u_hat = (A.T*A)**(-1) * A.T * b
#     p = A * u_hat
#
#     proj = [ele for ele in p + np.array(seg[0])]
#
#     return proj


def side_sign(p1, p2, p3):

    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def is_point_in_triangle(pt, v1, v2, v3):

    d1 = side_sign(pt, v1, v2)
    d2 = side_sign(pt, v2, v3)
    d3 = side_sign(pt, v3, v1)

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    return not(has_neg & has_pos)

# def flatten(t):
#     return [item for sublist in t for item in sublist]
#
#
# def is_crossing_(edge_a, edge_b):
#     # ===============================================================================================================
#     a1 = np.array(edge_a[0])
#     a2 = np.array(edge_a[1])
#     b1 = np.array(edge_b[0])
#     b2 = np.array(edge_b[1])
#     status = False
#     a21 = a2 - a1
#     if np.linalg.det([b1 - a1, a21]) * np.linalg.det([b2 - a1, a21]) < 0:
#         b21 = b2 - b1
#         if np.linalg.det([a1 - b1, b21]) * np.linalg.det([a2 - b1, b21]) < 0:
#             status = True
#     # ---------------------------------------------------------------------------------------------------------------
#     return status
#
#
# def intersection_of_(edge_a, edge_b):
#     a1 = np.array(edge_a[0])
#     a2 = np.array(edge_a[1])
#     b1 = np.array(edge_b[0])
#     b2 = np.array(edge_b[1])
#     # ===============================================================================================================
#     [da, db, dab] = (np.linalg.det([a2, a1]), np.linalg.det([b2, b1]), np.linalg.det([b1 - b2, a1 - a2]))
#     xi = np.linalg.det([[da, a1[0] - a2[0]], [db, b1[0] - b2[0]]]) / dab
#     yi = -np.linalg.det([[a1[1] - a2[1], da], [b1[1] - b2[1], db]]) / dab
#     xyi = [xi, yi]
#     # ---------------------------------------------------------------------------------------------------------------
#     return xyi
#
#
# def bnd_val(bnd, inpt):
#     # ===============================================================================================================
#     if bool(inpt):
#         out = inpt
#     else:
#         out = bnd
#     # ---------------------------------------------------------------------------------------------------------------
#     return out
#
#
# def expand_vector_list(list_in, factor=2):
#     """Expand the grid by a factor 2
#     """
#     expanded_vector_list = []
#     for pt in list_in:
#         expanded_vector_list.append([factor * pt[0], factor * pt[1]])
#     return expanded_vector_list
#
#
# def search_in_cells_around(zed_df_cell, P, N, X1, dX1, X2, dX2, crit_name, crit_ref, scaled_cst):
#     zed_df_piv = zed_df_cell.pivot(index="pt_x1", columns="pt_x2")
#     Cmin_list = []
#     dist = []
#     for k, p_pt in enumerate(P):
#         # print("p_pt = ", p_pt)
#         Y_ = []
#         D_ = []
#         C_ = []
#         for j, n_pt in enumerate(N):
#             # print("n_pt = ", n_pt)
#             [D, C, Y] = fct_scitwod_rs_([X1 + (n_pt[0] + p_pt[0]) * dX1, X2 + (n_pt[1] + p_pt[1]) * dX2],
#                                         zed_df_piv, crit_name,
#                                         scaled_cst)  # compute linear approx values
#             if (D == 0) & (crit_ref is None):
#                 crit_ref = C  # init criteria reference value
#             if crit_ref is not None:
#                 D = D + (
#                             C - crit_ref) / crit_ref  # if crit_ref is defined add this term to the point composite distance
#
#             D_.append(D)
#             C_.append(C)
#             Y_.append(Y)
#         Cmin_list.append(min(C_))
#         dist.append(sum(D_))
#
#     # find best cell candidate shift
#     # ---------------------------------------------------------------------------------------------------------------
#     print(Cmin_list)
#     shift_Cmin = np.argmin(Cmin_list)
#
#     # find best cell candidate
#     # ---------------------------------------------------------------------------------------------------------------
#     print(dist)
#     shift_Dsum = np.argmin(dist)
#
#     return crit_ref, shift_Cmin, shift_Dsum
#
#
# def initiate_storage_df(index_list, names_list):
#     zed_list = []  # initiate exact calculation storage
#     pt_eval_grid_list = []  # initiate exact calculation point list storage (to use as indices in zed_df)
#     df_index = pd.DataFrame(pt_eval_grid_list, columns=index_list, )
#     index_array = np.array(pt_eval_grid_list)
#     # idx = pd.MultiIndex.from_arrays(index_array.T, names=index_list)
#     zed_df = pd.DataFrame(zed_list,
#                           columns=names_list,
#                           #  index=pd.MultiIndex.from_arrays(index_array.T, names=index_list))
#                           index=pd.MultiIndex.from_frame(df_index))
#     return zed_df
#
#
# def expand_grid(zed_df):
#     zed_list_new = list(zed_df.values)  # .tolist()
#     pt_eval_grid_list_new = [list(ele) for ele in zed_df.index]
#     index_list = list(zed_df.index.names)
#     names_list = list(zed_df.columns)  # .to_list()
#
#     pt_eval_grid_list = expand_vector_list(pt_eval_grid_list_new, factor=2)
#
#     # df_index = pd.DataFrame(pt_eval_grid_list, columns=index_list, )
#     index_array = np.array(pt_eval_grid_list)
#     zed_df_new = pd.DataFrame(zed_list_new,
#                               columns=names_list,
#                               index=pd.MultiIndex.from_arrays(index_array.T, names=index_list))
#     # index=pd.MultiIndex.from_frame(df_index))
#     return zed_df_new
#
#
def compute_base_cell_data(zed_df, T, X1, X2, dX1, dX2, LwBs, LwFc, UpBs, UpFc, CrFc, crit_ref,
                           fct_SciTwoD, OneVar):  # compute base cell data
    # Update zed_df with base cell data

    # get information from existing zed_df
    zed_list_new = list(zed_df.values)  # .tolist()
    pt_eval_grid_list_new = [list(ele) for ele in zed_df.index]
    index_list = list(zed_df.index.names)
    names_list = list(zed_df.columns)

    new_cell = False
    # print("T = ", T)

    for cc in T:  # the objective of this loop is to iterate on N rows, i.e. on the 3 cell corners
        # print("cc : ", cc)
        pt_eval_grid = [int(cc[0]),
                        int(cc[1])]
        if ~zed_df.index.isin([tuple(pt_eval_grid)]).any():  # if the point has not been computed
            # print("len(dX1) = ", len(dX1))
            if len(dX1) == 1 and len(dX2) == 1:  # TODO : check if this is really useful
                pt_eval = [X1 + cc[0] * dX1, X2 + cc[1] * dX2]
            elif len(dX1) == 2 and len(dX2) == 2:
                pt_eval = [X1 + cc[0] * dX1[0] + cc[1] * dX2[0],
                           X2 + cc[0] * dX1[1] + cc[1] * dX2[1]]
            else:
                raise Exception("grid available only in 1D or 2D - review dX1 or dX2 size")

            print("pt_eval = ", pt_eval)
            [D, C, Y, V] = opt2d.fct_scitwod_ex_(
                pt_eval, LwBs, LwFc, UpBs, UpFc, CrFc, fct_SciTwoD, OneVar)  # compute exact values
            # print("D = ", D)
            # print("C = ", C)
            # print("Y = ", Y)
            # print("V = ", V)

            if (D == 0) & (crit_ref is None):
                crit_ref = C  # init criteria reference value
            if crit_ref is not None:
                D = D + (C - crit_ref) / crit_ref  # if crit_ref is defined, add it to the point composite distance

            zed_list_new.append(pt_eval + [C, D] + Y + V)
            pt_eval_grid_list_new.append(pt_eval_grid)

            new_cell = True

    # print("zed_list = ", zed_list)
    # df_index = pd.DataFrame(pt_eval_grid_list_new, columns=index_list, )
    index_array = np.array(pt_eval_grid_list_new)
    zed_df_new = pd.DataFrame(zed_list_new,
                              columns=names_list,
                              index=pd.MultiIndex.from_arrays(index_array.T, names=index_list))
    # index=pd.MultiIndex.from_frame(df_index))

    # print(zed_df_new)
    return zed_df_new, new_cell, crit_ref


def find_cst_points_in_cell(cell_df, param):
    cell_indices_list = list(cell_df.index)
    cell_indices_list.append(cell_indices_list[0])
    cst_bipoints = []
    cst_points_list = []

    # list of the vertices points
    ver_points_list = [list(cell_df[["pt_x1", "pt_x2"]].loc[ind]) for ind in cell_df.index]  # cell_indices_list[:-1]]

    # list of the sides (i.e. "vertices bipoints"). First : indices and then
    ver_bipoints_indices = [[cell_indices_list[j] for j in [i, i + 1]]
                            for i, ind in enumerate(cell_df.index)]
    ver_bipoints = [[list(cell_df[["pt_x1", "pt_x2"]].loc[ind]) for ind in side]
                    for side in ver_bipoints_indices]
    # print("ver_bipoints_indices = ", ver_bipoints_indices)
    # cell_pts = [[row["pt_x1"], row["pt_x2"]] for k, row in cell_df.iterrows()]
    # print("cell_pts = ", cell_pts)
    # cell_pts.append(cell_pts[0])
    # sides = [[cell_pts[i], cell_pts[i + 1]] for i, t in enumerate(cell_pts[:-1])]

    # for i, ind in enumerate(cell_indices_list[:-1]):
    #     # we
    #     side_indices = [cell_indices_list[j] for j in [i, i + 1]]
    for side_indices in ver_bipoints_indices:
        zed_df_side = cell_df[["pt_x1", "pt_x2", param]].loc[side_indices]
        # ver_points_list.append(list(zed_df_side[["pt_x1", "pt_x2"]].loc[zed_df_side.index[0]]))
        # ptx1 = list(zed_df_side["pt_x1"])  # .tolist()
        # ptx2 = list(zed_df_side["pt_x2"])  # .tolist()
        # ver_bipoints.append([[ptx1[0], ptx2[0]],
        #                      [ptx1[1], ptx2[1]]])
        zed_df_side = zed_df_side.sort_values(param)
        # print("zed_df_side = ", zed_df_side)
        ptx1 = list(zed_df_side["pt_x1"])  # .tolist()
        ptx2 = list(zed_df_side["pt_x2"])  # .tolist()

        # print("(zed_df_side[param] > 0).sum() = ", (zed_df_side[param] > 0).sum())
        if (zed_df_side[param] > 0).sum() == 1:
            # we can search the point
            cst_val = list(zed_df_side[param])  # .tolist()
            cst_ptx1 = np.interp(0, cst_val, ptx1)
            cst_ptx2 = np.interp(0, cst_val, ptx2)

            # # check [cst_ptx1, cst_ptx2] actually is IN the cell
            # cst_pt = [cst_ptx1, cst_ptx2]
            # 
            # pt_is_in_cell = is_point_in_triangle(cst_pt,
            #                                      v1=cell_pts[0],
            #                                      v2=cell_pts[1],
            #                                      v3=cell_pts[2])
            # 
            # print("Is cst_pt in cell_vx1, cell_vx2 triangle ? ", pt_is_in_cell)
            # 
            # # if not in the cell, we have to move it back in
            # if not pt_is_in_cell:
            #     side_dist = [side_sign(cst_pt, s[0], s[1]) for s in sides]
            #     closest_side = np.argmin(side_dist)


            cst_bipoints.append([cst_ptx1, cst_ptx2])
            cst_points_list.append([cst_ptx1, cst_ptx2])

    return cst_bipoints, cst_points_list


def find_best_opt_candidate(zed_df_cell, candidates_list, crit_name, scaled_cst, pt_tags):
    epsilon = 1e-10

    # zed_list_new = list(zed_df_cell.values)  # .tolist()
    # index_list = list(zed_df_cell.index.names)
    # names_list = list(zed_df_cell.columns)

    C_list = []
    Y_list = []
    for k, candidate in enumerate(candidates_list):
        # print("p_pt = ", p_pt)
        # print("n_pt = ", n_pt)
        [trash, C, Y] = fct_tritwod_rs_(candidate,  #[candidate[0], candidate[1]],
                                    zed_df_cell, crit_name,
                                    scaled_cst)  # compute linear approx values
        C_list.append(C)
        Y_list.append(Y)

    # print("C_list = ", C_list)

    candidates_df = pd.DataFrame([ele[0] + [ele[1]] + ele[2] + [ele[3]] for ele in zip(candidates_list,C_list,Y_list,pt_tags)],
                                 columns=['pt_x1', 'pt_x2'] + crit_name + scaled_cst + ['pt_tag'])

    # we keep the candidates that satisfy all the constraints
    # ---------------------------------------------------------------------------------------------------------------
    candidates_df_cst = candidates_df[(candidates_df[scaled_cst].T > epsilon).sum() == 0]

    # we sort the candidates by increasing
    # ---------------------------------------------------------------------------------------------------------------
    candidates_df_cst = candidates_df_cst.sort_values(crit_name)
    candidates_df_cst.reset_index(inplace=True)

    # # we sort the candidates by increasing
    # # ---------------------------------------------------------------------------------------------------------------
    # candidates_df_cst.sort_values(crit_name, inplace=True)
    # 
    # # get the indices that sorts C_list in ascending order
    # # ---------------------------------------------------------------------------------------------------------------
    # C_list_ordered_indices = np.argsort(np.array(C_list))
    # print("C_list_ordered_indices = ", C_list_ordered_indices.tolist())
    # # C_list_ordered = np.array(C_list)[C_list_ordered_indices.tolist()]
    # # print("C_list_ordered = ", C_list_ordered)
    # 
    # # keep "cst" and "int" points and "edg" satisfying all contraints
    # # ---------------------------------------------------------------------------------------------------------------
    # 
    # 
    # # keep the Y column corresponding to crossing constraints
    # # ---------------------------------------------------------------------------------------------------------------
    # Y_array = np.array(Y_list)
    # cross_Y_array = Y_array[:, np.any(Y_array > 0, axis=0)]
    # print("Y_array (= Y_list) = ", Y_array.tolist())
    # print("cross_Y_array = ", cross_Y_array.tolist())
    # 
    # # keep the point with Y negative (or close to zero) and minimum C
    # # ---------------------------------------------------------------------------------------------------------------
    # 
    # # in Y_list, sorted according to ascending order of C_list,
    # # ---------------------------------------------------------------------------------------------------------------
    # sorted_Ys = cross_Y_array[C_list_ordered_indices]
    # print("sorted_Ys = ", sorted_Ys.tolist())
    # 
    # # identify points satisfying all the constraints
    # # ---------------------------------------------------------------------------------------------------------------
    # do_satisfy_cst = [(y_row <= epsilon).all() for y_row in sorted_Ys]
    # print("do_satisfy_cst = ", do_satisfy_cst)
    # 
    # # keep the points that satisfy the constraints
    # # ---------------------------------------------------------------------------------------------------------------
    # C_list_ordered_indices = C_list_ordered_indices[do_satisfy_cst]
    # sorted_Ys = sorted_Ys[do_satisfy_cst]
    # # C_list_ordered = C_list[C_list_ordered_indices]
    # 
    # print("C_list_ordered_indices = ", C_list_ordered_indices.tolist())
    # print("sorted_Ys = ", sorted_Ys.tolist())
    # # print("C_list_ordered = ", C_list_ordered)
    # 
    # # find in sorted_Ys the first negative or close to zero element
    # sorted_Ys_sol_index = np.argwhere(sorted_Ys <= epsilon)
    # print("sorted_Ys_sol_index = ", sorted_Ys_sol_index)
    # 
    # best_C_index = C_list_ordered_indices[sorted_Ys_sol_index[0][0]]
    # best_C = C_list[best_C_index]
    # 
    # print("--------------------------> candidates_list         = ", candidates_list)
    # print("--------------------------> Y_array                 = ", Y_array.tolist())
    # print("--------------------------> cross_Y_array           = ", cross_Y_array.tolist())
    # print("--------------------------> C_list_ordered_indices  = ", C_list_ordered_indices.tolist())
    # print("--------------------------> sorted_Ys               = ", sorted_Ys.tolist())
    # print("--------------------------> sorted_Ys_sol_index     = ", sorted_Ys_sol_index.tolist())
    # print("--------------------------> best_C_index            = ", best_C_index)
    # print("--------------------------> best_C                  = ", best_C)

    return candidates_df_cst  # best_C_index, best_C


# def fct_scitwod_ex_(Xhyp, LwBs, LwFc, UpBs, UpFc, CrFc, fct_SciTwoD, OneVar):
#     # ===============================================================================================================
#     # LwBs : [list of down limit values]
#     # UpBs : [list of up limit values]
#     # ---------------------------------------------------------------------------------------------------------------
#
#     # # 2021_0804 - We evaluate the function at the point of evaluation
#     [LwCs, UpCs, Cr] = fct_SciTwoD(
#         Xhyp)  # the output of the function fct_SciTwoD include lower and upper constraints and the objective function (i.e. criteria)
#
#     # ---------------------------------------------------------------------------------------------------------------
#     if OneVar:
#         LwCs = [*LwCs, Xhyp(2)]
#         UpCs = [*UpCs, Xhyp(2)]
#
#     # ---------------------------------------------------------------------------------------------------------------
#     V = []
#     Y = []
#     for j, lw_cst in enumerate(LwCs):
#         V.append(lw_cst)
#         Y.append((LwBs[j] - lw_cst) / LwFc[j])
#
#     for j, up_cst in enumerate(UpCs):
#         V.append(up_cst)
#         Y.append((up_cst - UpBs[j]) / UpFc[j])
#
#     # ---------------------------------------------------------------------------------------------------------------
#     # composite distance (sum of individual relative distances grounded at zero)
#     D = sum([max(yy, 0) for yy in Y])  # if D == 0, all contraints are satisfied. if D > 0,
#
#     # ---------------------------------------------------------------------------------------------------------------
#     # criteria
#     C = 10 + Cr / CrFc
#     # ---------------------------------------------------------------------------------------------------------------
#     return [D, C, Y, V]  # [0, 1, 2, 3]
#
#
def fct_tritwod_rs_(xin, df_in, scaled_crit_name, scaled_cst_list):  # ff_vec, ss_vec, zz_mat):
    # =========================================================================================
    # Y relative distance to constraints SHOULD BE NEGATIVE WHEN CONSTRAINT IS SATISFIED
    Y = []
    # print("xin = ", xin)

    # ff_vec = list(df_piv.index)
    # ss_vec = list(df_piv.columns.levels[1])
    Xv = df_in["pt_x1"].values
    Yv = df_in["pt_x2"].values
    # print("Xv = ", Xv)
    # print("Yv = ", Yv)

    # print("Is xin in Xv,Yv triangle ? ", is_point_in_triangle(xin, v1=[Xv[0], Yv[0]], v2=[Xv[1], Yv[1]], v3=[Xv[2], Yv[2]]))

    # triObj = Triangulation(Xv,
    #                        Yv)  # Xv,Yv are the vertices (or nodes) of the triangles, and Zv the values at those nodes

    for param in scaled_cst_list:
        Zv = df_in[param].values
        # print("Zv = (Y)", Zv)
        # linear interpolation
        fz = sci_int.LinearNDInterpolator(list(zip(Xv, Yv)), Zv)
        # fz = LinearTriInterpolator(triObj, Zv)
        # Z = fz(X, Y)
        #
        # fy = RegularGridInterpolator((ff_vec, ss_vec),
        #                              data,
        #                              bounds_error=False,
        #                              fill_value=None)

        # print("fz(xin[0], xin[1])", fz(xin[0], xin[1]))
        Y.append(float(fz(xin[0], xin[1])))
        # print("fz(xin[0], xin[1]) = (Y)", fz(xin[0], xin[1]))
        # Y.append(fz(xin[0], xin[1]).data)
    # -----------------------------------------------------------------------------------------
    # criteria
    Zv = [ele for ele in opt2d.flatten(df_in[scaled_crit_name].values)]
    # print("Zv = (C)", Zv)
    # fc = RegularGridInterpolator((ff_vec, ss_vec),
    #                              data,
    #                              bounds_error=False,
    #                              fill_value=None)
    fz = sci_int.LinearNDInterpolator(list(zip(Xv, Yv)), Zv)
    # print("fz(xin[0], xin[1]) = (C)", fz(xin[0], xin[1]))
    C = float(fz(xin[0], xin[1]))

    # -----------------------------------------------------------------------------------------
    # global distance (sum of individual distances grounded at zero)
    # D is the relative composite distance to constraints
    D = sum([max(yy, 0) for yy in Y])  # if D == 0, all constraints are satisfied. if D > 0,

    # print("D = ", D)
    # print("C = ", C)

    # ---------------------------------------------------------------------------------------------------------------
    return [D, C, Y]


#
#
def scitwod_(X1, X2, dX1, dX2, noms, Units, Nzoom, LwBs, LwFc, UpBs, UpFc, CrFc, fct_SciTwoD, graph, varargin=[]):
    # =========================================================================================================================
    # global SciTwoD_gn, SciTwoD_gnum
    #
    # global GraphOptimSingle

    # if GraphOptimSingle:
    #     if isempty(SciTwoD_gn):
    #         SciTwoD_gn = get_window_id_()
    #         gn = SciTwoD_gn
    #         f = figure(gn)
    #         toolbar(gn, "off")
    #         f.figure_name = "Opt2D Monitoring"
    #         f.figure_position = [250, 50]
    #     else:
    #         gn = SciTwoD_gn
    #         f = figure(gn)
    # else:
    #     gn = get_window_id_()
    #     f = figure(gn)
    #     toolbar(gn, "off")
    #     f.figure_name = "Opt2D Monitoring"
    #     f.figure_position = [250, 50]

    # if graph == "tracker":
    #     draw_output = True
    #     a = gca()
    #     if ~isempty(a.children):
    #         delete(a.children)
    # elif graph == "yes":
    #     draw_output = True
    #     gn = []
    # else:
    #     draw_output = False;
    #     gn = []

    # if len(varargin) > 0:
    #     OneVar = True
    # else:
    #     OneVar = False
    # # print("OneVar : ", OneVar)

    sol = None
    zed_df_cell = None

    index_list = ["grid_x1", "grid_x2"]
    scaled_cst = ["scaled_" + cst_name for cst_name in noms]
    values_cst = ["values_" + cst_name for cst_name in noms]
    crit_name = ["scaled_crit"]
    names_list = ["pt_x1", "pt_x2"] + \
                 crit_name + \
                 ["D_dist"] + \
                 scaled_cst + \
                 values_cst

    # initiate tables
    # ---------------------------------------------------------------------------------------------------------------
    zed_df = opt2d.initiate_storage_df(index_list, names_list)

    new_pt_on_grid = [0, 0]  # defines the point of reference for the active cell in the grid ([0, 0] is [X1, X2])
    new_pt_on_grid_list = list()  # initiate cell reference point list storage
    new_pt_on_grid_list.append(new_pt_on_grid)

    CritRef = None  # no ref value is known at the beginning for the criteria

    T = [[0, 0],
         [1, 0],
         [0, 1]]  # define cell corner circuit (modified from scilab)

    # Nzoom: number of zooming cycle
    # ---------------------------------------------------------------------------------------------------------------
    for Nz in range(1, 1 + Nzoom):  # Nz goes from 1 to Nzoom

        print("======================")
        print("======================")
        print("Nz : ", Nz)

        # initiate search
        # ---------------------------------------------------------------------------------------------------------------
        new_cell = False

        # compute first base cell data

        zed_df, new_cell_proposed, CritRef = compute_base_cell_data(zed_df,
                                                                    T,
                                                                    X1, X2, dX1, dX2,
                                                                    LwBs, LwFc,
                                                                    UpBs, UpFc,
                                                                    CrFc, CritRef,
                                                                    fct_SciTwoD,
                                                                    OneVar=False)
        new_cell = new_cell or new_cell_proposed  # if new_cell_proposed is True, new_cell must be turned to True anyway

        # find point to be removed and set new T
        zed_df_cell = zed_df.loc[[tuple(ele) for ele in T]]
        zed_df_cell_sorted = zed_df_cell.sort_values(by=['D_dist'])
        zed_df_cell_sorted_ind_list = [list(ind) for ind in zed_df_cell_sorted.index]
        # print("zed_df_cell_sorted_ind_list = ", zed_df_cell_sorted_ind_list)
        new_T = zed_df_cell_sorted_ind_list[:-1]
        new_point = np.array(zed_df_cell_sorted_ind_list[0]) + \
                    (np.array(zed_df_cell_sorted_ind_list[1]) - np.array(zed_df_cell_sorted_ind_list[2]))
        new_T.append(list(new_point))
        new_pt_on_grid_list.append(new_point)

        # print("new_T = ", new_T)
        T = new_T

        # print(zed_df)
        # print(zed_df_cell)

        # stop when no surrounding cell has a better score than central one
        # or the search is comming back on an already visited cell
        # ---------------------------------------------------------------------------------------------------------------
        while new_cell:

            # # get base cell exact data from zed_df
            # # ---------------------------------------------------------------------------------------------------------------
            # new_cell_grid_point_list = []
            # for cc in T:  # the objective of this loop is to iterate on N rows, i.e. on the 4 cell corners
            #     # pt_eval_grid = [int(ref_cell_pt_on_grid[0]) + int(cc[0]), int(ref_cell_pt_on_grid[1]) + int(cc[1])]
            #     new_cell_grid_point_list.append(tuple(cc))
            # # print("new_cell_grid_point_list = ", new_cell_grid_point_list)

            new_cell = False

            # compute base cell data
            # ---------------------------------------------------------------------------------------------------------------
            zed_df, new_cell_proposed, CritRef = compute_base_cell_data(zed_df,
                                                                        T,
                                                                        X1, X2, dX1, dX2,
                                                                        LwBs, LwFc,
                                                                        UpBs, UpFc,
                                                                        CrFc, CritRef,
                                                                        fct_SciTwoD, OneVar=False)
            new_cell = new_cell or new_cell_proposed  # if new_cell_proposed is True, new_cell must be turned to True anyway

            if new_cell:
                zed_df_cell = zed_df.loc[[tuple(ele) for ele in T]]
                zed_df_cell_sorted = zed_df_cell.sort_values(by=['D_dist'])
                zed_df_cell_sorted_ind_list = [list(ind) for ind in zed_df_cell_sorted.index]
                # print("zed_df_cell_sorted_ind_list = ", zed_df_cell_sorted_ind_list)
                new_T = zed_df_cell_sorted_ind_list[:-1]
                new_point = np.array(zed_df_cell_sorted_ind_list[0]) + \
                            (np.array(zed_df_cell_sorted_ind_list[1]) - np.array(zed_df_cell_sorted_ind_list[2]))
                new_T.append(list(new_point))

                new_pt_on_grid_list.append(new_point)

                # print("new_T = ", new_T)
                T = new_T

            # new_cell = False

        if Nz < Nzoom:
            # expension of Zed matrix
            # ---------------------------------------------------------------------------------------------------------------
            zed_df = opt2d.expand_grid(zed_df)
            # print(zed_df)

            print("new_pt_on_grid_list = ", new_pt_on_grid_list)
            # change base cell co-ordinates and parameter shifts
            # ---------------------------------------------------------------------------------------------------------------
            new_pt_on_grid_list = opt2d.expand_vector_list(new_pt_on_grid_list, factor=2)
            print("new_pt_on_grid_list = (rescaled)", new_pt_on_grid_list)

            # print("T = ", T)
            new_T = opt2d.expand_vector_list([T[0]], factor=2)
            new_T.append(list(np.array(new_T[0]) + (np.array(T[1]) - np.array(T[0]))))
            new_T.append(list(np.array(new_T[0]) + (np.array(T[2]) - np.array(T[0]))))
            T = new_T
            # print("T (rescaled) = ", T)

            [dX1, dX2] = [dX1 / 2, dX2 / 2]
            # print(dX1, dX2)

            new_cell = True

    # check that at least one constraint crosses the last cell
    # if not, move toward a cell with better criteria
    # ---------------------------------------------------------------------------------------------------------------
    print("###########################################################@")
    print("Checking that constraints cross the last cell...")
    exist_cst_crossing = False
    while exist_cst_crossing is False:

        # ref_cell_pt_on_grid = ref_cell_pt_on_grid_list[-1]

        last_cell_grid_point_list = [tuple(cc) for cc in T]
        # for cc in T:  # the objective of this loop is to iterate on N rows, i.e. on the 4 cell corners
        #     # pt_eval_grid = [int(ref_cell_pt_on_grid[0]) + int(cc[0]), int(ref_cell_pt_on_grid[1]) + int(cc[1])]
        #     last_cell_grid_point_list.append(tuple(cc))

        zed_df_cell = zed_df.loc[last_cell_grid_point_list]
        cst_cross_sum = (zed_df_cell[scaled_cst] > 0).sum()

        # print("cst_cross_sum = ", cst_cross_sum)

        if ((cst_cross_sum != 3) & (cst_cross_sum != 0)).any():
            exist_cst_crossing = True
            print("--> At least one constraint crosses the last cell")

        # elif ~(((cst_cross_sum != 3) & (cst_cross_sum != 0)).all()):
        #     print(exist_cst_crossing)
        else:
            print("--> no constraint cross the cell. Looking for another cell...")
            # we search in the cells around for place where the criteria is reduced

        # zed_df_cell = zed_df.loc[[tuple(ele) for ele in T]]
        # zed_df_cell_sorted = zed_df_cell.sort_values(by=['scaled_crit'])
        # zed_df_cell_sorted_ind_list = [list(ind) for ind in zed_df_cell_sorted.index]
        # # print("zed_df_cell_sorted_ind_list = ", zed_df_cell_sorted_ind_list)
        # new_T = zed_df_cell_sorted_ind_list[:-1]
        # new_point = np.array(zed_df_cell_sorted_ind_list[0]) + \
        #             (np.array(zed_df_cell_sorted_ind_list[1]) - np.array(zed_df_cell_sorted_ind_list[2]))
        # new_T.append(list(new_point))
        #
        # new_pt_on_grid_list.append(new_point)
        # T = new_T
        # print("new_T = ", new_T)

        zed_df, new_cell_proposed, CritRef = compute_base_cell_data(zed_df,
                                                                    T,
                                                                    X1, X2, dX1, dX2,
                                                                    LwBs, LwFc,
                                                                    UpBs, UpFc,
                                                                    CrFc, CritRef,
                                                                    fct_SciTwoD, OneVar=False)
        # print("--> new cell (for crossing constraint) identified and calculated...")

        # exist_cst_crossing = exist_cst_crossing or new_cell_proposed

        if exist_cst_crossing is False:
            zed_df_cell = zed_df.loc[[tuple(ele) for ele in T]]
            zed_df_cell_sorted = zed_df_cell.sort_values(by=['scaled_crit'])
            zed_df_cell_sorted_ind_list = [list(ind) for ind in zed_df_cell_sorted.index]
            # print("zed_df_cell_sorted_ind_list = ", zed_df_cell_sorted_ind_list)
            new_T = zed_df_cell_sorted_ind_list[:-1]
            new_point = np.array(zed_df_cell_sorted_ind_list[0]) + \
                        (np.array(zed_df_cell_sorted_ind_list[1]) - np.array(zed_df_cell_sorted_ind_list[2]))
            new_T.append(list(new_point))
    
            new_pt_on_grid_list.append(new_point)
            T = new_T
            # print("new_T = ", new_T)

    # sub grid tracking when optimum is close to a frontiere
    # ---------------------------------------------------------------------------------------------------------------
    cells_on_grid_searched = []
    search_new_cell = True

    print("###########################################################@")
    print("Solving for optimum inside the last cell (with linearized local problem)...")
    while search_new_cell:

        print("--> extracting last cell...")
        last_cell_grid_point_list = [tuple(cc) for cc in T]
        # print("last_cell_grid_point_list = ", last_cell_grid_point_list)
        # for cc in T:  # the objective of this loop is to iterate on N rows, i.e. on the 4 cell corners
        #     # pt_eval_grid = [int(ref_cell_pt_on_grid[0]) + int(cc[0]), int(ref_cell_pt_on_grid[1]) + int(cc[1])]
        #     last_cell_grid_point_list.append(tuple(cc))

        zed_df_cell = zed_df.loc[last_cell_grid_point_list]
        cst_cross_sum = (zed_df_cell[scaled_cst] > 0).sum()
        # print("zed_df_cell = ", zed_df_cell)

        print("--> searching for candidate optima...")
        # edge_pts satisfying all the constraints are candidates
        edges_pts_list = [list(zed_df_cell[["pt_x1", "pt_x2"]].loc[ind])
                          for ind in zed_df_cell[scaled_cst][(zed_df_cell[scaled_cst].T > 0).sum()==0].index]

        # cst_pts are candidates
        csts_list = []  # list of [Ai, Bi] coordinates where Ai and Bi are one the constraint and on the cell boundaries
        cst_pts_list = []  # list of points Ai, Bi
        for param in cst_cross_sum.index:
            csts, cst_pts = find_cst_points_in_cell(zed_df_cell, param)
            if csts != []:
                csts_list.append(csts)
            cst_pts_list = cst_pts_list + cst_pts

        # # print("len(edges_list) = ", len(edges_list))
        # print("csts_list       = ", csts_list)
        # print("cst_pts_list    = ", cst_pts_list)
        # # print("edges_list      = ", edges_list)
        # print("edges_pts_list  = ", edges_pts_list)

        # points of intersections of the constraints are candidates
        intersec_pt_list = []
        for i, edge_1 in enumerate(csts_list):
            for j, edge_2 in enumerate(csts_list):  # enumerate(cst_edges_list[:i]+cst_edges_list[i+1:]):
                if opt2d.is_crossing_(edge_1, edge_2) & (j > i):
                    # print(i, j, edge_1, edge_2)
                    intersec_pt_list.append(opt2d.intersection_of_(edge_1, edge_2))
                    # print(intersec_pt_list)

        # print("intersec_pt_list = ", intersec_pt_list)

        # for cst in csts_list:
        #     plt.plot(np.array(cst).T[0], np.array(cst).T[1])
        # plt.scatter(np.array(intersec_pt_list).T[0], np.array(intersec_pt_list).T[1])
        # plt.show()

        print("--> select the best candidate...")
        all_opt_candidates_coord = cst_pts_list + edges_pts_list + intersec_pt_list
        # print("all_opt_candidates_coord = ", all_opt_candidates_coord)
        all_opt_candidates_tags = ["cst"] * len(cst_pts_list) + \
                                  ["edg"] * len(edges_pts_list) + \
                                  ["int"] * len(intersec_pt_list)
        print("all_opt_candidates_tags = ", all_opt_candidates_tags)

        best_candidates_df = find_best_opt_candidate(zed_df_cell, all_opt_candidates_coord, crit_name, scaled_cst, all_opt_candidates_tags)

        # print("best_candidates_df[0] = ", best_candidates_df.iloc[0])
        # print("all_opt_candidates_coord[bst_C_ind] = ", all_opt_candidates_coord[bst_C_ind])

        print("--> where is the best candidate...")
        if best_candidates_df['pt_tag'].iloc[0] == "int":  # all_opt_candidates_tags[bst_C_ind] == "int":
            print("----> the best solution is inside the cell, stop !")
            search_new_cell = False
            sol = best_candidates_df.iloc[0]

        elif best_candidates_df['pt_tag'].iloc[0] == "cst":
            print("----> the best solution is on a constraint on the cell border, then we move in this direction !")

            # we identify coordinates of the best point
            bst_pt = best_candidates_df[['pt_x1', 'pt_x2']].iloc[0]

            # we search the closest side of the cell
            tri_pts = [[row["pt_x1"], row["pt_x2"]] for k, row in zed_df_cell.iterrows()]
            tri_pts.append(tri_pts[0])
            sides = [[tri_pts[i], tri_pts[i + 1]] for i, t in enumerate(tri_pts[:-1])]
            oppsite_v_index = [int(np.mod(i-1,3)) for i in range(3)]
            side_dist = [abs(side_sign(bst_pt, s[0], s[1])) for s in sides]

            # the point opposite to the side the closest from the best point must be removed and the 2 others kept
            T_pt_index_to_be_removed = oppsite_v_index[np.argmin(side_dist)]
            del oppsite_v_index[np.argmin(side_dist)]
            T_pt_indices_to_keep = oppsite_v_index

            # we define the next point
            new_T = [T[ind] for ind in T_pt_indices_to_keep]
            # print("new_T = ", new_T)
            new_point = np.array(T[T_pt_indices_to_keep[0]]) + \
                        (np.array(T[T_pt_indices_to_keep[1]]) - np.array(T[T_pt_index_to_be_removed]))
            new_point = new_point.tolist()
            # print("new_point = ", new_point)
            new_T.append(new_point)  # list(new_point[0]))
            T = new_T
            new_pt_on_grid_list.append(new_point)
            cells_on_grid_searched.append([tuple(ele) for ele in T])

            zed_df, new_cell_proposed, CritRef = compute_base_cell_data(zed_df,
                                                                        T,
                                                                        X1, X2, dX1, dX2,
                                                                        LwBs, LwFc,
                                                                        UpBs, UpFc,
                                                                        CrFc, CritRef,
                                                                        fct_SciTwoD, OneVar=False)

        else:
            print("----> the best solution is on a vertex of the cell, then we move... but in which direction ??")
            # all_opt_candidates_coord[bst_C_ind]
            # where is it on the grid ?
            bst_pt = best_candidates_df.iloc[0]

            print("We move towards the best 'cst' side")
            # we search the best 'cst'
            bst_cst = best_candidates_df[['pt_x1', 'pt_x2']][best_candidates_df['pt_tag'] == 'cst'].iloc[0]

            # we search the closest side of the cell
            tri_pts = [[row["pt_x1"], row["pt_x2"]] for k, row in zed_df_cell.iterrows()]
            tri_pts.append(tri_pts[0])
            sides = [[tri_pts[i], tri_pts[i + 1]] for i, t in enumerate(tri_pts[:-1])]
            oppsite_v_index = [int(np.mod(i-1,3)) for i in range(3)]
            side_dist = [abs(side_sign(bst_cst, s[0], s[1])) for s in sides]

            # the point opposite to the side the closest from the best point must be removed and the 2 others kept
            T_pt_index_to_be_removed = oppsite_v_index[np.argmin(side_dist)]
            del oppsite_v_index[np.argmin(side_dist)]
            T_pt_indices_to_keep = oppsite_v_index

            # we define the next point
            new_T = [T[ind] for ind in T_pt_indices_to_keep]
            # print("new_T = ", new_T)
            new_point = np.array(T[T_pt_indices_to_keep[0]]) + \
                        (np.array(T[T_pt_indices_to_keep[1]]) - np.array(T[T_pt_index_to_be_removed]))
            new_point = new_point.tolist()
            # print("new_point = ", new_point)
            new_T.append(new_point)  # list(new_point[0]))

            # cells_on_grid_searched.append([tuple(ele) for ele in T])
            # 
            # opt_x1 = all_opt_candidates_coord[bst_C_ind][0]
            # opt_x2 = all_opt_candidates_coord[bst_C_ind][1]
            # opt_pt = [opt_x1, opt_x2]
            # 
            # # Search the side of the triangle in zed_df_cell the closest to the point
            # tri_pts = [[row["pt_x1"], row["pt_x2"]] for k, row in zed_df_cell.iterrows()]
            # tri_pts.append(tri_pts[0])
            # sides = [[tri_pts[i], tri_pts[i+1]] for i, t in enumerate(tri_pts[:-1])]
            # oppsite_v_index = [int(np.mod(i-1,3)) for i in range(3)]
            # side_dist = [side_sign(opt_pt, s[0], s[1]) for s in sides]
            # T_pt_index_to_be_removed = oppsite_v_index[np.argmin(side_dist)]
            # del oppsite_v_index[np.argmin(side_dist)]
            # T_pt_indices_to_keep = oppsite_v_index
            # 
            # # V # grid_x1_opt = (opt_x1 - X1) / dX1
            # # a # grid_x2_opt = (opt_x2 - X2) / dX2
            # # l #
            # # i # # get grid cell vertical and horizontal edges
            # # d # #    - get values related to vertical and horizontal edges ( i.e. those repeated twice in T)
            # #   # h_val = [d for d in Counter(np.array(T).T[1]) if Counter(np.array(T).T[1])[d] > 1][
            # # f #     0]  # correspond to grid_x2 value repeated twice in T
            # # o # v_val = [d for d in Counter(np.array(T).T[0]) if Counter(np.array(T).T[0])[d] > 1][
            # # r #     0]  # correspond to grid_x2 value repeated twice in T
            # #   #
            # # s # # shift_list = []  # contains only one or two elements (no more, no less)
            # # p # epsilon = 1e-4
            # # e # if abs(grid_x1_opt - v_val) < epsilon:
            # # c #     print("horizontal_move")
            # # i #     point_to_be_removed = [ele for ele in T if ele[0] != v_val]
            # # f #     new_T = [ele for ele in T if ele[0] == v_val]
            # # i # elif abs(grid_x2_opt - h_val) < epsilon:
            # # c #     print("vertical_move")
            # #   #     point_to_be_removed = [ele for ele in T if ele[1] != h_val]
            # # t #     new_T = [ele for ele in T if ele[1] == h_val]
            # # r # else:
            # # i #     point_to_be_removed = [ele for ele in T if (ele[0] == v_val and ele[1] == h_val)]
            # # a #     new_T = [ele for ele in T if ~(ele[0] == v_val and ele[1] == h_val)]
            # # n #
            # new_T = [T[ind] for ind in T_pt_indices_to_keep]
            # # print("new_T = ", new_T)
            # new_point = np.array(T[T_pt_indices_to_keep[0]]) + \
            #             (np.array(T[T_pt_indices_to_keep[1]]) - np.array(T[T_pt_index_to_be_removed]))
            # new_point = new_point.tolist()
            # # print("new_point = ", new_point)
            # new_T.append(new_point)  # list(new_point[0]))
            # 
            # # new_pt_on_grid_list.append(new_point)
            # # print("T = ", T)
            # # print("new_T = ", new_T)
            # 
            # # print("[opt_x1, opt_x2] = ", [opt_x1, opt_x2])
            # # print("[grid_x1_opt, grid_x2_opt] = ", [grid_x1_opt, grid_x2_opt])
            # # print("shift_list = ", shift_list)
            # # # if the ref_cell_pt_on_grid is already in zed_df,
            # # # then this means the new cell has already been searched
            # # # then last opt candidate is kept as the solution
            # #
            # # # consider the first possible shift_ in shift_list.
            # # shift_ = shift_list[0]
            # # print("ref_cell_pt_on_grid_list = ", ref_cell_pt_on_grid_list)
            # # print("ref_cell_pt_on_grid = ", ref_cell_pt_on_grid)
            # # ref_cell_pt_on_grid = list(np.array(ref_cell_pt_on_grid) + np.array(shift_))
            # # print("ref_cell_pt_on_grid = ", ref_cell_pt_on_grid)
            # # print("zed_df.index = ", zed_df.index)
            # new_pt_on_grid_list_tuples = [tuple(ele) for ele in new_pt_on_grid_list]
            # print("new_pt_on_grid_list_tuples = ", new_pt_on_grid_list_tuples)
            # print("new_point (A) = ", new_point)
            # print("type(new_point) = ", type(new_point))
            # # print("tuple(new_point[0]) = ", tuple(new_point[0]))

            # look if the cell has already been searched, i.e. look if new_T is in cells_on_grid_searched
            new_T_tuple = [tuple(ele) for ele in new_T]
            # print("new_T_tuple = ", new_T_tuple)
            # print("cells_on_grid_searched = ", cells_on_grid_searched)
            searched = any([all([ele in new_T_tuple for ele in toto1]) for toto1 in cells_on_grid_searched])

            print("searched = ", searched)

            if not searched:  # tuple(new_point[0]) not in new_pt_on_grid_list_tuples: # ~zed_df.index.isin([tuple(ref_cell_pt_on_grid)]).any():
                print("------> ref_cell_pt_on_grid has not already been calculated")
                # print(ref_cell_pt_on_grid)
                new_pt_on_grid_list.append(new_point[0])
                T = new_T

                zed_df, new_cell_proposed, CritRef = compute_base_cell_data(zed_df,
                                                                            T,
                                                                            X1, X2, dX1, dX2,
                                                                            LwBs, LwFc,
                                                                            UpBs, UpFc,
                                                                            CrFc, CritRef,
                                                                            fct_SciTwoD, OneVar=False)
                print("--> new cell identified and calculated...")
                # print("zed_df.index = ", zed_df.index)
                # 
                # print("--> extracting last cell...")
                # last_cell_grid_point_list = [last_cell_grid_point_list.append(tuple(cc)) for cc in T]
                # # for cc in T:  # the objective of this loop is to iterate on N rows, i.e. on the 4 cell corners
                # #     # pt_eval_grid = [int(ref_cell_pt_on_grid[0]) + int(cc[0]), int(ref_cell_pt_on_grid[1]) + int(cc[1])]
                # #     last_cell_grid_point_list.append(tuple(cc))
                # 
                # zed_df_cell = zed_df.loc[last_cell_grid_point_list]
                # cst_cross_sum = (zed_df_cell[scaled_cst] > 0).sum()
                # 
                # if max(cst_cross_sum) == 3:
                #     # this means one constraint is not satisfied by any corner of the cell
                #     # then stop the iteration
                #     print("------> shifting the cell to find a better optimum is not possible")
                #     # search is over
                #     sol = [opt_x1, opt_x2]
                #     search_new_cell = False
                # else:
                #     print("------> search for a new cell continues")
                #     # we replace the worst element of T
                #     zed_df_cell = zed_df.loc[[tuple(ele) for ele in T]]
                #     zed_df_cell_sorted = zed_df_cell.sort_values(by=['scaled_crit'])
                #     zed_df_cell_sorted_ind_list = [list(ind) for ind in zed_df_cell_sorted.index]
                #     # print("zed_df_cell_sorted_ind_list = ", zed_df_cell_sorted_ind_list)
                #     new_T = zed_df_cell_sorted_ind_list[:-1]
                #     new_point = np.array(zed_df_cell_sorted_ind_list[0]) + \
                #                 (np.array(zed_df_cell_sorted_ind_list[1]) - np.array(zed_df_cell_sorted_ind_list[2]))
                #     print("new_point (B) = ", new_point)
                #     print("type(new_point) = ", type(new_point))
                #     new_T.append(list(new_point))
                #     print("new_T = ", new_T)

            else:
                print("------> cell has already been searched - search is over")
                # ref_cell_pt_on_grid = ref_cell_pt_on_grid_list[-1]

                # we search the cell and is no solution we get out
                sol = bst_pt
                search_new_cell = False

        # search_new_cell = False
    return zed_df, new_pt_on_grid_list, zed_df_cell, sol  # XYCopt, gnum


# # ================================================================================================================
# #
# #	2D optimisation
# #
# # ================================================================================================================
#
def optim2d_poly(Xini, dXini, Names, Units, Nzoom, LwBs, LwFc, UpBs, UpFc, CrFc, fct_SciTwoD, graph):
    # ===============================================================================================================
    # 2021_0804 : Xini, LwBs, UpBs, Names, Units, LwFc, UpFc are lists
    #             Nzoom = number of zooming cycles
    n = len(Xini)  # size(Xini,'*')

    if n == 1:
        raise Exception("Does not work in 1D !")
        # nL = len(LwBs)
        # nU = len(UpBs)
        # X1 = Xini[0]  # first element of Xini list
        # dX1 = dXini[0]  # first element of dXini list
        # X2 = 0
        # dX2 = 0.1
        # Names = [Names[0], "dummy", *Names[1:1 + nL], "low", *Names[1 + nL:1 + nL + nU], "up"]
        # LwBs_ = [*LwBs, -1]
        # LwFc_ = [*LwFc, 10]
        # UpBs_ = [*UpBs, 1]
        # UpFc_ = [*UpFc, 10]

        # print(nL   )
        # print(nU   )
        # print(X1   )
        # print(dX1  )
        # print(X2   )
        # print(dX2  )
        # print(Names)
        # print(Units)
        # print(LwBs_)
        # print(LwFc_)
        # print(UpBs_)
        # print(UpFc_)

        # zed_df, ref_cell_pt_on_grid_list = scitwod_(X1, X2, dX1, dX2, Names, Units, Nzoom, LwBs_, LwFc_, UpBs_, UpFc_,
        #                                             CrFc, fct_SciTwoD, graph, "OneVar")
        # [XYCopt,gnum] = SciTwoD_(X1,X2,dX1,dX2,Names,Units,Nzoom,LwBs,LwFc,UpBs,UpFc,CrFc,fct_SciTwoD,graph,"OneVar")
    elif n == 2:
        X1 = Xini[0]
        dX1 = dXini[0]
        X2 = Xini[1]
        dX2 = dXini[1]

        # print(X1 )
        # print(dX1)
        # print(X2 )
        # print(dX2)

        zed_df, new_pt_on_grid_list, zed_df_cell, sol = scitwod_(X1, X2, dX1, dX2, Names, Units, Nzoom, LwBs, LwFc,
                                                                 UpBs, UpFc, CrFc, fct_SciTwoD, graph, )
        # [XYCopt,gnum] = scitwod_(X1,X2,dX1,dX2,Names,Units,Nzoom,LwBs_,LwFc_,UpBs_,UpFc_,CrFc,fct_SciTwoD,graph)
    else:
        raise Exception("optim2d cannot handle more than 2 degrees of freedom")

    # if n==1:
    #     Y = real(XYCopt(1,1))
    # elif n==2:
    #     Y = real(XYCopt(1,1:2))
    #
    # Cr = real(XYCopt(1,3))
    # ---------------------------------------------------------------------------------------------------------------
    return zed_df, new_pt_on_grid_list, zed_df_cell, sol  # [Y,Cr]


def my_criteria(x1, x2):
    return (x1 + 50) ** 2 + (x2 + 35) ** 2


def my_lower_constraints(x1, x2):
    return [x1, x2, x1 + x2]


def my_upper_constraints(x1, x2):
    return []  # [x1+x2]


def my_fct_scitwod(x):
    x1 = x[0]
    x2 = x[1]
    return [my_lower_constraints(x1, x2),
            my_upper_constraints(x1, x2),
            my_criteria(x1, x2)]


if __name__ == "__main__":

    xini = [rd.uniform(-10, 10), rd.uniform(-10, 10)]  # [10, 10]  # [3.5, 5.2]
    initial_scale = 2
    dxini = [initial_scale * np.array([  1,             0]),
             initial_scale * np.array([0.5, np.sqrt(3)/2])]  # [rd.uniform(0.1, 2), rd.uniform(0.1, 2)]
    nzoom = 2

    # set lower bounds information : gi(x) >= lwbs
    names_lwbs = ["toto_lwb1", "toto_lwb2", "toto_lwbs3"]
    units_lwbs = ["no_dim", "no_dim", "no_dim"]
    lwbs = [rd.uniform(-8, 5), rd.uniform(-4, 4), rd.uniform(-4, 4)]  # [-4, -1]
    lwfc = [0.5, 0.5, 0.5]  # [1, 1]

    # set upper bounds information : hi(x) <= upbs
    names_upbs = []  # ["toto_upb1"]
    units_upbs = []  # ["no_dim"]
    if sum(lwbs) > 0:
        upbs = []  # [rd.uniform(sum(lwbs), max(2*sum(lwbs),5))]  # [10]
    else:
        upbs = []  # [rd.uniform(0, 10)]
    upfc = []  # [0.5]

    # set criteria information : f(x)
    names_crit = ["crit"]
    units_crit = ["no_dim"]
    crfc = 1

    # set function information [gi(x), hj(x), f(x)
    fct_scitwod = my_fct_scitwod
    names = names_lwbs + names_upbs  # + names_crit
    units = units_lwbs + units_upbs  # + units_crit
    graph = None

    my_zed_df, ref_cell_pts_on_grid_list, my_zed_df_cell, my_sol = optim2d_poly(xini,
                                                                                dxini,
                                                                                names,
                                                                                units,
                                                                                nzoom,
                                                                                lwbs,
                                                                                lwfc,
                                                                                upbs,
                                                                                upfc,
                                                                                crfc,
                                                                                fct_scitwod,
                                                                                graph)

    my_zed_df.to_html('my_zed_df.html')
    my_zed_df_cell.to_html('my_zed_df_cell.html')

    print("The solution is : ", my_sol)
    ref_cell_pt_list = []
    for ref_cell_pts_on_grid in ref_cell_pts_on_grid_list:
        ref_cell_pt = [xini[0] + (ref_cell_pts_on_grid[0] * dxini[0][0] + ref_cell_pts_on_grid[1] * dxini[1][0]) / (2 ** (nzoom - 1)),
                       xini[1] + (ref_cell_pts_on_grid[0] * dxini[0][1] + ref_cell_pts_on_grid[1] * dxini[1][1]) / (2 ** (nzoom - 1))]
        
        ref_cell_pt_list.append(ref_cell_pt)

    # =========================
    # plot the results
    delta = 0.025
    x1 = np.arange(-10.0, 10.0, delta)
    x2 = np.arange(-10.0, 10.0, delta)
    X1, X2 = np.meshgrid(x1, x2)
    Z = my_criteria(X1, X2)

    plt.imshow(Z,
               extent=[-10, 10, -10, 10],
               origin="lower")
    plt.contour(X1, X2, Z, 15, colors='g')
    plt.vlines(lwbs[0], min(x2), max(x2), "r")
    plt.hlines(lwbs[1], min(x1), max(x1), "r")
    plt.plot(x1, -x1 + lwbs[2], '-r')  # upbs[0], '-r')
    # plt.scatter(np.array(my_sol).T[0], np.array(my_sol).T[1], marker="x", c="yellow")
    plt.scatter(my_sol["pt_x1"], my_sol["pt_x2"], marker="x", c="yellow")
    plt.xlim([min(x1), max(x1)])
    plt.ylim([min(x2), max(x2)])
    plt.xlabel("x1")
    plt.ylabel("x2")
    str_xini = "xini (" + str(xini[0]) + ", " + str(xini[1]) + ")"
    str_lwbs = "lwbs (" + str(lwbs[0]) + ", " + str(lwbs[1]) + ", " + str(lwbs[2]) + ")"
    str_upbs = "no upbs"  # "upbs (" + str(upbs[0]) + ")"

    plt.title(str_xini + "\n" +
              str_lwbs + "\n" +
              str_upbs + "\n" +
              str(ref_cell_pt_list[-1]) + "\n" +
              "my_sol = " + str([my_sol["pt_x1"], my_sol["pt_x2"]]))

    plt.plot(list(map(list, zip(*ref_cell_pt_list)))[0],
             list(map(list, zip(*ref_cell_pt_list)))[1])

    plt.plot(my_zed_df_cell["pt_x1"], my_zed_df_cell["pt_x2"], "y")
    plt.show()

    ## # Test is_crossing and intersection_of_
    ## a_1 = np.array([2, 3])
    ## a_2 = np.array([6, 6])
    ## b_1 = np.array([4, 1])
    ## b_2 = np.array([4, 6])
    ##
    ## status = is_crossing_(a_1, a_2, b_1, b_2)
    ##
    ## if status:
    ##     ixy = intersection_of_(a_1, a_2, b_1, b_2)
    ##     print("intersection coordinates are : ", ixy)
    ## else:
    ##     print("no intersection")
    ##
    ## # Test of bnd_val
    ## print(bnd_val(10., 2.))

    # epsilon = 1e-4
    # n = 8
    # C_list = []
    # Y_list = []
    # for i in range(n):
    #     C_list.append(rd.uniform(0, 10))
    #     Y_list.append([rd.uniform(-1, 1), rd.uniform(-1, 1)])
    # Y_array = np.array(Y_list)
    # C_array = np.array(C_list)
    # C_list_ordered_indices = np.argsort(np.array(C_list))
    # sorted_Ys = Y_array[C_list_ordered_indices]
    # sorted_Cs = C_array[C_list_ordered_indices]
    # do_satisfy_cst = [(y_row <= epsilon).all() for y_row in sorted_Ys]







