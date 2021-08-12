#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 04 13:56 2021
@author: Nicolas Peteilh
"""

import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import random as rd


def flatten(t):
    return [item for sublist in t for item in sublist]


def is_crossing_(edge_a, edge_b):
    # ===============================================================================================================
    a1 = np.array(edge_a[0])
    a2 = np.array(edge_a[1])
    b1 = np.array(edge_b[0])
    b2 = np.array(edge_b[1])
    status = False
    a21 = a2 - a1
    if np.linalg.det([b1 - a1, a21]) * np.linalg.det([b2 - a1, a21]) < 0:
        b21 = b2 - b1
        if np.linalg.det([a1 - b1, b21]) * np.linalg.det([a2 - b1, b21]) < 0:
            status = True
    # ---------------------------------------------------------------------------------------------------------------
    return status


def intersection_of_(edge_a, edge_b):
    a1 = np.array(edge_a[0])
    a2 = np.array(edge_a[1])
    b1 = np.array(edge_b[0])
    b2 = np.array(edge_b[1])
    # ===============================================================================================================
    [da, db, dab] = (np.linalg.det([a2, a1]), np.linalg.det([b2, b1]), np.linalg.det([b1 - b2, a1 - a2]))
    xi = np.linalg.det([[da, a1[0] - a2[0]], [db, b1[0] - b2[0]]]) / dab
    yi = -np.linalg.det([[a1[1] - a2[1], da], [b1[1] - b2[1], db]]) / dab
    xyi = [xi, yi]
    # ---------------------------------------------------------------------------------------------------------------
    return xyi


def bnd_val(bnd, inpt):
    # ===============================================================================================================
    if bool(inpt):
        out = inpt
    else:
        out = bnd
    # ---------------------------------------------------------------------------------------------------------------
    return out


def expand_vector_list(list_in, factor=2):
    """Expand the grid by a factor 2
    """
    expanded_vector_list = []
    for pt in list_in:
        expanded_vector_list.append([factor * pt[0], factor * pt[1]])
    return expanded_vector_list


def search_in_cells_around(zed_df_cell, P, N, X1, dX1, X2, dX2, crit_name, crit_ref, scaled_cst):
    zed_df_piv = zed_df_cell.pivot(index="pt_x1", columns="pt_x2")
    Cmin_list = []
    dist = []
    for k, p_pt in enumerate(P):
        # print("p_pt = ", p_pt)
        Y_ = []
        D_ = []
        C_ = []
        for j, n_pt in enumerate(N):
            # print("n_pt = ", n_pt)
            [D, C, Y] = fct_scitwod_rs_([X1 + (n_pt[0] + p_pt[0]) * dX1, X2 + (n_pt[1] + p_pt[1]) * dX2],
                                        zed_df_piv, crit_name,
                                        scaled_cst)  # compute linear approx values
            if (D == 0) & (crit_ref is None):
                crit_ref = C  # init criteria reference value
            if crit_ref is not None:
                D = D + (
                            C - crit_ref) / crit_ref  # if crit_ref is defined add this term to the point composite distance

            D_.append(D)
            C_.append(C)
            Y_.append(Y)
        Cmin_list.append(min(C_))
        dist.append(sum(D_))

    # find best cell candidate shift
    # ---------------------------------------------------------------------------------------------------------------
    # print(Cmin_list)
    shift_Cmin = np.argmin(Cmin_list)

    # find best cell candidate
    # ---------------------------------------------------------------------------------------------------------------
    # print(dist)
    shift_Dsum = np.argmin(dist)

    return crit_ref, shift_Cmin, shift_Dsum


def initiate_storage_df(index_list, names_list):
    zed_list = []  # initiate exact calculation storage
    pt_eval_grid_list = []  # initiate exact calculation point list storage (to use as indices in zed_df)
    # df_index = pd.DataFrame(pt_eval_grid_list, columns=index_list, )
    index_array = np.array(pt_eval_grid_list)
    idx = pd.MultiIndex.from_tuples([(None, None)])
    idx.names = index_list
    zed_df = pd.DataFrame(zed_list,
                          columns=names_list,
                          index=idx)  # pd.MultiIndex.set_names(names=index_list))
    # index=pd.MultiIndex.from_frame(df_index))
    return zed_df


def expand_grid(zed_df):
    zed_list_new = list(zed_df.values)  # .tolist()
    pt_eval_grid_list_new = [list(ele) for ele in zed_df.index]
    index_list = list(zed_df.index.names)
    names_list = list(zed_df.columns)  # .to_list()

    pt_eval_grid_list = expand_vector_list(pt_eval_grid_list_new, factor=2)

    # df_index = pd.DataFrame(pt_eval_grid_list, columns=index_list, )
    index_array = np.array(pt_eval_grid_list)
    zed_df_new = pd.DataFrame(zed_list_new,
                              columns=names_list,
                              index=pd.MultiIndex.from_arrays(index_array.T, names=index_list))
    # index=pd.MultiIndex.from_frame(df_index))
    return zed_df_new


def compute_base_cell_data(zed_df, ref_cell_point_on_grid, N, X1, X2, dX1, dX2, LwBs, LwFc, UpBs, UpFc, CrFc, crit_ref,
                           fct_SciTwoD, OneVar):  # compute base cell data
    # Update zed_df with base cell data

    # get information from existing zed_df
    zed_list_new = list(zed_df.values)  # .tolist()
    pt_eval_grid_list_new = [list(ele) for ele in zed_df.index]
    # print("zed_df.index.names = ", zed_df.index.names)
    index_list = list(zed_df.index.names)
    names_list = list(zed_df.columns)

    new_cell = False

    for cc in N:  # the objective of this loop is to iterate on N rows, i.e. on the 4 cell corners
        # print("cc : ", cc)
        pt_eval_grid = [int(ref_cell_point_on_grid[0]) + int(cc[0]),
                        int(ref_cell_point_on_grid[1]) + int(cc[1])]
        if ~zed_df.index.isin([tuple(pt_eval_grid)]).any():  # if the point has not been computed
            pt_eval = [X1 + cc[0] * dX1, X2 + cc[1] * dX2]

            [D, C, Y, V] = fct_scitwod_ex_(
                pt_eval, LwBs, LwFc, UpBs, UpFc, CrFc, fct_SciTwoD, OneVar)  # compute exact values
            # print("D = ", D)
            # print("C = ", C)
            # print("Y = ", Y)
            # print("V = ", V)

            if (D == 0) & (crit_ref is None):
                crit_ref = C  # init criteria reference value

            zed_list_new.append(pt_eval + [C, D] + Y + V)
            pt_eval_grid_list_new.append(pt_eval_grid)

            new_cell = True

    # print("zed_list = ", zed_list)
    # df_index = pd.DataFrame(pt_eval_grid_list_new, columns=index_list, )
    index_array = np.array(pt_eval_grid_list_new)
    # print("index_array = ", index_array)
    # print("index_list = ", index_list)
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
    ver_bipoints = []
    cst_points_list = []
    ver_points_list = []
    for i, ind in enumerate(cell_indices_list[:-1]):
        side_indices = [cell_indices_list[j] for j in [i, i + 1]]
        zed_df_side = cell_df[["pt_x1", "pt_x2", param]].loc[side_indices]
        ver_points_list.append(list(zed_df_side[["pt_x1", "pt_x2"]].loc[zed_df_side.index[0]]))
        ptx1 = list(zed_df_side["pt_x1"])  # .tolist()
        ptx2 = list(zed_df_side["pt_x2"])  # .tolist()
        ver_bipoints.append([[ptx1[0], ptx2[0]],
                             [ptx1[1], ptx2[1]]])
        zed_df_side = zed_df_side.sort_values(param)
        ptx1 = list(zed_df_side["pt_x1"])  # .tolist()
        ptx2 = list(zed_df_side["pt_x2"])  # .tolist()
        if (zed_df_side[param] > 0).sum() == 1:
            # we can search the point
            cst_val = list(zed_df_side[param])  # .tolist()
            cst_ptx1 = np.interp(0, cst_val, ptx1)
            cst_ptx2 = np.interp(0, cst_val, ptx2)
            cst_bipoints.append([cst_ptx1, cst_ptx2])
            cst_points_list.append([cst_ptx1, cst_ptx2])
        else:
            pass
    return ver_bipoints, ver_points_list, cst_bipoints, cst_points_list


def find_best_opt_candidate(zed_df_cell, candidates_list, crit_name, scaled_cst):
    epsilon = 1e-4
    zed_df_piv = zed_df_cell.pivot(index="pt_x1", columns="pt_x2")
    C_list = []
    Y_list = []
    for k, candidate in enumerate(candidates_list):
        # print("p_pt = ", p_pt)
        # print("n_pt = ", n_pt)
        [D, C, Y] = fct_scitwod_rs_([candidate[0], candidate[1]],
                                    zed_df_piv, crit_name,
                                    scaled_cst)  # compute linear approx values
        C_list.append(C)
        Y_list.append(Y)

    # keep the Y column corresponding to crossing constraints
    # ---------------------------------------------------------------------------------------------------------------
    Y_array = np.array(Y_list)
    cross_Y_array = Y_array[:, np.any(Y_array > epsilon, axis=0)]

    # keep the point with Y negative (or close to zero) and minimum C
    # ---------------------------------------------------------------------------------------------------------------
    # get the indices that sorts C_list in ascending order
    C_list_ordered_indices = np.argsort(np.array(C_list))

    # in Y_list, sorted according to ascending order of C_list,
    # ---------------------------------------------------------------------------------------------------------------
    sorted_Ys = cross_Y_array[C_list_ordered_indices]

    # identify points satisfying all the constraints
    # ---------------------------------------------------------------------------------------------------------------
    do_satisfy_cst = [(y_row <= epsilon).all() for y_row in sorted_Ys]

    # keep the points that satisfy the constraints
    # ---------------------------------------------------------------------------------------------------------------
    C_list_ordered_indices = C_list_ordered_indices[do_satisfy_cst]
    # sorted_Ys = sorted_Ys[do_satisfy_cst]

    # find in sorted_Ys the first negative or close to zero element
    # sorted_Ys_sol_index = np.argwhere(sorted_Ys < epsilon)

    # best_C_index = C_list_ordered_indices[sorted_Ys_sol_index[0][0]]
    best_C_index = C_list_ordered_indices[0]
    best_C = C_list[best_C_index]

    print("--------------------------> candidates_list         = ", candidates_list)
    print("--------------------------> Y_array                 = ", Y_array)
    print("--------------------------> cross_Y_array           = ", cross_Y_array)
    print("--------------------------> C_list_ordered_indices  = ", C_list_ordered_indices)
    print("--------------------------> sorted_Ys               = ", sorted_Ys)
    # print("--------------------------> sorted_Ys_sol_index     = ", sorted_Ys_sol_index)
    print("--------------------------> best_C_index            = ", best_C_index)
    print("--------------------------> best_C                  = ", best_C)

    return best_C_index, best_C


def fct_scitwod_ex_(Xhyp, LwBs, LwFc, UpBs, UpFc, CrFc, fct_SciTwoD, OneVar):
    # ===============================================================================================================
    # LwBs : [list of down limit values]
    # UpBs : [list of up limit values]
    # ---------------------------------------------------------------------------------------------------------------

    # # 2021_0804 - We evaluate the function at the point of evaluation
    [LwCs, UpCs, Cr] = fct_SciTwoD(Xhyp)  # the output of the function fct_SciTwoD include lower and upper constraints and the objective function (i.e. criteria)

    # ---------------------------------------------------------------------------------------------------------------
    if OneVar:
        LwCs = [*LwCs, Xhyp(2)]
        UpCs = [*UpCs, Xhyp(2)]

    # ---------------------------------------------------------------------------------------------------------------
    V = []
    Y = []
    for j, lw_cst in enumerate(LwCs):
        V.append(lw_cst)
        Y.append((LwBs[j] - lw_cst) / LwFc[j])

    for j, up_cst in enumerate(UpCs):
        V.append(up_cst)
        Y.append((up_cst - UpBs[j]) / UpFc[j])

    # ---------------------------------------------------------------------------------------------------------------
    # composite distance (sum of individual relative distances grounded at zero)
    D = sum([max(yy, 0) for yy in Y])  # if D == 0, all contraints are satisfied. if D > 0,

    # ---------------------------------------------------------------------------------------------------------------
    # criteria
    C = 10 + Cr / CrFc
    # ---------------------------------------------------------------------------------------------------------------
    return [D, C, Y, V]  # [0, 1, 2, 3]


def fct_scitwod_rs_(xin, df_piv, scaled_crit_name, scaled_cst_list):  # ff_vec, ss_vec, zz_mat):
    # =========================================================================================
    # Y relative distance to constraints SHOULD BE NEGATIVE WHEN CONSTRAINT IS SATISFIED
    Y = []

    ff_vec = list(df_piv.index)
    ss_vec = list(df_piv.columns.levels[1])

    for param in scaled_cst_list:
        data = df_piv[param].values
        fy = RegularGridInterpolator((ff_vec, ss_vec),
                                     data,
                                     bounds_error=False,
                                     fill_value=None)
        Y.append(fy(xin)[0])
    # -----------------------------------------------------------------------------------------
    # criteria
    data = df_piv[scaled_crit_name].values
    fc = RegularGridInterpolator((ff_vec, ss_vec),
                                 data,
                                 bounds_error=False,
                                 fill_value=None)
    C = fc(xin)[0]

    # -----------------------------------------------------------------------------------------
    # global distance (sum of individual distances grounded at zero)
    # D is the relative composite distance to constraints
    D = sum([max(yy, 0) for yy in Y])  # if D == 0, all contraints are satisfied. if D > 0,

    # ---------------------------------------------------------------------------------------------------------------
    return [D, C, Y]


def scitwod_(X1, X2, dX1, dX2, noms, Nzoom, LwBs, LwFc, UpBs, UpFc, CrFc, fct_SciTwoD, graph, varargin=[]):
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

    if len(varargin) > 0:
        OneVar = True
    else:
        OneVar = False
    # print("OneVar : ", OneVar)

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
    zed_df = initiate_storage_df(index_list, names_list)

    ref_cell_pt_on_grid = [0, 0]  # defines the point of reference for the active cell in the grid ([0, 0] is [X1, X2])
    ref_cell_pt_on_grid_list = list()  # initiate cell reference point list storage
    ref_cell_pt_on_grid_list.append(ref_cell_pt_on_grid)

    CritRef = None  # no ref value is known at the beginning for the criteria

    N = [[0, 0],
         [0, 1],
         [1, 1],
         [1, 0]]  # define cell corner circuit (modified from scilab)

    P = [[0, 0],
         [1, 0],
         [1, 1],
         [0, 1],
         [-1, 1],
         [-1, 0],
         [-1, -1],
         [0, -1],
         [1, -1]]  # define cell exploration circuit

    # Nzoom: number of zooming cycle
    # ---------------------------------------------------------------------------------------------------------------
    for Nz in range(1, 1 + Nzoom):  # Nz goes from 1 to Nzoom

        # print("======================")
        # print("======================")
        print("Nz : ", Nz)

        # initiate search
        # ---------------------------------------------------------------------------------------------------------------
        new_cell = True

        # compute first base cell data

        zed_df, new_cell_proposed, CritRef = compute_base_cell_data(zed_df, ref_cell_pt_on_grid,
                                                                    N,
                                                                    X1, X2, dX1, dX2,
                                                                    LwBs, LwFc,
                                                                    UpBs, UpFc,
                                                                    CrFc, CritRef,
                                                                    fct_SciTwoD, OneVar)
        new_cell = new_cell or new_cell_proposed  # if new_cell_proposed is True, new_cell must be turned to True anyway

        # print(zed_df)

        # stop when no surrounding cell has a better score than central one
        # or the search is comming back on an already visited cell
        # ---------------------------------------------------------------------------------------------------------------
        while new_cell:

            # z_mat = []
            # get base cell eact data from zed_df
            # ---------------------------------------------------------------------------------------------------------------
            new_cell_grid_point_list = []
            for cc in N:  # the objective of this loop is to iterate on N rows, i.e. on the 4 cell corners
                pt_eval_grid = [int(ref_cell_pt_on_grid[0]) + int(cc[0]), int(ref_cell_pt_on_grid[1]) + int(cc[1])]
                new_cell_grid_point_list.append(tuple(pt_eval_grid))
            # print("new_cell_grid_point_list = ", new_cell_grid_point_list)

            # explore base cell surrounding
            # ---------------------------------------------------------------------------------------------------------------
            zed_df_cell = zed_df.loc[new_cell_grid_point_list]

            CritRef, trash, q_Dsum = search_in_cells_around(zed_df_cell, P, N, X1, dX1, X2, dX2, crit_name,
                                                            CritRef, scaled_cst)

            # set new base cell
            # ---------------------------------------------------------------------------------------------------------------
            shift_ = P[int(q_Dsum)]
            ref_cell_pt_on_grid = list(np.array(ref_cell_pt_on_grid) + np.array(shift_))
            # print(ref_cell_pt_on_grid)
            ref_cell_pt_on_grid_list.append(ref_cell_pt_on_grid)

            X1 = X1 + shift_[0] * dX1
            X2 = X2 + shift_[1] * dX2

            new_cell = False

            # compute base cell data
            # ---------------------------------------------------------------------------------------------------------------
            zed_df, new_cell_proposed, CritRef = compute_base_cell_data(zed_df, ref_cell_pt_on_grid,
                                                                        N,
                                                                        X1, X2, dX1, dX2,
                                                                        LwBs, LwFc,
                                                                        UpBs, UpFc,
                                                                        CrFc, CritRef,
                                                                        fct_SciTwoD, OneVar)
            new_cell = new_cell or new_cell_proposed  # if new_cell_proposed is True, new_cell must be turned to True anyway

            # print(zed_df)

        if Nz < Nzoom:
            # expension of Zed matrix
            # ---------------------------------------------------------------------------------------------------------------
            zed_df = expand_grid(zed_df)
            # print(zed_df)

            # change base cell co-ordinates and parameter shifts
            # ---------------------------------------------------------------------------------------------------------------
            ref_cell_pt_on_grid_list = expand_vector_list(ref_cell_pt_on_grid_list, factor=2)
            ref_cell_pt_on_grid = ref_cell_pt_on_grid_list[-1]

            [dX1, dX2] = [dX1 / 2, dX2 / 2]
            # print(dX1, dX2)

            new_cell = True

    # check that at least one constraint crosses the last cell
    # if not, move toward a cell with better criteria
    # ---------------------------------------------------------------------------------------------------------------
    # print("###########################################################@")
    # print("Checking that constraints cross the last cell...")
    exist_cst_crossing = False
    while exist_cst_crossing is False:

        ref_cell_pt_on_grid = ref_cell_pt_on_grid_list[-1]

        last_cell_grid_point_list = []
        for cc in N:  # the objective of this loop is to iterate on N rows, i.e. on the 4 cell corners
            pt_eval_grid = [int(ref_cell_pt_on_grid[0]) + int(cc[0]), int(ref_cell_pt_on_grid[1]) + int(cc[1])]
            last_cell_grid_point_list.append(tuple(pt_eval_grid))

        zed_df_last_cell = zed_df.loc[last_cell_grid_point_list]
        cst_cross_sum = (zed_df_last_cell[scaled_cst] > 0).sum()

        if ((cst_cross_sum != 4) & (cst_cross_sum != 0)).any():
            exist_cst_crossing = True
            # print("--> At least one constraint crosses the last cell")

        elif ~(((cst_cross_sum != 4) & (cst_cross_sum != 0)).all()):
            # print(exist_cst_crossing)
            # print("--> no constraint cross the cell. Looking for another cell...")
            # we search in the cells around for place where the criteria is reduced

            CritRef, q_Cmin, trash = search_in_cells_around(zed_df_cell, P, N, X1, dX1, X2, dX2, crit_name, CritRef,
                                                            scaled_cst)
            # trash, q_Dsum

            # set new base cell
            # ---------------------------------------------------------------------------------------------------------------
            shift_ = P[int(q_Cmin)]
            ref_cell_pt_on_grid = list(np.array(ref_cell_pt_on_grid) + np.array(shift_))
            # print(ref_cell_pt_on_grid)
            ref_cell_pt_on_grid_list.append(ref_cell_pt_on_grid)

            X1 = X1 + shift_[0] * dX1
            X2 = X2 + shift_[1] * dX2

            zed_df, new_cell_proposed, CritRef = compute_base_cell_data(zed_df, ref_cell_pt_on_grid,
                                                                        N,
                                                                        X1, X2, dX1, dX2,
                                                                        LwBs, LwFc,
                                                                        UpBs, UpFc,
                                                                        CrFc, CritRef,
                                                                        fct_SciTwoD, OneVar)
            # print("--> new cell (for crossing constraint) identified and calculated...")

            exist_cst_crossing = exist_cst_crossing or new_cell_proposed

    # sub grid tracking when optimum is close to a frontiere
    # ---------------------------------------------------------------------------------------------------------------
    search_new_cell = True
    shift_ = [0, 0]

    # print("###########################################################@")
    # print("Solving for optimum inside the last cell (with linearized local problem)...")
    while search_new_cell:

        # print("--> extracting last cell...")
        ref_cell_pt_on_grid = ref_cell_pt_on_grid_list[-1]

        new_cell_grid_point_list = []
        for cc in N:  # the objective of this loop is to iterate on N rows, i.e. on the 4 cell corners
            pt_eval_grid = [int(ref_cell_pt_on_grid[0]) + int(cc[0]), int(ref_cell_pt_on_grid[1]) + int(cc[1])]
            new_cell_grid_point_list.append(tuple(pt_eval_grid))

        zed_df_cell = zed_df.loc[new_cell_grid_point_list]
        cst_cross_sum = (zed_df_cell[scaled_cst] > 0).sum()

        # print("--> searching for candidates optima...")
        csts_list = []  # list of [Ai, Bi] coordinates where Ai and Bi are one the constraint and on the cell boundaries
        edges_list = []  # list of [Cj, Dj] representing the edges of the cell
        cst_pts_list = []  # list of points Ai, Bi
        edges_pts_list = []  # list of points Cj, Dj
        for param in cst_cross_sum.index:
            edges_list, edges_pts_list, csts, cst_pts = find_cst_points_in_cell(zed_df_cell, param)
            if csts != []:
                csts_list.append(csts)
            cst_pts_list = cst_pts_list + cst_pts

        # print("len(edges_list) = ", len(edges_list))
        # print("csts_list       = ", csts_list)
        # print("cst_pts_list    = ", cst_pts_list)
        # print("edges_list      = ", edges_list)
        # print("edges_pts_list  = ", edges_pts_list)

        cst_edges_list = csts_list + edges_list
        intersec_pt_list = []
        for i, edge_1 in enumerate(cst_edges_list):
            for j, edge_2 in enumerate(cst_edges_list):  # enumerate(cst_edges_list[:i]+cst_edges_list[i+1:]):
                if is_crossing_(edge_1, edge_2) & (j > i):
                    # print(i, j, edge_1, edge_2)
                    intersec_pt_list.append(intersection_of_(edge_1, edge_2))
                    # print(intersec_pt_list)

        # print(intersec_pt_list)

        # for cst in csts_list:
        #     plt.plot(np.array(cst).T[0], np.array(cst).T[1])
        # plt.scatter(np.array(intersec_pt_list).T[0], np.array(intersec_pt_list).T[1])
        # plt.show()

        # print("--> select the best candidate...")
        all_opt_candidates_coord = cst_pts_list + edges_pts_list + intersec_pt_list
        # print("all_opt_candidates_coord = ", all_opt_candidates_coord)
        all_opt_candidates_tags = ["cst"] * len(cst_pts_list) + \
                                  ["edg"] * len(edges_pts_list) + \
                                  ["int"] * len(intersec_pt_list)
        # print("all_opt_candidates_tags = ", all_opt_candidates_tags)

        bst_C_ind, bst_C = find_best_opt_candidate(zed_df_cell, all_opt_candidates_coord, crit_name, scaled_cst)

        # print("bst_C_ind, bst_C = ", bst_C_ind, bst_C)
        # print("all_opt_candidates_coord[bst_C_ind] = ", all_opt_candidates_coord[bst_C_ind])

        # print("--> where is the best candidate...")
        if all_opt_candidates_tags[bst_C_ind] == "int":
            # print("----> if inside the cell, stop !")
            search_new_cell = False
            sol = all_opt_candidates_coord[bst_C_ind]
        else:
            # print("----> if on the border, shift the cell and start again...")
            all_opt_candidates_coord[bst_C_ind]
            # where is it on the grid ?

            opt_x1 = all_opt_candidates_coord[bst_C_ind][0]
            opt_x2 = all_opt_candidates_coord[bst_C_ind][1]

            grid_x1_opt = (opt_x1 - X1) / dX1
            grid_x2_opt = (opt_x2 - X2) / dX2

            shift_list = []  # contains only one or two elements (no more, no less)
            epsilon = 1e-4
            if abs(grid_x1_opt - 0) < epsilon:
                shift_list.append([-1, 0])
            elif abs(grid_x1_opt - 1) < epsilon:
                shift_list.append([1, 0])
            if abs(grid_x2_opt - 0) < epsilon:
                shift_list.append([0, -1])
            elif abs(grid_x2_opt - 1) < epsilon:
                shift_list.append([0, 1])

            # print("[opt_x1, opt_x2] = ", [opt_x1, opt_x2])
            # print("[grid_x1_opt, grid_x2_opt] = ", [grid_x1_opt, grid_x2_opt])
            # print("shift_list = ", shift_list)
            # if the ref_cell_pt_on_grid is already in zed_df,
            # then this means the new cell has already been searched
            # then last opt candidate is kept as the solution

            # consider the first possible shift_ in shift_list.
            shift_ = shift_list[0]
            ref_cell_pt_on_grid = list(np.array(ref_cell_pt_on_grid) + np.array(shift_))
            # print("ref_cell_pt_on_grid = ", ref_cell_pt_on_grid)
            # print("zed_df.index = ", zed_df.index)
            if ~zed_df.index.isin([tuple(ref_cell_pt_on_grid)]).any():
                # print("------> ref_cell_pt_on_grid has not already been calculated")
                # print(ref_cell_pt_on_grid)
                ref_cell_pt_on_grid_list.append(ref_cell_pt_on_grid)

                X1 = X1 + shift_[0] * dX1
                X2 = X2 + shift_[1] * dX2

                zed_df, new_cell_proposed, CritRef = compute_base_cell_data(zed_df, ref_cell_pt_on_grid,
                                                                            N,
                                                                            X1, X2, dX1, dX2,
                                                                            LwBs, LwFc,
                                                                            UpBs, UpFc,
                                                                            CrFc, CritRef,
                                                                            fct_SciTwoD, OneVar)
                # print("--> new cell identified and calculated...")
                # print("zed_df.index = ", zed_df.index)

                # print("--> extracting last cell...")
                ref_cell_pt_on_grid = ref_cell_pt_on_grid_list[-1]

                new_cell_grid_point_list = []
                for cc in N:  # the objective of this loop is to iterate on N rows, i.e. on the 4 cell corners
                    pt_eval_grid = [int(ref_cell_pt_on_grid[0]) + int(cc[0]), int(ref_cell_pt_on_grid[1]) + int(cc[1])]
                    new_cell_grid_point_list.append(tuple(pt_eval_grid))

                zed_df_cell = zed_df.loc[new_cell_grid_point_list]
                cst_cross_sum = (zed_df_cell[scaled_cst] > 0).sum()

                if max(cst_cross_sum) == 4:
                    # this means one constraint is not satisfied by any corner of the cell
                    # then check the shift_list[1] option (if any)
                    if len(shift_list) == 1:
                        # print("------> shifting the cell to find a better optimum is not possible")
                        # search is over
                        sol = [opt_x1, opt_x2]
                        search_new_cell = False
                    elif len(shift_list) == 2:
                        # print("------> shift and check the cell in the other direction")
                        shift_ = list(- np.array(shift_list[0]) + np.array(shift_list[1]))
                        ref_cell_pt_on_grid = list(np.array(ref_cell_pt_on_grid) + np.array(shift_))

                if ~zed_df.index.isin([tuple(ref_cell_pt_on_grid)]).any():
                    # print("------> ref_cell_pt_on_grid has not already been calculated")
                    # print(ref_cell_pt_on_grid)
                    ref_cell_pt_on_grid_list.append(ref_cell_pt_on_grid)

                    X1 = X1 + shift_[0] * dX1
                    X2 = X2 + shift_[1] * dX2

                    zed_df, new_cell_proposed, CritRef = compute_base_cell_data(zed_df, ref_cell_pt_on_grid,
                                                                                N,
                                                                                X1, X2, dX1, dX2,
                                                                                LwBs, LwFc,
                                                                                UpBs, UpFc,
                                                                                CrFc, CritRef,
                                                                                fct_SciTwoD, OneVar)
                    # print("--> new cell identified and calculated...")
                    # print("zed_df.index = ", zed_df.index)

                else:
                    # print("------> ref_cell_pt_on_grid has already been calculated - search is over")
                    ref_cell_pt_on_grid = ref_cell_pt_on_grid_list[-1]
                    sol = [opt_x1, opt_x2]
                    search_new_cell = False

                # search_new_cell = search_new_cell or new_cell_proposed
            else:
                # print("------> ref_cell_pt_on_grid has already been calculated - search is over")
                ref_cell_pt_on_grid = ref_cell_pt_on_grid_list[-1]
                sol = [opt_x1, opt_x2]
                search_new_cell = False

        # search_new_cell = False

    # # draw field and constraints
    # # ---------------------------------------------------------------------------------------------------------------
    # if draw_output,
    #
    #     ISO = tlist("ISO_data");
    #     for i=1:size(Names(4:$), "*"),
    #         ISO(1)($+1) = Names(3 + i);
    #         ISO($+1) = list("up", [0]);
    #     graph = tlist(["surf_graph", "NAME", "TYP", "MERGE", "ABS", "ORD", "FIELD", "ISO", "SUP", "DATA"], ...
    #     "", "2D", "yes", Names(1), Names(2), Names(3), ISO, "", tlist(["graph_data"]));
    #     unit = tlist(["graph_unit", "ABS", "ORD", "FIELD", "ISO"], Units(1), Units(2), Units(3), Units(4:$));
    #
    #     surf_ = Z_mat(:,:, 4);
    #     iso_ = Z_mat(:,:, 5: 4 + Nc);
    #     txt = "SciTwoD Optimiser ";
    #
    #     [abs_c] = convert_to(Units(1), abs_);
    #     [ord_c] = convert_to(Units(2), ord_);
    #     [surf_c] = convert_to(Units(3), surf_);
    #
    #     for j=1:size(Units(4:$), '*'),
    #         V_mat_c(:,:, j) = convert_to(Units(3 + j), V_mat(:,:, j));
    #
    #     if GraphOptimSingle:
    #
    #         if isempty(SciTwoD_gnum):
    #             SciTwoD_gnum = get_window_id_();
    #             gnum = SciTwoD_gnum;
    #             wpos = [250, 50];
    #         else
    #             gnum = SciTwoD_gnum;
    #             f = scf(gnum);
    #             wpos = f.figure_position;
    #         [gnum, wpos] = SciTowD_graph_(graph, abs_c, ord_c, surf_c, iso_, V_mat_c, unit, txt, gnum, wpos);
    #
    #     else:
    #
    #         [gnum] = SciTowD_graph_(graph, abs_c, ord_c, surf_c, iso_, V_mat_c, unit, txt);
    #
    #     # draw grid
    #     # ---------------------------------------------------------------------------------------------------------------
    #     N1 = [0, 0; 1, 0; 1, 1; 0, 1; 0, 0]; # define cell corner circuit
    #     for i=1:Np,
    #         X = zeros(1, 5);
    #         Y = zeros(1, 5);
    #         for j=1:5,
    #             [ix, iy] = (2 + N1(j, 1) + P(i, 1), 2 + N1(j, 2) + P(i, 2));
    #             [X(j), Y(j)] = (Z_mat(ix, iy, 1), Z_mat(ix, iy, 2));
    #         [X_c] = convert_to(Units(1), X);
    #         [Y_c] = convert_to(Units(2), Y);
    #         plot2d(X_c, Y_c);
    #
    #     # draw
    #     result(s)
    #     # ---------------------------------------------------------------------------------------------------------------
    #     [XYCopt_c(1, 1)] = convert_to(Units(1), XYCopt(1, 1));
    #     [XYCopt_c(1, 2)] = convert_to(Units(2), XYCopt(1, 2));
    #
    #     plot2d(XYCopt_c(1, 1), XYCopt_c(1, 2), style=-5);
    #
    #     else,
    #
    #     gnum = [];
    # ---------------------------------------------------------------------------------------------------------------
    return zed_df, ref_cell_pt_on_grid_list, zed_df_cell, sol  # XYCopt, gnum


# def SciTowD_graph_(graph,abs_,ord_,surf_,iso_,val_,unit,txt,varargin):
#     # ===============================================================================================================
#     nbcl = 254 ; colormap0 = hotcolormap(nbcl+46) ;
#     colormap1 = [zeros(nbcl,1) colormap0(1:nbcl,1) colormap0(1:nbcl,2)] ;
#     colormap2 = [colormap1([nbcl:-1:1],:) ; colormap0(1:nbcl,:) ; [1 1 1]] ;
#
#     # ---------------------------------------------------------------------------------------------------------------
#     conside = tlist(["conside","up","no","lw"],-1,0,1) ;
#     consymb = tlist(["consymb","up","no","lw"],"<","=",">") ;
#
#       ngw = 1 ;		# number of window
#       nsp = 1 ;		# number of sub-plot
#       c = sqrt(nsp) ; cc = ceil(c) ; fc = floor(c) ; mat = [fc+ceil((nsp-(cc*fc))/cc),cc] ;
#
#       if lstsize(varargin)>0,
#         gnum = varargin(1) ;
#       else,
#         gnum = get_window_id_() ;
#       end,
#       if lstsize(varargin)>1,
#         wpos = varargin(2) ;
#       else,
#         wpos = [250,50] ;
#       end,
#       f = scf(gnum) ;
#       clf(gcf(),"reset") ;
#       f.immediate_drawing = "off" ;
#       f.figure_name = txt ;
#       f.figure_position = wpos ;
#       toolbar(gnum,"off") ;
#       f.color_map = colormap2
#       xset("fpf","%1.4g") ;
#       show_window() ;
#
#       for sp=1:nsp,
#         subplot(mat(1),mat(2),sp) ;
#         a = gca() ; a.tight_limits = "on" ;
#         a_txt = "" ;
#         if ~isempty(surf_),
#           a_txt = a_txt+"  Field:"+graph.FIELD+"("+unit.FIELD+")" ;
#           colorbar(min(surf_),max(surf_),colminmax=[1,nbcl]) ;
#           Sgrayplot(abs_,ord_,surf_(:,:,sp),colminmax=[1,nbcl]) ;
#         end,
#         iso_val = iso_ ;
#         n_iso = size(graph.ISO(1),'*')-1 ;
#         if n_iso>0,
#           sbcl = floor(nbcl/n_iso) ; leg_str = [] ;
#           for j=1:n_iso,
#             if typeof(graph.ISO(1+j))=="constant",
#               val = graph.ISO(1+j) ; stl = (nbcl+j*sbcl)*ones(val) ;
#           hside = 0 ; # hside = -1 upper bound ; hside = 1 lower bound
#         elseif typeof(graph.ISO(1+j))=="list",
#               val = graph.ISO(1+j)(2) ; stl = (nbcl+j*sbcl)*ones(val) ;
#               hside = conside(graph.ISO(1+j)(1)) ; # hside = -1 upper bound ; hside = 1 lower bound
#             end,
#             [visible] = optim_contour2d_(abs_,ord_,iso_val(:,:,j),val,stl,hside) ;
#             if visible,
#               leg_str($+1) = string(graph.ISO(1)(1+j)) ;
#             end,
#           end,
#           a.margins(2) = 0.240 ;
#           bnd = a.data_bounds ;
#
#           #Hack to get the polyline handle of each constraint with its hatchets, so that a correct legend may be drawn
#           #this might become useless in the future if a beter legend() function is available
#           currcolor = [], handles_vect = []
#           for i=3:size(a.children,1)-1
#             if (a.children(i).children(1).foreground <> currcolor) then
#                 currcolor = a.children(i).children(1).foreground
#                 handles_vect = [a.children(i).children(1), handles_vect] ;
#             end
#           end
#           if (handles_vect == []) then #no constraint displayed on graph
#                 handles_vect = gca() ;
#           else
#             if abs(bnd(2)-bnd(1))*abs(bnd(4)-bnd(3))>0,
#               h = legend(handles_vect, leg_str,[bnd(2),bnd(4)]) ;
#             else,
#               h = legend(handles_vect, leg_str) ;
#             end,
#             h.font_size = 2 #font of the legend labels
#           end
#           #end hack
#         end,
#         a.title.text = a_txt ;
#       end,
#
#       if nsp==1,
#     #    a.title.text = txt+a.title.text ;
#         a.x_label.text = graph.ABS+"("+unit.ABS+")" ;
#         a.y_label.text = graph.ORD+"("+unit.ORD+")" ;
#         #formatting
#         a.thickness = 2
#         a.font_size = 3
#         a.title.font_size = 4
#         a.x_label.font_size = 3
#         a.y_label.font_size = 3
#         a.title.foreground = 215 #green
#         a.labels_font_color = 215
#       end,
#
#     f.user_data = list(graph,mat,abs_,ord_,surf_,val_,[],unit,1,0,txt,colormap2,nbcl,iso_) ;
#
#     f.immediate_drawing = "on" ;
#
#     seteventhandler("SciTowD_ping_event") ;
#     # ---------------------------------------------------------------------------------------------------------------
#     retrun [gnum,wpos]
#
#
# def SciTowD_ping_event(win, x, y, ibut):
#     # =========================================================================================================================
#     # return
#     if "move cursor" or "close win" event
#     if (ibut == -1000 | ibut == -1), return, end,
#
#     select ibut,
#     case 3,
#     f = scf(win); # current figure
#
#     graph = f.user_data(1);
#     mat = f.user_data(2);
#     abs_ = f.user_data(3);
#     ord_ = f.user_data(4);
#     surf_ = f.user_data(5);
#     iso_ = f.user_data(6);
#     sup_ = f.user_data(7);
#     unit = f.user_data(8);
#     id = f.user_data(9);
#     in = f.user_data(10);
#     nbcl = f.user_data(13);
#
#     [xx, yy, rect] = xchange(x, y, "i2f")
#
#     txt = [graph.ABS + " = " + string(xx) + " (" + unit.ABS + ")";
#     graph.ORD + " = " + string(yy) + " (" + unit.ORD + ")"];
#
#     if ~isempty(surf_),
#     zz = linear_interpn(xx, yy, abs_, ord_, surf_(:,:, id));
#     txt = [txt;
#     graph.FIELD + " = " + msprintf("%1.5g", zz) + " (" + unit.FIELD + ")"];
#     end,
#
#     iso_val = iso_;
#
#     if graph.MERGE == "yes",
#     n_iso = size(graph.ISO(1), '*') - 1;
#     if n_iso > 0, sbcl = floor(nbcl / n_iso); end,
#     for j=1:n_iso,
#     zz = linear_interpn(xx, yy, abs_, ord_, iso_val(:,:, j));
#     txt = [txt;
#     graph.ISO(1)(1 + j) + " = " + msprintf("%1.5g", zz) + " (" + unit.ISO(j) + ")"];
#     end
#     else,
#     zz = linear_interpn(xx, yy, abs_, ord_, iso_val(:,:, in ));
#     txt = [txt;
#     graph.ISO(1)(1 + in) + " = " + msprintf("%1.5g", zz) + " (" + unit.ISO( in)+")"];
#     end,
#
#     my_messagebox_("SciTowD Graph", txt);
#     end,
#     # ---------------------------------------------------------------------------------------------------------------
#     return
#
#
# def optim_contour2d_(abs_, ord_, iso_val, val, stl, hside) :
#     # ===============================================================================================================
#     [xjc, yjc] = oriented_contour2di_(abs_, ord_, iso_val, val, hside);
#     if ~isempty(xjc),
#     # k = 1;
#     # while k <= size(xjc, "*"),
#     # n = yjc(k);
#     # for i=1:n,
#     # xstring(xjc(k + i), yjc(k + i), tick);
#     # end,
#     # k = k + n + 1;
#     # end,
#
#     if (hside <> 0),
#     dtx = max(abs_) - min(abs_);
#     dty = max(ord_) - min(ord_);
#     [xjh, yjh] = hatchings_(xjc, yjc, dtx, dty, 0.06);
#     xjc = [xjc, xjh];
#     yjc = [yjc, yjh];
#     end,
#     draw_constraint2d2(xjc, yjc, stl);
#     visible = % T;
#     else,
#     visible = % F;
#     end,
#     # ---------------------------------------------------------------------------------------------------------------
#     return [visible]
#
# def oriented_contour2di_(x, y, zv, val, hatch):
#     # ===============================================================================================================
#     # this function compute contour lines, and orient them so that the hatchings will be on the good side
#     # ----------------------------------------------------------------------------------------------------------------
#     # hack in case we want only one contour level
#     if (size(val, '*') == 1) then
#     val_ = [val, val];
#     else
#     val_ = val;
#     end
#
#     [xc, yc] = contour2di(x, y, zv, val_);
#
#     if (size(val, '*') == 1) then
#     xc = xc(1:size(xc, '*') / 2);
#     yc = yc(1:size(yc, '*') / 2);
#     end
#
#     # loop through the curves
#     k = 1;
#     while (k < size(xc, '*'))
#         n = yc(k);
#
#     # mid - point index
#     n2 = max(min(int(n / 2), n - 1), 1);
#
#     # indexes of the point in the grid nearest to the mid point of the selected segment
#     i = find(abs(x - 0.5 * (xc(k + n2) + xc(k + n2 + 1))) == min(abs(x - 0.5 * (xc(k + n2) + xc(k + n2 + 1)))))
#     j = find(abs(y - 0.5 * (yc(k + n2) + yc(k + n2 + 1))) == min(abs(y - 0.5 * (yc(k + n2) + yc(k + n2 + 1)))))
#
#     # decide which side is good
#     v0 = [xc(k + n2 + 1) - xc(k + n2), yc(k + n2 + 1) - yc(k + n2), 0];
#     v1 = [x(i(1)) - xc(k + n2), y(j(1)) - yc(k + n2), zv(i(1), j(1)) - xc(k)];
#     # full formula
#     # p = sign(hatch) * sign(v0(1) * v1(2) - v0(2) * v1(1)) * sign(
#         v1(3) * (v0(1) ^ 2 + v0(2) ^ 2) - v0(3) * (v0(1) * v1(1) + v0(2) * v1(2)))
#     # Simplified because v0(3) = 0
#     p = sign(v0(1) * v1(2) - v0(2) * v1(1)) * sign(hatch) * sign(v1(3))
#
#     # reverse the order of the points of the curve
#     if (p < 0) then
#     for i=(1:n2)
#     tmp = xc(k + i);
#     xc(k + i) = xc(k + n - i + 1);
#     xc(k + n - i + 1) = tmp;
#     tmp = yc(k + i);
#     yc(k + i) = yc(k + n - i + 1);
#     yc(k + n - i + 1) = tmp;
#     end
#     end
#     k = k + n + 1;
#     end
#     # ----------------------------------------------------------------------------------------------------------------
#     return [xc, yc, p]
#
#
# def hatchings_(x,y,dtx,dty,lh):
#     # ===============================================================================================================
#     # the following function computes the hatchings
#     # ----------------------------------------------------------------------------------------------------------------
#     # x,y: the curve, in contour2di output format
#     # dtx,dty: size of the surface
#     # lh: hatchings length, as a fraction of graph size in pixel
#     # hatchings are always on the same side of the curves
#     # ----------------------------------------------------------------------------------------------------------------
#     dl=0.8*lh;
#     k=1;
#     xh=[];
#     yh=[];
#     rot=lh*[0.5*dtx,-dty*0.8660254;dtx*0.8660254,dty*0.5]; # rotation matrix
#     while (k<size(x,'*')),  # loop through the curves
#       l=0;
#         n=y(k);
#         x0=[x(k+1),y(k+1)];   # starting point
#         for i=2:n,            # loop through the segments of the curve
#             x1=[x(k+i),y(k+i)]; # segment end point
#             xx=x1-x0;           # segment vector
#             xxx=[xx(1)/dtx, xx(2)/dty]; # scaled segment vector
#             lg=norm(xxx,2);
#             if (lg>0) then
#                 xxx=xxx/lg;        # normalized & scaled segment vector
#                 dxx=xxx*rot;       # hatchings vector
#                 while (l<=lg)      # add several hatchings along the segment
#                     xh=[xh, x(k), x0(1)+l*xxx(1)*dtx, x0(1)+l*xxx(1)*dtx+dxx(1)];
#                     yh=[yh,   2 , x0(2)+l*xxx(2)*dty, x0(2)+l*xxx(2)*dty+dxx(2)];
#                     l=l+dl;
#                 end
#                 l=l-lg;
#             end
#             x0=x1; # starting point of the next segment is the ending of the previous one
#         end
#         k=k+n+1; # next curve position
#     end
#     # ----------------------------------------------------------------------------------------------------------------
#     return [xh,yh]
#
#
# def draw_constraint2d2(xc1,yc1,st):
#     # ===============================================================================================================
#     k=1;c=1;
#     while(k<size(xc1,'*'))
#         n=yc1(k);
#         plot2d(xc1(k+(1:n)),yc1(k+(1:n)),style=st,frameflag=0);
#         c=c+1;
#         k=k+n+1;
#     end
#     # ---------------------------------------------------------------------------------------------------------------


# ================================================================================================================
#
#	2D optimisation
#
# ================================================================================================================

def optim2d_(Xini, dXini, Names, Nzoom, LwBs, LwFc, UpBs, UpFc, CrFc, fct_SciTwoD, graph):
    # ===============================================================================================================
    # 2021_0804 : Xini, LwBs, UpBs, Names, Units, LwFc, UpFc are lists
    #             Nzoom = number of zooming cycles
    n = len(Xini)  # size(Xini,'*')

    if n == 1:
        nL = len(LwBs)
        nU = len(UpBs)
        X1 = Xini[0]  # first element of Xini list
        dX1 = dXini[0]  # first element of dXini list
        X2 = 0
        dX2 = 0.1
        Names = [Names[0], "dummy", *Names[1:1 + nL], "low", *Names[1 + nL:1 + nL + nU], "up"]
        LwBs_ = [*LwBs, -1]
        LwFc_ = [*LwFc, 10]
        UpBs_ = [*UpBs, 1]
        UpFc_ = [*UpFc, 10]

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

        zed_df, ref_cell_pt_on_grid_list = scitwod_(X1, X2, dX1, dX2, Names, Nzoom, LwBs_, LwFc_, UpBs_, UpFc_,
                                                    CrFc, fct_SciTwoD, graph, "OneVar")
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

        zed_df, ref_cell_pt_on_grid_list, zed_df_cell, sol = scitwod_(X1, X2, dX1, dX2, Names, Nzoom, LwBs, LwFc,
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
    return zed_df, ref_cell_pt_on_grid_list, zed_df_cell, sol  # [Y,Cr]


def my_criteria(x1, x2):
    return (x1 + 0) ** 2 + (x2 + 5) ** 2


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
    dxini = [1, 1]  # [rd.uniform(0.1, 2), rd.uniform(0.1, 2)]
    nzoom = 3

    # set lower bounds information : gi(x) >= lwbs
    names_lwbs = ["toto_lwb1", "toto_lwb2", "toto_lwbs3"]
    units_lwbs = ["no_dim", "no_dim", "no_dim"]
    lwbs = [rd.uniform(-8, 5), rd.uniform(-4, 4), rd.uniform(-4, 4)]  # [-4, -1]
    lwfc = [0.5, 0.5, 0.5]  # [1, 1]

    # set upper bounds information : hi(x) <= upbs
    names_upbs = []  # ["toto_upb1"]
    if sum(lwbs) > 0:
        upbs = []  # [rd.uniform(sum(lwbs), max(2*sum(lwbs),5))]  # [10]
    else:
        upbs = []  # [rd.uniform(0, 10)]
    upfc = []  # [0.5]

    # set criteria information : f(x)
    names_crit = ["crit"]
    crfc = 1

    # set function information [gi(x), hj(x), f(x)
    fct_scitwod = my_fct_scitwod
    names = names_lwbs + names_upbs  # + names_crit
    graph = None

    my_zed_df, ref_cell_pts_on_grid_list, my_zed_df_cell, my_sol = optim2d_(xini,
                                                                            dxini,
                                                                            names,
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
        ref_cell_pt = [xini[0] + ref_cell_pts_on_grid[0] * dxini[0] / (2 ** (nzoom - 1)),
                       xini[1] + ref_cell_pts_on_grid[1] * dxini[1] / (2 ** (nzoom - 1))]
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
    plt.scatter(np.array(my_sol).T[0], np.array(my_sol).T[1], marker="x", c="yellow")
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
              "my_sol = " + str(my_sol))

    plt.plot(list(map(list, zip(*ref_cell_pt_list)))[0],
             list(map(list, zip(*ref_cell_pt_list)))[1])

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






