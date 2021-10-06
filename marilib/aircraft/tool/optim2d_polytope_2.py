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



def side_sign(p1, p2, p3):

    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def is_point_in_triangle(pt, v1, v2, v3):

    d1 = side_sign(pt, v1, v2)
    d2 = side_sign(pt, v2, v3)
    d3 = side_sign(pt, v3, v1)

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    return not(has_neg & has_pos)


def compute_base_cell_data(zed_df, T, X1, X2, dX1, dX2, LwBs, LwFc, UpBs, UpFc, CrFc, crit_ref,
                           fct_SciTwoD, OneVar):  # compute base cell data
    # Update zed_df with base cell data

    # get information from existing zed_df
    zed_list_new = list(zed_df.values)
    pt_eval_grid_list_new = [list(ele) for ele in zed_df.index]
    index_list = list(zed_df.index.names)
    names_list = list(zed_df.columns)

    # new_cell = False

    for cc in T:  # the objective of this loop is to iterate on the 3 cell vertices
        pt_eval_grid = [int(cc[0]),
                        int(cc[1])]
        if ~zed_df.index.isin([tuple(pt_eval_grid)]).any():  # if the point has not been computed
            if len(dX1) == 1 and len(dX2) == 1:  # TODO : check if this is really useful
                pt_eval = [X1 + cc[0] * dX1, X2 + cc[1] * dX2]
            elif len(dX1) == 2 and len(dX2) == 2:  # set the coordinates to be calculated
                pt_eval = [X1 + cc[0] * dX1[0] + cc[1] * dX2[0],
                           X2 + cc[0] * dX1[1] + cc[1] * dX2[1]]
            else:
                raise Exception("grid available only in 1D or 2D - review dX1 or dX2 size")

            print("pt_eval = ", pt_eval)
            [D, C, Y, V] = opt2d.fct_scitwod_ex_(
                pt_eval, LwBs, LwFc, UpBs, UpFc, CrFc, fct_SciTwoD, OneVar)  # compute exact values

            if (D == 0) & (crit_ref is None):
                crit_ref = C  # init criteria reference value
            if crit_ref is not None:
                D = D + (C - crit_ref) / crit_ref  # if crit_ref is defined, add it to the point composite distance

            zed_list_new.append(pt_eval + [C, D] + Y + V)  # add the values calculated for the new point to zed_list_new
            pt_eval_grid_list_new.append(pt_eval_grid)  # add the grid coordinates of the new point to the index list

            # new_cell = True

    # create zed_df_new with the same columns as zed_df and the content on zed_df + the new calculated points
    index_array = np.array(pt_eval_grid_list_new)
    zed_df_new = pd.DataFrame(zed_list_new,
                              columns=names_list,
                              index=pd.MultiIndex.from_arrays(index_array.T, names=index_list))

    return zed_df_new, crit_ref


def find_cst_points_in_cell(cell_df, param):
    cell_indices_list = list(cell_df.index)
    cell_indices_list.append(cell_indices_list[0])
    cst_bipoints = []
    cst_points_list = []

    # list of the sides (i.e. "vertices bipoints"). First : indices and then
    ver_bipoints_indices = [[cell_indices_list[j] for j in [i, i + 1]]
                            for i, ind in enumerate(cell_df.index)]

    for side_indices in ver_bipoints_indices:
        zed_df_side = cell_df[["pt_x1", "pt_x2", param]].loc[side_indices]
        zed_df_side = zed_df_side.sort_values(param)
        ptx1 = list(zed_df_side["pt_x1"])
        ptx2 = list(zed_df_side["pt_x2"])

        if (zed_df_side[param] > 0).sum() == 1:
            # we search the point
            cst_val = list(zed_df_side[param])
            cst_ptx1 = np.interp(0, cst_val, ptx1)
            cst_ptx2 = np.interp(0, cst_val, ptx2)

            cst_bipoints.append([cst_ptx1, cst_ptx2])
            cst_points_list.append([cst_ptx1, cst_ptx2])

    return cst_bipoints, cst_points_list


def find_best_opt_candidate(zed_df_cell, candidates_list, crit_name, scaled_cst, pt_tags):
    epsilon = 1e-10

    C_list = []
    Y_list = []
    for k, candidate in enumerate(candidates_list):
        [trash, C, Y] = fct_tritwod_rs_(candidate,
                                    zed_df_cell, crit_name,
                                    scaled_cst)  # compute linear approx values
        C_list.append(C)
        Y_list.append(Y)

    candidates_df = pd.DataFrame([ele[0] + [ele[1]] + ele[2] + [ele[3]] for ele in zip(candidates_list,C_list,Y_list,pt_tags)],
                                 columns=['pt_x1', 'pt_x2'] + crit_name + scaled_cst + ['pt_tag'])

    # we keep the candidates that satisfy all the constraints
    # ---------------------------------------------------------------------------------------------------------------
    candidates_df_cst = candidates_df[(candidates_df[scaled_cst].T > epsilon).sum() == 0]

    # we sort the candidates by increasing
    # ---------------------------------------------------------------------------------------------------------------
    candidates_df_cst = candidates_df_cst.sort_values(crit_name)
    candidates_df_cst.reset_index(inplace=True)

    return candidates_df_cst  # best_C_index, best_C


def fct_tritwod_rs_(xin, df_in, scaled_crit_name, scaled_cst_list):  # ff_vec, ss_vec, zz_mat):
    # =========================================================================================
    # Y relative distance to constraints SHOULD BE NEGATIVE WHEN CONSTRAINT IS SATISFIED
    Y = []

    Xv = df_in["pt_x1"].values
    Yv = df_in["pt_x2"].values

    for param in scaled_cst_list:
        Zv = df_in[param].values
        # linear interpolation
        fz = sci_int.LinearNDInterpolator(list(zip(Xv, Yv)), Zv)

        Y.append(float(fz(xin[0], xin[1])))

    # -----------------------------------------------------------------------------------------
    # criteria
    Zv = [ele for ele in opt2d.flatten(df_in[scaled_crit_name].values)]
    fz = sci_int.LinearNDInterpolator(list(zip(Xv, Yv)), Zv)
    C = float(fz(xin[0], xin[1]))

    # -----------------------------------------------------------------------------------------
    # global distance (sum of individual distances grounded at zero)
    # D is the relative composite distance to constraints
    D = sum([max(yy, 0) for yy in Y])  # if D == 0, all constraints are satisfied. if D > 0,

    # ---------------------------------------------------------------------------------------------------------------
    return [D, C, Y]


def scitwod_(X1, X2, dX1, dX2, noms, Units, Nzoom, LwBs, LwFc, UpBs, UpFc, CrFc, fct_SciTwoD, graph, varargin=[]):

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

    zed_df_cell = pd.DataFrame()

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



    return zed_df, new_pt_on_grid_list, zed_df_cell, sol


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







