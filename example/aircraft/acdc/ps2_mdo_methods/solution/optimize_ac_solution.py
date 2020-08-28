# -*- coding: utf-8 -*-
# Author : Francois Gallard
from example.aircraft.acdc.ps2_mdo_methods.models import *
from marilib.utils import unit
from scipy.optimize import minimize
from matplotlib import pyplot as plt

"""

MDO optimization of a 150 passengers aircraft that flies at Mach 0.78
---------------------------------------------------------------------

The optimization problem to be solved :
Minimise : mtow
wrt : area, ar, slst    100 < area < 200, 5 < ar < 20, 100e5 < slst < 200e5
st : tofl < 2000 m
     vapp < 135 kt
     300 ft/min < vz

The disiplines are provided as python functions in the models.py file

The mtow and owe variables create a strong coupling through
the mass and mission, and mass_coupling
functions

The N2 diagram is provided in the n2.pdf file.


Use the MDF formulation to solve this MDO problem.

1 - Analyse the n2 diagram (n2.pdf file) and deduce the strong couplings
2- Propose a calculation sequence, ie a MDO process based on the MDF formulation
3- Implement the MDA (coupling computation)
4- Implement the MDO process based on the MDA :
    a/ generate an objective function that valls the main process function
    b/ generate a constraints function that valls the main process function
    c/ call the optimization algorithm

5/ analyse the optimization result, discuss objective function and constraints values

"""


def run_sequentially(area=155, ar=9, bpr=9, slst=130000.,
                     ac_range=5556000., mtow=77000.):
    """
    Runs the analysis of the aircraft
    No strong coupling is taken into account : the mtow taken
    as input is not equal to the output MTOW

    """

    lod, cl_max_to, cl_max_ld = aerodynamic(area, ar)

    sfc, fn_ton, fn_mcl = engine(bpr, slst)

    fuel = mission(mtow, ac_range, sfc, lod)

    payload, owe, mzfw, mlw = mass(mtow, area, ar, slst)

    mtow2 = mtow_coupling(owe, payload, fuel)

    tofl = take_off(mtow, fn_ton, area, cl_max_to)

    vapp = approach(mlw, area, cl_max_ld)

    vz = climb(mtow, fn_mcl, lod)

    print("fuel", fuel)
    print("MTOW=", mtow2)
    print("TOFL=", tofl)
    print("vapp=", unit.kt_mps(vapp))
    print("vz=", unit.ftpmin_mps(vz))
    print("lod", lod)
    print("sfc", sfc)


########################
# MDA and main process #
########################


def mda(area=155, ar=9, slst=130000., mtow=77000.,
        ac_range=5556000., sfc=1.5e-05, lod=17.85, mtox_tol=1.e-12):
    """
    Solves the strong coupling between "mass" and "mtow_coupling" functions

    :param area: wing area
    :param slst:
    """

    mtow_diff = 1e100
    while(abs(mtow_diff) > mtox_tol):
        payload, owe, mzfw, mlw = mass(mtow, area, ar, slst)
        fuel = mission(mtow, ac_range, sfc, lod)
        mtow2 = mtow_coupling(owe, payload, fuel)

        mtow_diff = mtow2 - mtow
        mtow = mtow2

    return mtow, payload, owe, mzfw, mlw


def main_process(area=155, ar=9, slst=130000.,
                 ac_range=5556000., mtow=77000., bpr=9.):
    """
    For a given area, ar, slst, and bpr, computes the aircraft simulation

    Generates the calculation sequence : calls the disciplines sequentially,
    and the MDA when needed
    """

    lod, cl_max_to, cl_max_ld = aerodynamic(area, ar)

    sfc, fn_ton, fn_mcl = engine(bpr, slst)

    mtow, payload, owe, mzfw, mlw = mda(area, ar, slst,
                                        mtow, ac_range, sfc, lod)

    tofl = take_off(mtow, fn_ton, area, cl_max_to)

    vapp = approach(mlw, area, cl_max_ld)

    vz = climb(mtow, fn_mcl, lod)

    return tofl, vapp, vz, mtow

#####################
# Optimization part #
#####################

x_constr_history = []
x_obj_history = []
constr_history = []
obj_history = []


def objective_func(x):
    """
    Objective function : compute the mtow
    :param x: the [area, ar, slst] vector
    :returns: the objective value
    """
    area, ar, slst = x
    tofl, vapp, vz, mtow = main_process(area, ar, slst)
    x_obj_history.append(x)
    obj_history.append(mtow)
    return mtow


def constraints_func(x):
    """
    Constraint function : compute the following constraints :
    st : tofl < 2000 m
     vapp < 135 kt
     300 ft/min < vz
    :param x: the [area, ar, slst] vector
    :returns: the vector of constraints values
    """
    area, ar, slst = x
    tofl, vapp, vz, mtow = main_process(area, ar, slst)
    tofl_constr = 2000 - tofl
    vapp_constr = unit.mps_kt(135) - vapp
    vz_constr = vz - unit.mps_ftpmin(300)
    constraints_vals = tofl_constr, vapp_constr, vz_constr
    x_constr_history.append(x)
    constr_history.append(constraints_vals)
    return constraints_vals


def optimize(x0, bounds, objective, constraints):
    """
    Interface to the optimization algorithm : solve
    min objective(x)
    with respect to x
    such that constraints(x)>=0

    parameters:
    :param x0: initial guess, starting point of the optimization
    :param bounds: list of (lower, upper) bounds for the variables
    :param objective: a python function that computes the objective function to be minimized
    :param constraints: a python function that computes the constraints values
    """
    constraints_dict = {"type": "ineq", "fun": constraints}
    res = minimize(objective, x0=x0, method="SLSQP",
                   bounds=bounds, constraints=constraints_dict, tol=1e-3)
    print("Optimization result ")
    print(res)
    plot_opt_history()


def plot_opt_history():
    """
    Plots the optimization history : objective, constraints
    """
    plt.figure(1)
    plt.subplot(411)
    plt.plot(range(len(obj_history)), obj_history,
             label="obj_history")
    plt.legend()
    plt.subplot(412)
    plt.plot(range(len(constr_history)), [ci[0] for ci in constr_history],
             label="tofl constr")
    plt.legend()
    plt.subplot(413)
    plt.plot(range(len(constr_history)), [ci[1] for ci in constr_history],
             label="vapp constr")
    plt.legend()
    plt.subplot(414)
    plt.plot(range(len(constr_history)), [ci[2] for ci in constr_history],
             label="vz constr")
    plt.legend()

    plt.show()

#################
# Main function #
#################

if __name__ == "__main__":
    run_sequentially()
    mda()
    main_process()

    x0 = [180., 10., 150000.]
    bounds = [(100., 200.), (5., 20.), (80000., 200000.)]
    objective_func(x0)
    constraints_func(x0)
    optimize(x0, bounds, objective_func, constraints_func)
