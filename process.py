#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import numpy as np
from scipy.optimize import minimize

from scipy import interpolate

import pandas

import six

import matplotlib as mpl

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from aircraft.tool import unit


def mda(aircraft):
    """Perform Multidsciplinary_Design_Analysis
    All coupling constraints are solved
    """
    # aircraft.airframe.geometry_analysis()
    aircraft.airframe.statistical_pre_design()

    # aircraft.weight_cg.mass_analysis()
    aircraft.weight_cg.mass_pre_design()

    aircraft.performance.mission.mass_mission_adaptation()

    aircraft.performance.analysis()

    aircraft.aerodynamics.aerodynamic_analysis()

    aircraft.power_system.thrust_analysis()



def eval_optim_data(x_in ,aircraft,var,cst,cst_mag,crt,crt_mag):
    """Compute criterion and constraints
    """
    for k,key in enumerate(var):    # put optimization variables in aircraft object
        exec(key+" = x_in[k]")

    mda(aircraft)                   # Run MDA

    constraint = np.zeros(len(cst))
    for k,key in enumerate(cst):    # put optimization variables in aircraft object
        exec("constraint[k] = eval(key)/eval(cst_mag[k])")
        print(constraint[k])

    criterion = eval(crt) * (20./crt_mag)

    return criterion,constraint


def eval_optim_cst(x_in,aircraft,var,cst,cst_mag,crt,crt_mag):
    """Retrieve constraints
    """
    crit,cst = eval_optim_data(x_in ,aircraft,var,cst,cst_mag,crt,crt_mag)
    print("cst :",cst)
    return cst


#===========================================================================================================
def eval_optim_crt(x_in,aircraft,var,cst,cst_mag,crt,crt_mag):
    """Retreve criteria
    """
    crit,cst = eval_optim_data(x_in,aircraft,var,cst,cst_mag,crt,crt_mag)
    print("Design :",x_in)
    print("Crit :",crit)
    return crit


#===========================================================================================================
def mdf(aircraft,var,var_bnd,cst,cst_mag,crt):
    """
    Compute criterion and constraints
    """
    from scipy.optimize import SR1, NonlinearConstraint

    start_value = np.zeros(len(var))
    for k,key in enumerate(var):    # put optimization variables in aircraft object
        exec("start_value[k] = eval(key)")

    crt_mag,unused = eval_optim_data(start_value,aircraft,var,cst,cst_mag,crt,1.)

    res = minimize(eval_optim_crt, start_value, args=(aircraft,var,cst,cst_mag,crt,crt_mag,), method="trust-constr",
                   jac="3-point", hess=SR1(), hessp=None, bounds=var_bnd, tol=1e-5,
                   constraints=NonlinearConstraint(fun=lambda x:eval_optim_cst(x,aircraft,var,cst,cst_mag,crt,crt_mag),
                                                   lb=0., ub=np.inf, jac='3-point'),
                   options={'maxiter':500,'gtol': 1e-6})

    #              tol=None, callback=None,
    #              options={'grad': None, 'xtol': 1e-08, 'gtol': 1e-08, 'barrier_tol': 1e-08,
    #                       'sparse_jacobian': None, 'maxiter': 1000, 'verbose': 0,
    #                       'finite_diff_rel_step': None, 'initial_constr_penalty': 1.0,
    #                       'initial_tr_radius': 1.0, 'initial_barrier_parameter': 0.1,
    #                       'initial_barrier_tolerance': 0.1, 'factorization_method': None, 'disp': False})

    # res = minimize(eval_optim_crt, start_value, args=(aircraft,crit_index,crit_ref,mda_type,), method="SLSQP", bounds=search_domain,
    #                constraints={"type":"ineq","fun":eval_optim_cst,"args":(aircraft,crit_index,crit_ref,mda_type,)},
    #                jac="2-point",options={"maxiter":30,"ftol":1e-14,"eps":0.01},tol=1e-14)

    #res = minimize(eval_optim_crt, x_in, args=(aircraft,crit_index,crit_ref,mda_type,), method="COBYLA", bounds=((110000,140000),(120,160)),
    #               constraints={"type":"ineq","fun":eval_optim_cst,"args":(aircraft,crit_index,crit_ref,mda_type,)},
    #               options={"maxiter":100,"tol":0.1,"catol":0.0002,'rhobeg': 1.0})
    print(res)

    return res


def explore_design_space(aircraft, res, step, file):

    slst_list = [res[0]*(1-1.5*step[0]), res[0]*(1-0.5*step[0]), res[0]*(1+0.5*step[0]), res[0]*(1+1.5*step[0])]
    area_list = [res[1]*(1-1.5*step[1]), res[1]*(1-0.5*step[1]), res[1]*(1+0.5*step[1]), res[1]*(1+1.5*step[1])]

    print(slst_list)
    print(area_list)

    # txt = np.array([["SLST","daN"],
    #                 ["Wing_area","m2"],
    #                 ["Wing_span","m"],
    #                 ["MTOW","kg"],
    #                 ["MLW","kg"],
    #                 ["OWE","kg"],
    #                 ["MWE","kg"],
    #                 ["Cruise_SFC","kg/daN/h"],
    #                 ["Cruise_LoD","no_dim"],
    #                 ["TOFL","m"],
    #                 ["App_speed","kt"],
    #                 ["OEI_path","%"],
    #                 ["Vz_MCL","ft/min"],
    #                 ["Vz_MCR","ft/min"],
    #                 ["TTC","min"],
    #                 ["Fuel_margin","m3"],
    #                 ["Block_fuel","kg"],
    #                 ["COC","$/trip"],
    #                 ["DOC","$/trip"],
    #                 ["CO2_metric","10e-3uc"]])

    txt = np.array([["SLST","daN"],
                    ["Wing_area","m2"],
                    ["Wing_span","m"],
                    ["MTOW","kg"],
                    ["MLW","kg"],
                    ["OWE","kg"],
                    ["MWE","kg"],
                    ["Cruise_LoD","no_dim"],
                    ["TOFL","m"],
                    ["App_speed","kt"],
                    ["OEI_path","%"],
                    ["Vz_MCL","ft/min"],
                    ["Vz_MCR","ft/min"],
                    ["TTC","min"],
                    ["Block_fuel","kg"]])

    for area in area_list:
        for thrust in slst_list:

            aircraft.airframe.nacelle.reference_thrust = thrust
            aircraft.airframe.wing.area = area

            print("-------------------------------------------")
            print("Doing case for : thrust = ",thrust/10.," daN    area = ",area, " m")

            # Perform MDA
            #------------------------------------------------------------------------------------------------------

            mda(aircraft)

            print("Done")

            # Store results
            #------------------------------------------------------------------------------------------------------
            res = np.array([
                            ["%8.1f"%(aircraft.airframe.nacelle.reference_thrust/10.)],
                            ["%8.1f"%aircraft.airframe.wing.area],
                            ["%8.1f"%aircraft.airframe.wing.span],
                            ["%8.1f"%aircraft.weight_cg.mtow],
                            ["%8.1f"%aircraft.weight_cg.mlw],
                            ["%8.1f"%aircraft.weight_cg.owe],
                            ["%8.1f"%aircraft.weight_cg.mwe],
                            ["%8.4f"%(aircraft.aerodynamics.cruise_lodmax)],
                            ["%8.1f"%aircraft.performance.take_off.tofl_eff],
                            ["%8.1f"%unit.kt_mps(aircraft.performance.approach.app_speed_eff)],
                            ["%8.2f"%(aircraft.performance.oei_ceiling.path_eff*100)],
                            ["%8.1f"%unit.ftpmin_mps(aircraft.performance.mcl_ceiling.vz_eff)],
                            ["%8.1f"%unit.ftpmin_mps(aircraft.performance.mcr_ceiling.vz_eff)],
                            ["%8.1f"%unit.min_s(aircraft.performance.time_to_climb.ttc_eff)],
                            ["%8.1f"%aircraft.performance.mission.cost.fuel_block]
                          ])

            txt = np.hstack([txt,res])

    #------------------------------------------------------------------------------------------------------
    np.savetxt(file,txt,delimiter=";",fmt='%10s')


def draw_design_space(file, mark, field, const, color, limit, bound):
    # Read information
    #------------------------------------------------------------------------------------------------------
    #dat = numpy.genfromtxt(file,delimiter = ";")
    data_frame = pandas.read_csv(file, delimiter = ";",skipinitialspace=True, header=None)

    # Create figure
    #------------------------------------------------------------------------------------------------------
    name = [el.strip() for el in data_frame[0]]     # Remove blanks
    uni_ = [el.strip() for el in data_frame[1]]     # Remove blanks
    data = data_frame.iloc[:,2:].values             # Extract matrix of data

    abs = list(set(data[0,:]))
    abs.sort()
    nx = len(abs)

    ord = list(set(data[1,:]))
    ord.sort()
    ny = len(ord)

    dat = {}
    for j in range(2,len(data)):
       dat[name[j]] = data[j,:]

    uni = {}
    for j in range(2,len(data)):
       uni[name[j]] = uni_[j]

    res = []
    res.append(unit.convert_to(uni_[0],mark[0]))
    res.append(unit.convert_to(uni_[1],mark[1]))

    mpl.rcParams['hatch.linewidth'] = 0.3

    fig, axs = plt.subplots(figsize=(7,7))
    gs = mpl.gridspec.GridSpec(2,1, height_ratios=[3,1])

    F = {}
    typ = 'cubic'

    axe = plt.subplot(gs[0])
    X, Y = np.meshgrid(abs, ord)
    Z = dat[field].reshape(ny,nx)
    F[field] = interpolate.interp2d(X, Y, Z, kind=typ)
    ctf = axe.contourf(X, Y, Z, cmap=mpl.cm.Greens, levels=100)
    axins = inset_axes(axe,
                       width="5%",  # width = 5% of parent_bbox width
                       height="60%",  # height : 50%
                       loc='upper left',
                       bbox_to_anchor=(1.03, 0., 1, 1),
                       bbox_transform=axe.transAxes,
                       borderpad=0,
                       )
    plt.colorbar(ctf,cax=axins)
    axe.set_title("Criterion : "+field+" ("+uni[field]+")")
    axe.set_xlabel(name[0]+" ("+uni_[0]+")")
    axe.set_ylabel(name[1]+" ("+uni_[1]+")")

    axe.plot(res[0],res[1],'ok',ms='10',mfc='none')             # Draw solution point
    marker, = axe.plot(res[0],res[1],'+k',ms='10',mfc='none')     # Draw plot marker

    bnd = [{"ub":1.e10,"lb":-1.e10}.get(s) for s in bound]

    ctr = []
    hdl = []
    for j in range(0,len(const),1):
        Z = dat[const[j]].reshape(ny,nx)
        F[const[j]] = interpolate.interp2d(X, Y, Z, kind=typ)
        ctr.append(axe.contour(X, Y, Z, levels=[limit[j]], colors=color[j]))
        levels = [limit[j],bnd[j]]
        levels.sort()
        axe.contourf(X, Y, Z, levels=levels,alpha=0.,hatches=['/'])
        h,_ = ctr[j].legend_elements()
        hdl.append(h[0])

    handle = [hl for hl in hdl]

    axe.legend(handle,const, loc = "lower left", bbox_to_anchor=(1.02, 0.))

    # Set introspection
    #------------------------------------------------------------------------------------------------------
    axe =  plt.subplot(gs[1])
    axe.axis('off')
    val1 = [["%6.0f"%12000., uni_[0]], ["%5.2f"%135.4, uni_[1]], ["%6.0f"%70000., uni[field]]]
    rowlabel=(name[0], name[1], field)

    the_table = axe.table(cellText=val1,rowLabels=rowlabel, rowLoc='right', cellLoc='left', bbox=[0.18,0.25,0.4,0.6])
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    for k,cell in six.iteritems(the_table._cells):
        cell.set_edgecolor("silver")

    cst_uni = [uni[c] for c in const]

    val2 = np.random.random(len(const))*100.
    val2 = ["%8.1f" %v for v in val2]
    val2 = list(map(list, zip(*[val2,cst_uni])))

    the_table2 = axe.table(cellText=val2,rowLabels=const, rowLoc='right', cellLoc='left', bbox=[0.85,0.,0.4,1.])
    the_table2.auto_set_font_size(False)
    the_table2.set_fontsize(10)

    for k,cell in six.iteritems(the_table2._cells):
        cell.set_edgecolor("silver")

    the_table[0,0].get_text().set_text("%6.0f" %res[0])
    the_table[1,0].get_text().set_text("%5.2f" %res[1])
    the_table[2,0].get_text().set_text("%6.0f" %F[field](res[0],res[1]))
    for j in range(len(const)):
        the_table2[j,0].get_text().set_text("%8.1f" %F[const[j]](res[0],res[1]))


    def onclick(event):
    #    global ix, iy
        try:
            ix, iy = event.xdata, event.ydata
            the_table[0,0].get_text().set_text("%6.0f" %ix)
            the_table[1,0].get_text().set_text("%5.2f" %iy)
            the_table[2,0].get_text().set_text("%6.0f" %F[field](ix,iy))
            for j in range(len(const)):
                the_table2[j,0].get_text().set_text("%8.1f" %F[const[j]](ix,iy))
            marker.set_xdata(ix)
            marker.set_ydata(iy)
            plt.draw()
        except TypeError:
            no_op = 0

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Pack and draw
    #------------------------------------------------------------------------------------------------------
    plt.tight_layout()
    plt.show()







