#!/usr/bin/env python3
"""
:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC

The main design processes are defined in this module:

* Multidisciplanary Design Analysis
* Mulitdisciplinary Design Feasible

Allow you to draw design space charts.

.. todo: improve documentation
"""

import numpy as np
from copy import deepcopy

from scipy import interpolate
from scipy.optimize import SR1, NonlinearConstraint, minimize

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from marilib.utils import unit


def eval_this(aircraft,design_variables):
    """Evaluate the current value of the design variables of the aircraft
    :param aircraft: the aircraft being designed
    :param design_variables: a list of string path to the variables. Example : ::
            design_variables = ["aircraft.airframe.nacelle.reference_thrust",
                                "aircraft.airframe.wing.area"]
    :return: the value of the designed variables
    """
    res = []
    for str in design_variables:
        res.append(eval(str))
    return res



def mda(aircraft):
    """Perform Multidsciplinary_Design_Analysis
    All coupling constraints are solved in a relevent order
    """
    # aircraft.airframe.geometry_analysis()
    aircraft.airframe.statistical_pre_design()

    # aircraft.weight_cg.mass_analysis()
    aircraft.weight_cg.mass_pre_design()

    aircraft.aerodynamics.aerodynamic_analysis()

    aircraft.handling_quality.analysis()

    aircraft.performance.mission.mass_mission_adaptation()

    aircraft.performance.mission.payload_range()

    aircraft.performance.analysis()

    aircraft.economics.operating_cost_analysis()

    aircraft.environment.fuel_efficiency_metric()

    aircraft.power_system.thrust_analysis()



def mda_hq(aircraft):
    """Perform Multidsciplinary_Design_Analysis
    All coupling constraints are solved in a relevent order
    """
    # aircraft.airframe.geometry_analysis()
    aircraft.airframe.statistical_pre_design()

    # aircraft.weight_cg.mass_analysis()
    aircraft.weight_cg.mass_pre_design()

    aircraft.aerodynamics.aerodynamic_analysis()

    aircraft.handling_quality.optimization()        # Perform optimization instead of analysis

    aircraft.performance.mission.mass_mission_adaptation()

    aircraft.performance.mission.payload_range()

    aircraft.performance.analysis()

    aircraft.economics.operating_cost_analysis()

    aircraft.environment.fuel_efficiency_metric()

    aircraft.power_system.thrust_analysis()



def eval_optim_data(x_in ,aircraft,var,cst,cst_mag,crt,crt_mag):
    """Compute criterion and constraints
    """
    for k,key in enumerate(var):    #  Put optimization variables in aircraft object
        exec(key+" = x_in[k]")

    mda(aircraft)                   #  Run MDA

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

def eval_optim_crt(x_in,aircraft,var,cst,cst_mag,crt,crt_mag):
    """Retreve criteria
    """
    crit,cst = eval_optim_data(x_in,aircraft,var,cst,cst_mag,crt,crt_mag)
    print("Design :",x_in)
    print("Crit :",crit)
    return crit


def mdf(aircraft,var,var_bnd,cst,cst_mag,crt):
    """
    Compute criterion and constraints
    """

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


def explore_design_space(ac, var, step, data, file):

    aircraft = deepcopy(ac)

    res = eval_this(aircraft,var)

    slst_list = [res[0]*(1-1.5*step[0]), res[0]*(1-0.5*step[0]), res[0]*(1+0.5*step[0]), res[0]*(1+1.5*step[0])]
    area_list = [res[1]*(1-1.5*step[1]), res[1]*(1-0.5*step[1]), res[1]*(1+0.5*step[1]), res[1]*(1+1.5*step[1])]

    print(slst_list)
    print(area_list)

    txt_list = []
    val_list = []
    for j in range(len(data)):
        txt_list.append(data[j][0:2])
        val_list.append("'"+data[j][2]+"'%("+data[j][3]+")")
    txt = np.array(txt_list)
    val = np.array(val_list)

    for area in area_list:
        for thrust in slst_list:

            exec(var[0]+" = thrust")
            exec(var[1]+" = area")
            # aircraft.airframe.nacelle.reference_thrust = thrust
            # aircraft.airframe.wing.area = area

            print("-----------------------------------------------------------------------")
            print("Doing case for : thrust = ",thrust/10.," daN    area = ",area, " m")

            mda(aircraft)   # Perform MDA
            print("Done")

            res_list = []
            for j in range(len(data)):
                res_list.append([str(eval(val[j]))])
            res_array = np.array(res_list)
            txt = np.hstack([txt,res_array])

    np.savetxt(file,txt,delimiter=";",fmt='%14s')

    return res


def draw_design_space(file, mark, field, const, color, limit, bound):
    # Read information
    #------------------------------------------------------------------------------------------------------
    dataframe = np.genfromtxt(file, dtype=str, delimiter=";")
    #data_frame = pandas.read_csv(file, delimiter = ";",skipinitialspace=True, header=None)

    # Create figure
    #------------------------------------------------------------------------------------------------------
    name = [el.strip() for el in dataframe[:,0]]     # Remove blanks
    uni_ = [el.strip() for el in dataframe[:,1]]     # Remove blanks
    data = dataframe[:,2:].astype('float64')           # Extract matrix of data

    abs = list(set(data[0,:]))
    abs.sort()
    nx = len(abs)

    ord = list(set(data[1,:]))
    ord.sort()
    ny = len(ord)

    dat = {}
    for j in range(2,len(data[0])):
        dat[name[j]] = data[j,:]

    uni = {}
    for j in range(2,len(data[0])):
        uni[name[j]] = uni_[j]

    res = []
    res.append(unit.convert_to(uni_[0], mark[0]))
    res.append(unit.convert_to(uni_[1], mark[1]))

    mpl.rcParams['hatch.linewidth'] = 0.3

    fig, axs = plt.subplots(figsize=(7,7))
    gs = mpl.gridspec.GridSpec(2,2, height_ratios=[3,1])

    F = {}
    typ = 'cubic'

    axe = plt.subplot(gs[0,:])
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

    axe.legend(hdl,const, loc = "lower left", bbox_to_anchor=(1.02, 0.))

    # Set introspection
    #------------------------------------------------------------------------------------------------------
    axe = plt.subplot(gs[1,0])
    axe.axis('off')
    val1 = [["%6.0f"%12000., uni_[0]], ["%5.2f"%135.4, uni_[1]], ["%6.0f"%70000., uni[field]]]
    rowlabel=(name[0], name[1], field)

    the_table = axe.table(cellText=val1,rowLabels=rowlabel,rowLoc='right', cellLoc='left',
                          colWidths=[0.3,0.3], bbox=[0.1,0.5,0.8,0.5],edges='closed')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    for k,cell in the_table._cells.items():
        cell.set_edgecolor("silver")

    cst_uni = [uni[c] for c in const]

    val2 = np.random.random(len(const))*100.
    val2 = ["%8.1f" %v for v in val2]
    val2 = list(map(list, zip(*[val2,cst_uni])))

    ax3 = plt.subplot(gs[1,1])
    ax3.axis("off")
    the_table2 = ax3.table(cellText=val2,rowLabels=const, rowLoc='right', cellLoc='left', bbox=[0.5,0.,1.,1.])
    the_table2.auto_set_font_size(False)
    the_table2.set_fontsize(10)

    for k,cell in the_table2._cells.items():
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







