#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: DRUOT Thierry
"""

import numpy as np
from copy import deepcopy

from scipy import interpolate

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import unit

import process

#-----------------------------------------------------------------------------------------------------------------------
# Atmosphere
#-----------------------------------------------------------------------------------------------------------------------

def atmosphere(altp, disa=0., full_output=False):
    """
    Ambiant data from pressure altitude from ground to 50 km according to Standard Atmosphere
    """
    g = 9.80665
    r = 287.053
    gam = 1.4

    Z = np.array([0., 11000., 20000.,32000., 47000., 50000.])
    dtodz = np.array([-0.0065, 0., 0.0010, 0.0028, 0.])
    P = np.array([101325., 0., 0., 0., 0., 0.])
    T = np.array([288.15, 0., 0., 0., 0., 0.])

    if (Z[-1]<altp):
        raise Exception("atmosphere, altitude cannot exceed 50km")

    j = 0
    while (Z[1+j]<=altp):
        T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
        if (0.<np.abs(dtodz[j])):
            P[j+1] = P[j]*(1. + (dtodz[j]/T[j])*(Z[j+1]-Z[j]))**(-g/(r*dtodz[j]))
        else:
            P[j+1] = P[j]*np.exp(-(g/r)*((Z[j+1]-Z[j])/T[j]))
        j = j + 1

    if (0.<np.abs(dtodz[j])):
        pamb = P[j]*(1 + (dtodz[j]/T[j])*(altp-Z[j]))**(-g/(r*dtodz[j]))
    else:
        pamb = P[j]*np.exp(-(g/r)*((altp-Z[j])/T[j]))
    tstd = T[j] + dtodz[j]*(altp-Z[j])
    tamb = tstd + disa
    if full_output:
        return pamb,tamb,tstd,dtodz[j]
    else:
        return pamb,tamb

def sound_speed(tamb):
    """Sound speed for ideal gas
    """
    r = 287.053
    gam = 1.4
    vsnd = np.sqrt( gam * r * tamb )
    return vsnd

def air_density(pamb,tamb):
    """Ideal gas density
    """
    r = 287.053
    rho0 = 1.225
    rho = pamb / ( r * tamb )
    sig = rho / rho0
    return rho, sig

def gas_viscosity(tamb, gas="air"):
    mu0,T0,S = [1.715e-5, 273.15, 110.4]
    mu = (mu0*((T0+S)/(tamb+S))*(tamb/T0)**1.5)
    return mu

def reynolds_number(pamb,tamb,mach):
    """Reynolds number based on Sutherland viscosity model
    """
    vsnd = sound_speed(tamb)
    rho,sig = air_density(pamb,tamb)
    mu = gas_viscosity(tamb)
    re = rho*vsnd*mach/mu
    return re

#-----------------------------------------------------------------------------------------------------------------------
# Maths
#-----------------------------------------------------------------------------------------------------------------------

def lin_interp_1d(x,X,Y):
    """linear interpolation without any control

    :param x: current position
    :param X: array of the abscissa of the known points
    :param Y: array of the known values at given abscissa
    :return: y the interpolated value of Y at x

    """
    n = np.size(X)
    for j in range(1,n):
        if x<X[j] :
            y = Y[j-1]+(Y[j]-Y[j-1])*(x-X[j-1])/(X[j]-X[j-1])
            return y
    y = Y[n-2]+(Y[n-1]-Y[n-2])*(x-X[n-2])/(X[n-1]-X[n-2])
    return y


def vander3(X):
    """Return the vandermonde matrix of a dim 3 array A = [X^2, X, 1]
    """
    V = np.array([[X[0]**2, X[0], 1.],
                  [X[1]**2, X[1], 1.],
                  [X[2]**2, X[2], 1.]])
    return V


def trinome(A,Y):
    """calculate trinome coefficients from 3 given points
    A = [X2, X, 1] (Vandermonde matrix)
    """
    X = np.array([A[0][1], A[1][1], A[2][1]])
    X2 = np.array([A[0][0], A[1][0], A[2][0]])

    det = X2[0]*(X[1]-X[2])-X2[1]*(X[0]-X[2])+X2[2]*(X[0]-X[1])

    adet = Y[0]*(X[1]-X[2])-Y[1]*(X[0]-X[2])+Y[2]*(X[0]-X[1])

    bdet = X2[0]*(Y[1]-Y[2])-X2[1]*(Y[0]-Y[2])+X2[2]*(Y[0]-Y[1])

    cdet =  X2[0]*(X[1]*Y[2]-X[2]*Y[1])-X2[1]*(X[0]*Y[2]-X[2]*Y[0]) \
          + X2[2]*(X[0]*Y[1]-X[1]*Y[0])

    if det!=0:
        C = np.array([adet/det, bdet/det, cdet/det])
    elif X[0]!=X[2]:
        C = np.array([0., Y[0]-Y[2], Y[2]*X[0]-Y[0]*X[2]/(X[0]-X[2])])
    else:
        C = np.array([0., 0., (Y[0]+Y[1]+Y[2])/3.])

    return C


def maximize_1d(xini,dx,*fct):
    """Optimize 1 single variable, no constraint.

    :param xini: initial value of the variable.
    :param dx: fixed search step.
    :param fct: function with the signature : ['function_name',a1,a2,a3,...,an] and function_name(x,a1,a2,a3,...,an).

    """
    n = len(fct[0])

    X0 = xini
    Y0 = fct[0][0](X0,*fct[0][1:n])

    X1 = X0+dx
    Y1 = fct[0][0](X1,*fct[0][1:n])

    if Y0>Y1:
        dx = -dx
        X0,X1 = X1,X0

    X2 = X1+dx
    Y2 = fct[0][0](X2,*fct[0][1:n])

    while Y1<Y2:
        X0 = X1
        X1 = X2
        X2 = X2+dx
        Y0 = Y1
        Y1 = Y2
        Y2 = fct[0][0](X2,*fct[0][1:n])

    X = np.array([X0,X1,X2])
    Y = np.array([Y0,Y1,Y2])

    A = vander3(X)     # [X**2, X, numpy.ones(3)]
    C = trinome(A,Y)

    xres = -C[1]/(2.*C[0])
    yres = fct[0][0](xres,*fct[0][1:n])

    rc = 1

    return (xres,yres,rc)



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


def explore_design_space(ac, var, step, data, file, proc="mda"):

    aircraft = deepcopy(ac)

    res = eval_this(aircraft,var)

    slst_list = [res[0]*(1-1.5*step[0]), res[0]*(1-0.5*step[0]), res[0]*(1+0.5*step[0]), res[0]*(1+1.5*step[0])]
    area_list = [res[1]*(1-1.5*step[1]), res[1]*(1-0.5*step[1]), res[1]*(1+0.5*step[1]), res[1]*(1+1.5*step[1])]

    #print(slst_list)
    #print(area_list)

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
            try:
                eval(proc+"(aircraft)")   # Perform MDA
            except Exception:
                print("WARNING: unable to perform MDA at : thrust = ",thrust/10.," daN    area = ",area, " m")
            #print("Done")

            res_list = []
            for j in range(len(data)):
                res_list.append([str(eval(val[j]))])
            res_array = np.array(res_list)
            txt = np.hstack([txt,res_array])

    np.savetxt(file,txt,delimiter=";",fmt='%15s')

    return res


def draw_design_space(file, mark, other, field, const, color, limit, bound, optim_points=None):
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
    for j in range(2,len(data[:,0])):
        dat[name[j]] = data[j,:]

    uni = {}
    for j in range(2,len(data[:,0])):
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
    ctf = axe.contourf(X, Y, Z, cmap=mpl.cm.Greens, levels=10)
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

    # Build interpolator for other data
    for j in range(0,len(other),1):
        Z = dat[other[j]].reshape(ny,nx)
        F[other[j]] = interpolate.interp2d(X, Y, Z, kind=typ)

    bnd = [{"ub":1.e10,"lb":-1.e10}.get(s) for s in bound]

    ctr = []
    hdl = []
    for j in range(0,len(const),1):
        Z = dat[const[j]].reshape(ny,nx)
        F[const[j]] = interpolate.interp2d(X, Y, Z, kind=typ)
        ctr.append(axe.contour(X, Y, Z, levels=[limit[j]], colors=color[j]))
        levels = [limit[j],bnd[j]]
        levels.sort()
        axe.contourf(X, Y, Z, levels=levels, alpha=0.,hatches=['/'])
        h,_ = ctr[j].legend_elements()
        hdl.append(h[0])

    axe.legend(hdl,const, loc = "lower left", bbox_to_anchor=(1.02, 0.))

    # Add optim points if specified -------------------------------------- MICOLAS
    if optim_points is not None:
        x,y = zip(*optim_points)
        axe.scatter(np.array(x)/10,y)  # /10 to rescale units ... WARNING not very robust !!

    # Set introspection
    #------------------------------------------------------------------------------------------------------
    axe = plt.subplot(gs[1,0])
    axe.axis('off')
    val1 = [["%6.0f"%12000., uni_[0]], ["%5.2f"%135.4, uni_[1]], ["%6.0f"%70000., uni[field]]]
    rowlabel = [name[0], name[1], field]
    for j in range(len(other)):
        val1.append(["%6.0f"%70000., uni[other[j]]])
        rowlabel.append(other[j])

    the_table = axe.table(cellText=val1,rowLabels=rowlabel,rowLoc='right', cellLoc='left',
                          colWidths=[0.3,0.3], bbox=[0.1,0.,1.,1.],edges='closed')
                          # colWidths=[0.3,0.3], bbox=[0.1,0.5,0.8,0.5],edges='closed')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    for k,cell in the_table._cells.items():
        cell.set_edgecolor("silver")

    cst_uni = [uni[c] for c in const]

    val2 = np.random.random(len(const))*100.
    val2 = ["%8.1f" %v for v in val2]
    cst_val = ["%8.1f" %v for v in limit]
    val2 = list(map(list, zip(*[val2,cst_val,cst_uni])))

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
    for j in range(len(other)):
        the_table[3+j,0].get_text().set_text("%6.0f" %F[other[j]](res[0],res[1]))
    for j in range(len(const)):
        the_table2[j,0].get_text().set_text("%8.1f" %F[const[j]](res[0],res[1]))


    def onclick(event):
    #    global ix, iy
        try:
            ix, iy = event.xdata, event.ydata
            the_table[0,0].get_text().set_text("%6.0f" %ix)
            the_table[1,0].get_text().set_text("%5.2f" %iy)
            the_table[2,0].get_text().set_text("%6.0f" %F[field](ix,iy))
            for j in range(len(other)):
                the_table[3+j,0].get_text().set_text("%6.0f" %F[other[j]](res[0],res[1]))
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


