#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

import unit


#-----------------------------------------------------------------------------------------------------------------------
#
#  Analysis functions
#
#-----------------------------------------------------------------------------------------------------------------------

coloration = {"general":"gold", "commuter":"green", "business":"blue", "narrow_body":"darkorange", "wide_body":"red"}

# Set font size
plt.rc('axes',labelsize=12,titlesize=20)
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('legend',fontsize=12)

def read_db(file):
    """Read data base and convert to standard units
    WARNING: special treatment for cruise_speed and max_speed which can be Mach number
    """
    raw_data = pd.read_excel(file)     # Load data base as a Pandas data frame
    un = raw_data.iloc[0:2,0:]                          # Take unit structure only
    df = raw_data.iloc[2:,0:].reset_index(drop=True)    # Remove unit rows and reset index

    for name in df.columns:
        if un.loc[0,name] not in ["string","int"] and name not in ["cruise_speed","max_speed"]:
            df[name] = unit.convert_from(un.loc[0,name], list(df[name]))
    for name in ["cruise_speed","max_speed"]:
        for j in df.index:
            if df.loc[j,name]>1.:
                df.loc[j,name] = float(unit.convert_from(un.loc[0,name], df.loc[j,name]))
    return df,un


def lin_lst_reg(df, abs, ord, order):
    """Linear least square regression of "ord" versus "abs" with given order
    order is the list of exponents to apply
    """
    def make_mat(param,order):
        mat_list = []
        for j in order:
            mat_list.append(param**j)
        mat = np.vstack(mat_list)
        return mat.T      # Need to transpose the stacked matrix

    param = np.array(list(df[abs]))
    A = make_mat(param, order)
    B = np.array(list(df[ord]))
    (C, res, rnk, s) = np.linalg.lstsq(A, B, rcond=None)

    AC = np.dot(A,C)
    res = np.sqrt(np.sum((AC-B)**2))

    x_reg = np.array(np.linspace(0, max(df[abs]), 400))
    F = make_mat(x_reg, order)
    y_reg = np.dot(F,C)

    return {"coef":C, "res":res, "reg":[x_reg,y_reg]}


def draw_reg(df, un, abs, ord, reg, coloration, leg_loc="lower right"):
    """Draw the cloud of point in perspective of the regression given into "reg" as [abs_list, ord_list]
    Coloration of each airplane type is given into "coloration"
    """
    fig,axes = plt.subplots(1,1)
    fig.canvas.set_window_title("Regression")

    title = ord + " - " + abs
    fig.suptitle(title, fontsize=12)

    xrange=[0, unit.convert_to(un.loc[0,abs],max(df[abs])*1.05)]
    yrange=[0, unit.convert_to(un.loc[0,ord],max(df[ord])*1.05)]
    draw_colored_cloud_on_axis(axes,df,un,abs,ord,xrange=xrange,yrange=yrange,coloration=coloration)

    if len(reg[0])>0:
        plt.plot(unit.convert_to(un.loc[0,abs],reg[0]), unit.convert_to(un.loc[0,ord],reg[1]), linewidth=2, color="grey")

    plt.tight_layout()
    plt.show()

def subplots_by_varname(df,un,var_names,figwidth=12,savefig=False):
    """create set of subplots for each variable in the list var_name"""

    if len(var_names)<1:                          # Check the length of var_names
        raise ValueError("var_names is empty")

    figsize = (figwidth, figwidth*0.4*len(var_names) )
    fig,axes = plt.subplots(len(var_names),2,figsize=figsize)  # Create subplots and eventually refactor the axis list
    if len(var_names)==1:
        axes = [axes]

    first_line=True
    for (line,var) in zip(axes,var_names):                     # fill the subplots
        if first_line:                                                 # add title to first line
            first_line=False
            line[0].set_title(r'Bissectrice $x_{mod}=f(x)$')
            line[1].set_title(r'Erreur relative $\frac{x_{mod}-x}{x}$ (%)')

        var_range = [0, unit.convert_to(un.loc[0, var], max(df[var]))]

        # first cell  : bisectrice plot
        line[0].plot(var_range, var_range, '-k', lw=2)  # draw y=x line
        draw_colored_cloud_on_axis(line[0],df,un,var,'model_'+var,xrange=var_range)      # draw x_model versus x_data

        # second cell : relative error
        line[1].plot(var_range, [0, 0], '-k', lw=2)
        df['error_'+var] = (df['model_'+var]-df[var])/df[var]*100
        un['error_'+var] = 'no_dim'
        draw_colored_cloud_on_axis(line[1],df,un,var,'error_'+var,xrange=var_range,yrange=[-100,100])

    plt.tight_layout()
    if savefig:
        plt.savefig("multiplot.pdf")
    plt.show()


def draw_colored_cloud_on_axis(ax,df,un,abs,ord,xrange=None,yrange=None,coloration=coloration,leg_loc="lower right"):
    """Build a colored scatter plot according to given coloration categories.
    :param ax: a figure axis to plot on
    :param df: a dataframe
    :param un: the dict of units to use for the axis scale
    :param abs: the label of the dataframe column to plot on x-axis
    :param ord: the label of the dataframe column to plot on y-axis
    :param xrange: list of 2 value for the x-axis range. Matplotlib default if not specified.
    :param yrange: list of 2 value for the y-axis range. Matplotlib default if not specified.
    :return:
    """
    cloud = []
    for typ in coloration.keys():
        abs_list = unit.convert_to(un.loc[0, abs], list(df.loc[df['airplane_type'] == typ][abs]))
        ord_list = unit.convert_to(un.loc[0, ord], list(df.loc[df['airplane_type'] == typ][ord]))
        if len(abs_list)>0:
            subcloud = ax.scatter(abs_list, ord_list, marker="o", c=coloration[typ], s=10, label=typ)
            cloud.append(subcloud)

    ax.set_ylabel(ord + ' (' + un.loc[0, ord] + ')')
    ax.set_xlabel(abs + ' (' + un.loc[0, abs] + ')')
    if xrange is not None:
        ax.set_xlim(xrange)
    if yrange is not None:
        ax.set_ylim(yrange)
    ax.legend(handles=cloud, loc=leg_loc)
    ax.grid(True) 


def get_error(df, un, abs, ord, reg, abs_interval):

    df1 = df[abs_interval[0]<=df[abs]].reset_index(drop=True).copy()
    df1 = df1[df1[abs]<=abs_interval[1]].reset_index(drop=True)

    fct = interp1d(reg[0], reg[1], kind="cubic", fill_value='extrapolate')

    df1['relative_error'] = (fct(df1[abs]) - df1[ord]) / df1[ord]

    print("Mean relative error = ", np.mean(list(df1['relative_error'])))
    print("Variance of relative error = ", np.var(list(df1['relative_error'])))

    draw_hist(list(df1['relative_error']), "error")


def draw_hist(rer,title):
    """Draw the histogram of relative errors given into "reg" as [abs_list, ord_list]
    """
    fig,axes = plt.subplots(1,1)
    fig.canvas.set_window_title("Relative error distribution")
    fig.suptitle(title, fontsize=12)

    plt.hist(rer, bins=20, range=(-1,1))

    plt.ylabel('Count')
    plt.xlabel('Relative Error')
    plt.show()


def do_regression(df, un, abs, ord, coloration, order):
    """Perform regression and draw the corresponding graph
    """
    dict = lin_lst_reg(df, abs, ord, order)
    print("Coef = ", dict["coef"])
    print("Res = ", dict["res"])
    draw_reg(df, un, abs, ord, dict["reg"], coloration)
    return dict





#-----------------------------------------------------------------------------------------------------------------------
#
#  Analysis
#
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Read data
    #-------------------------------------------------------------------------------------------------------------------
    # path_to_data_base = "All_Data_extract.xlsx"
    path_to_data_base = "All_Data_v3.xlsx"

    df,un = read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    df = df[df['name']!='A380-800'].reset_index(drop=True)


    # perform regressions
    #-------------------------------------------------------------------------------------------------------------------
    abs = "MTOW"
    ord = "OWE"

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
    # df = df[df['MTOW']<6000].reset_index(drop=True)                     # Remove all airplane with MTOW > 6t

    # order = [1]
    order = [2, 1]
    dict_owe = do_regression(df, un, abs, ord, coloration, order)


    #----------------------------------------------------------------------------------
    abs = "MTOW"
    ord = "total_power"                           # Name of the new column

    df[ord] = df['max_power']*df['n_engine']      # Add the new column to the dataframe
    un[ord] = un['max_power']                     # Add its unit

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    order = [2, 1]
    dict = do_regression(df, un, abs, ord, coloration, order)


    #----------------------------------------------------------------------------------
    abs = "nominal_range"                           # Name of the new column
    ord = "n_pax"

    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    un1 = un.copy()

    dict = draw_reg(df1, un1, abs, ord, [[],[]], coloration)


    #----------------------------------------------------------------------------------
    abs = "nominal_range"                           # Name of the new column
    ord = "PKoM"


    df1[ord] = df1['n_pax']*df1['nominal_range']/df1['MTOW']     # Add the new column to the dataframe
    un1[ord] = "m/kg"                 # Add its unit

    # order = [1.8, 0.8]
    dict = draw_reg(df1, un1, abs, ord, [[],[]], coloration)

