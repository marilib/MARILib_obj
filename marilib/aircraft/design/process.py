#!/usr/bin/env python3
"""
:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC

The main design processes are defined in this module:

* Multidisciplanary Design Analysis
* Mulitdisciplinary Design Feasible

Allow you to draw design space charts.

.. todo: improve documentation
"""

import numpy as np

from scipy.optimize import fsolve

from copy import deepcopy

from scipy import interpolate
from scipy.optimize import SR1, NonlinearConstraint, minimize

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from marilib.utils import unit

from marilib.aircraft.tool import optim2d_polytope


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



def mda(aircraft, mass_mission_matching=True):
    """Perform Multidsciplinary_Design_Analysis
    All coupling constraints are solved in a relevent order
    """
    # aircraft.airframe.geometry_analysis()     # Without statistical empennage sizing
    aircraft.airframe.statistical_pre_design()  # With statistical empennage sizing

    # aircraft.weight_cg.mass_analysis()      # Without MZFW - MLW coupling
    aircraft.weight_cg.mass_pre_design()    # With MZFW - MLW coupling

    aircraft.aerodynamics.aerodynamic_analysis()

    aircraft.handling_quality.analysis()
    # aircraft.handling_quality.optimization()        # Perform optimization instead of analysis

    if mass_mission_matching:
        aircraft.performance.mission.mass_mission_adaptation()

    aircraft.performance.mission.payload_range()

    aircraft.performance.analysis()

    aircraft.economics.operating_cost_analysis()

    aircraft.environment.fuel_efficiency_metric()

    # aircraft.power_system.thrust_analysis()


def mda_plus(aircraft):
    """Solves coupling between MTOW and OWE
    """
    dist = aircraft.requirement.design_range
    kdist = aircraft.requirement.max_fuel_range_factor

    def fct(x):
        aircraft.airframe.tank.mfw_factor = x[0]
        if aircraft.arrangement.tank_architecture=="pods":
            aircraft.airframe.other_tank.mfw_factor = x[0]
        mda(aircraft)
        max_fuel_range = aircraft.performance.mission.max_fuel.range
        return dist*kdist - max_fuel_range

    x_ini = aircraft.airframe.tank.mfw_factor
    output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
    if (output_dict[2]!=1): raise Exception("Convergence problem")

    aircraft.airframe.tank.mfw_factor = output_dict[0][0]
    if aircraft.arrangement.tank_architecture=="pods":
        aircraft.airframe.other_tank.mfw_factor = output_dict[0][0]
    mda(aircraft)


def mda_plus_plus(aircraft):
    """Solves coupling between MTOW and OWE
    """
    kdist = aircraft.requirement.max_fuel_range_factor
    mtow = aircraft.weight_cg.mtow

    def fct(x):
        aircraft.requirement.design_range = x[0]
        aircraft.airframe.tank.mfw_factor = x[1]
        if aircraft.arrangement.tank_architecture=="pods":
            aircraft.airframe.other_tank.mfw_factor = x[1]
        mda(aircraft)
        max_fuel_range = aircraft.performance.mission.max_fuel.range
        return [x[0]*kdist-max_fuel_range, mtow-aircraft.weight_cg.mtow]

    x_ini = [aircraft.requirement.design_range, aircraft.airframe.tank.mfw_factor]
    output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
    if (output_dict[2]!=1): raise Exception("Convergence problem")

    aircraft.requirement.design_range = output_dict[0][0]
    aircraft.airframe.tank.mfw_factor = output_dict[0][1]
    if aircraft.arrangement.tank_architecture=="pods":
        aircraft.airframe.other_tank.mfw_factor = output_dict[0][0]
    mda(aircraft)


def mda_max_fuel(aircraft):
    """Perform Multidsciplinary_Design_Analysis
    All coupling constraints are solved in a relevent order
    """
    # aircraft.airframe.geometry_analysis()     # Without statistical empennage sizing
    aircraft.airframe.statistical_pre_design()  # With statistical empennage sizing

    # aircraft.weight_cg.mass_analysis()      # Without MZFW - MLW coupling
    aircraft.weight_cg.mass_pre_design()    # With MZFW - MLW coupling

    aircraft.aerodynamics.aerodynamic_analysis()

    aircraft.handling_quality.analysis()
    # aircraft.handling_quality.optimization()        # Perform optimization instead of analysis

    aircraft.performance.mission.mass_mission_adaptation_max_fuel()

    aircraft.performance.mission.payload_range()

    aircraft.performance.analysis()

    aircraft.economics.operating_cost_analysis()

    aircraft.environment.fuel_efficiency_metric()

    # aircraft.power_system.thrust_analysis()


def mda_max_fuel(aircraft):
    """Perform Multidsciplinary_Design_Analysis
    WARNING, SPECIAL PROCESS, MTOW is given here and tank factor is adjusted accordingly
    """
    aircraft.airframe.statistical_pre_design()  # With statistical empennage sizing
    aircraft.weight_cg.mass_pre_design()

    altp = aircraft.requirement.cruise_altp
    mach = aircraft.requirement.cruise_mach
    disa = aircraft.requirement.cruise_disa

    payload = aircraft.airframe.cabin.nominal_payload

    def fct(x):
        aircraft.requirement.design_range = x[0]
        aircraft.weight_cg.mtow = x[1]

        aircraft.airframe.statistical_pre_design()  # With statistical empennage sizing
        aircraft.weight_cg.mass_pre_design()

        owe = aircraft.weight_cg.owe
        fuel_total = aircraft.weight_cg.mfw

        aircraft.performance.mission.max_fuel.eval(owe,altp,mach,disa, fuel_total=fuel_total, tow=x[1])

        return [aircraft.requirement.design_range - aircraft.performance.mission.max_fuel.range,
                x[1] - (owe + payload + fuel_total)]

    x_ini = [aircraft.requirement.design_range, aircraft.weight_cg.mtow]
    output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
    if (output_dict[2]!=1): raise Exception("Convergence problem")

    aircraft.requirement.design_range = output_dict[0][0]
    aircraft.weight_cg.mtow = output_dict[0][1]

    aircraft.airframe.statistical_pre_design()  # With statistical empennage sizing
    aircraft.weight_cg.mass_pre_design()
    aircraft.aerodynamics.aerodynamic_analysis()
    aircraft.handling_quality.analysis()
    aircraft.performance.mission.payload_range()
    aircraft.performance.analysis()
    aircraft.economics.operating_cost_analysis()
    aircraft.environment.fuel_efficiency_metric()
    # aircraft.power_system.thrust_analysis()


def mda_ligeois(aircraft):
    """Perform Multidsciplinary_Design_Analysis
    WARNING, SPECIAL PROCESS, MTOW is given here and tank factor is adjusted accordingly
    """
    aircraft.airframe.statistical_pre_design()  # With statistical empennage sizing
    aircraft.weight_cg.mass_pre_design()

    altp = aircraft.requirement.cruise_altp
    mach = aircraft.requirement.cruise_mach
    disa = aircraft.requirement.cruise_disa

    payload = aircraft.airframe.cabin.nominal_payload
    mtow = aircraft.weight_cg.mtow

    def fct(x):
        aircraft.requirement.design_range = x[0]
        aircraft.airframe.tank.mfw_factor = x[1]
        if aircraft.arrangement.tank_architecture=="pods":
            aircraft.airframe.other_tank.mfw_factor = x[1]

        aircraft.weight_cg.mtow = mtow

        aircraft.airframe.statistical_pre_design()  # With statistical empennage sizing
        aircraft.weight_cg.mass_pre_design()

        owe = aircraft.weight_cg.owe
        fuel_total = aircraft.weight_cg.mfw

        aircraft.performance.mission.max_fuel.eval(owe,altp,mach,disa, fuel_total=fuel_total, tow=mtow)

        return [aircraft.requirement.design_range - aircraft.performance.mission.max_fuel.range,
                mtow - (owe + payload + fuel_total)]

    x_ini = [aircraft.requirement.design_range, aircraft.airframe.tank.mfw_factor]
    output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
    if (output_dict[2]!=1): raise Exception("Convergence problem")

    aircraft.requirement.design_range = output_dict[0][0]
    aircraft.airframe.tank.mfw_factor = output_dict[0][1]

    aircraft.airframe.statistical_pre_design()  # With statistical empennage sizing
    aircraft.weight_cg.mass_pre_design()
    aircraft.aerodynamics.aerodynamic_analysis()
    aircraft.handling_quality.analysis()
    aircraft.performance.mission.payload_range()
    aircraft.performance.analysis()
    aircraft.economics.operating_cost_analysis()
    aircraft.environment.fuel_efficiency_metric()
    # aircraft.power_system.thrust_analysis()


def mda_hq(aircraft):
    """Perform Multidsciplinary_Design_Analysis
    All coupling constraints are solved in a relevent order
    """
    # aircraft.airframe.geometry_analysis()
    aircraft.airframe.statistical_pre_design()

    # aircraft.weight_cg.mass_analysis()
    aircraft.weight_cg.mass_pre_design()

    aircraft.aerodynamics.aerodynamic_analysis()

    # aircraft.handling_quality.analysis()
    aircraft.handling_quality.optimization()        # Perform optimization instead of analysis

    aircraft.performance.mission.mass_mission_adaptation()

    aircraft.performance.mission.payload_range()

    aircraft.performance.analysis()

    aircraft.economics.operating_cost_analysis()

    aircraft.environment.fuel_efficiency_metric()

    aircraft.power_system.thrust_analysis()



class Optimizer(object):
    """A container for the optimization procedure.
    The optimizer will prevent any optimization routine to run twice the MDA at the same point.
    The computed_points dictionnary has the foolowing keys and values: :

        * key: the position of the computed point as a key "(x,y)" a
        * value : the value of the criterion and contraints list.
    """

    def __init__(self):
        self.computed_points = {}  # store the points that are already evaluated (to avoid running mda twice during optimization)
        self.check_for_doublon = True # should the optimizer check if points were already been computed before calling eval_optim_data

    def reset(self):
        """Empty the computed_points dict
        """
        self.computed_points = {}

    def eval_optim_data(self,x_in,aircraft,var,cst,cst_mag,crt,crt_mag, proc):
        """Compute criterion and constraints.
        """
        for k, key in enumerate(var):  # Put optimization variables in aircraft object
            exec(key + " = x_in[k]")

        eval(proc+"(aircraft)")  # Run MDA

        constraint = np.zeros(len(cst))
        for k, key in enumerate(cst):  # put optimization variables in aircraft object
            constraint[k] = eval(key) / eval(cst_mag[k])

        criterion = eval(crt) / crt_mag

        self.computed_points[tuple(x_in)] = [criterion, constraint]

        return criterion,constraint

    def eval_optim_data_checked(self,x_in,aircraft,var,cst,cst_mag,crt,crt_mag, proc):
        """Compute criterion and constraints and check that it was not already computed.
        """
        in_key = tuple(x_in)
        if self.check_for_doublon:

            if in_key not in self.computed_points.keys(): # check if this point has not been already evaluated
                criterion,constraint = self.eval_optim_data(x_in,aircraft,var,cst,cst_mag,crt,crt_mag, proc)

            else:
                criterion = self.computed_points[in_key][0]
                constraint = self.computed_points[in_key][1]

        else:
            criterion, constraint = self.eval_optim_data(x_in, aircraft, var, cst, cst_mag, crt, crt_mag, proc)

        print("-->Design point:", x_in)
        print("Criterion :", criterion)
        print("Constraints :", constraint)
        return criterion,constraint

    def mdf(self,aircraft,var,var_bnd,cst,cst_mag,crt, method='trust-constr', proc='mda'):
        """Run the Multidisciplinary Design Feasible procedure for a given aircraft.
         The minimization procedure finds the minimal value of a given criteria with respect to given design variables,
         and given constraints.

         Ex: minimize the MTOW with respect to reference thrust and wing area.

         :param method: {'trust-constr','custom'} default is 'trust-constr'.

                * 'trust-constr' refers to the method :meth:`mdf_scipy_trust_constraint` which uses scipy.
                * 'custom' refers to the method :meth:`custom_descent_search` with a kind of descent search algorythm.
                    Recquires less evaluations and is often faster.
        """
        self.reset()
        start_value = np.zeros(len(var))
        for k, key in enumerate(var):  # put optimization variables in aircraft object
            exec("start_value[k] = eval(key)")
        print("--------------------------------------------------------------")
        print("start_value = ", start_value)
        print("--------------------------------------------------------------")

        crt_mag, unused = self.eval_optim_data(start_value, aircraft, var, cst, cst_mag, crt, 1., proc)

        if method == 'trust-constr':
            res = self.scipy_trust_constraint(aircraft,start_value,var,var_bnd,cst,cst_mag,crt,crt_mag, proc)

        elif method == 'optim2d_poly':
            res = self.optim_2d_polytope(aircraft,start_value,var,var_bnd,cst,cst_mag,crt,crt_mag, proc)

        elif method == 'custom':
            cost_const = lambda x_in: self.eval_optim_data(x_in, aircraft, var, cst, cst_mag, crt, 1., proc)
            res = self.custom_descent_search(cost_const,start_value)

        print(res)


    def scipy_trust_constraint(self,aircraft,start_value,var,var_bnd,cst,cst_mag,crt,crt_mag, proc):
        """
        Run the trust-constraint minimization procedure :func:`scipy.optimize.minimize` to minimize a given criterion
        and satisfy given constraints for a given aircraft.
        """
        def cost(x,*args):
            return self.eval_optim_data_checked(x,*args)[0]

        def constraints(x):
            return self.eval_optim_data_checked(x,aircraft,var,cst,cst_mag,crt,crt_mag, proc)[1]

        res = minimize(cost, start_value, args=(aircraft,var,cst,cst_mag,crt,crt_mag,),
                       constraints=NonlinearConstraint(fun=constraints,
                                                       lb=0., ub=np.inf, jac='3-point'),
                       method="trust-constr", jac="3-point", bounds=var_bnd, hess=SR1(), hessp=None,
                       options={'maxiter':500,'xtol':1.e-6})
                       #options={'maxiter':500,'xtol': np.linalg.norm(start_value)*0.01,
                       #         'initial_tr_radius': np.linalg.norm(start_value)*0.05 })
        return res


    def optim_2d_polytope(self,aircraft,start_value,var,var_bnd,cst,cst_mag,crt,crt_mag, proc):

        def fct_optim2d(x_in):
            criterion, constraints = self.eval_optim_data(x_in,aircraft,var,cst,cst_mag,crt,crt_mag, proc)
            return [constraints, [], criterion]

        n = len(cst)

        xini = start_value.tolist()
        initial_scales = [0.1*(b[1]-b[0]) for b in var_bnd]
        initial_scales_matrix = np.diag(np.array(initial_scales))

        if len(var) == 2:
            poly_unit_matrix = np.array([[1, 0],[0.5, np.sqrt(3) / 2]]).T

            dxini_matrix = np.dot(initial_scales_matrix, poly_unit_matrix)
            dxini = [dxini_matrix.T[0],
                     dxini_matrix.T[1]]  # [rd.uniform(0.1, 2), rd.uniform(0.1, 2)]
        else:
            raise Exception("this optimization algorithm is for 2D problem only")

        # dxini = [0.1*(b[1]-b[0]) for b in var_bnd]
        names = ["CST_"+str(j) for j in range(n)]
        nzoom = 4
        lwbs = [0]*n
        lwfc = [1]*n
        upbs = []
        upfc = []
        crfc = 1
        graph = None

        zed_df, ref_cell_pts_on_grid_list, zed_df_cell, sol \
        = optim2d_polytope.optim2d_poly(xini,dxini, names, [], nzoom, lwbs,lwfc, upbs,upfc, crfc, fct_optim2d, graph)
                                  # (Xini, dXini, Names, Units, Nzoom, LwBs, LwFc, UpBs, UpFc, CrFc, fct_SciTwoD, graph)

        zed_df.to_html('my_zed_df.html')
        zed_df_cell.to_html('my_zed_df_cell.html')

        fct_optim2d(sol[["pt_x1", "pt_x2"]])  # to update aircraft object with solution values

        return sol


    def custom_descent_search(self,cost_fun, x0, delta=0.02, delta_end=0.005, pen=1e6):
        """ A custom minimization method limited to 2 parameters problems (x1,x2).
        This method is based on mximum descent algorythm.

            1. Evaluate cost function (with constraint penalisation) on 3 points.

        (-1,0)    (0,0)         + : computed points
           +-------+
                   |
                   |
                   +
                 (0,-1)

            2. Compute the 2D gradient of the cost function (taking into account penalized constraints).
            3. Build a linear approximation of the cost function based on the the gradient.
            4. Extrapolate the cost function on the gradient direction step by step until cost increases
            5. Reduce the step size 'delta' by a factor 2 and restart from step 1.

    The algorythm ends when the step is small enough.
    More precisely when the relative step delta (percentage of initial starting point x0) is smaller than delta_end.

        :param cost_fun: a function that returns the criterion to be minimized and the constraints value for given
                        values of the parameters. In MARILib, cost and constraints are evaluated simultaneously.
        :param x0: a list of the two initial parameter values (x1,x2).
        :param delta: the relative step for initial pattern size : 0< delta < 1.
            :Example: If delta = 0.05, the pattern size will be 5% of the magnitude of x0 values.
        :param delta_end: the relative step for for algorythm ending.
        :param pen: penalisation factor to multiply the constraint value. The constraint is negative when unsatisfied.
            :Example: This algorythm minimizes the modified cost function : criterion + pen*constraint
        """
        points = {}  # initialize the list of computed points (delta unit coordinate)

        print("x0 ", x0)
        if not isinstance(x0, type(np.array)):
            x0 = np.array(list(x0))

        def custom_cost(x):
            crt, cst = cost_fun(x)
            return crt - sum([c*pen for c in cst if c<0])

        xy = np.array([0, 0])  # set initial value in delta coordinates
        xy_ref = xy
        points[(0, 0)] = custom_cost(x0)  # put initial point in the points dict

        scale = delta * x0  # one unit displacement of xy corresponds to delta0*x0 displacement in real x
        k = 0
        while delta >= delta_end:
            print("Iter %d, delta = %f" % (k, delta))
            current_points = []
            for step in [(0, 0), (-1, 0), (0, -1)]:
                xy = xy_ref + np.array(step) / 2 ** k  # move current location by a step
                x = x0 + xy * scale
                # print(x)
                if tuple(xy) not in points.keys():  # this value is not already computed
                    crt = custom_cost(x)
                    points[tuple(xy)] = crt
                    current_points.append([xy[0], xy[1], crt])
                else:
                    current_points.append([xy[0], xy[1], points[tuple(xy)]])

            # Find the equation of the plan passing through the 3 points
            zgradient, extrapolator = self.update_interpolator_and_gradient(current_points)
            xy = xy_ref
            crt_old = points[tuple(xy)] + 1  # set initialy criter_old > criter
            crt = points[tuple(xy)]
            while crt < crt_old:  # descent search
                crt_old = crt
                xy_ref = xy
                xy = xy - zgradient / 2 ** k
                x = x0 + xy * scale
                crt = custom_cost(x)
                points[tuple(xy)] = crt
                print("\t", x)

            k += 1
            delta = delta / 2.

        res = "---- Custom descent search ----\n>Number of evaluations : %d" %len(points)
        return res

    def update_interpolator_and_gradient(self,threePoints):
        """
        Compute the plane equation through three points.
        Returns the gradient vector and interpolator function.
        :param xxx : a three point matrix [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]
        """
        a, b, c = np.linalg.solve(threePoints,
                                  [1, 1, 1])  # find the coefficient of the plan passing through the 3 points
        extrapolator = lambda x, y: (1. - a * x - b * y) / c
        zgradient = np.array([-a / c, -b / c]) / np.linalg.norm([-a / c, -b / c])
        return zgradient, extrapolator


    def grid_minimum_search(cost_fun,x0,delta=0.02,pen=5e5):
        """ A custom minimization method limited to 2 parameters problems (x1,x2).
        This method uses a custom pattern search algorithm that requires a minimum number of call to the cost function.
        It takes advantage of the prior knowledge of the problem:

            1. Evaluate cost function (with constraint penalisation) at current point : coordinate (0,0)
            2. Evaluate, if not already done, the cost function in three other points to draw a square area of side1 :
               coordinates (-1,0), (0,-1) relatively to current point
            3. Build a linear approximation of the cost function based on the the 3 previous points.
            4. Extrapolate the cost function on the surrounding points:

            o------o-------o------o
            |                     |     o : extrapolated points
            |   (-1,0)   (0,0)    |     + : computed points
            o      +-------+      o
            |      |       |      |
            |      |       |      |
            o      o-------+      o
            |            (0,-1)   |
            |                     |
            o------o-------o------o

            5. Find the square cell (among the 9 cells defined by computed and extrapolated points)
            with minimal average criterion value
            Crt  = cost_function(x1,y1) - sum(unsatisfied constraints)
                * If current cell is minimum : STOP (or refine of asked so)
                * Else : move current point to the top right corner of the new candidate cell and go back to step 1.

        :param cost_fun: a function that returns the criterion to be minimized and the constraints value for given
                        value of the parameters : length-2 tuple (x1,x2).
        :param x0: a list of the two initial parameter values (x1,x2).
        :param args: list of additional argument for cost_fun.
        :param delta: the relative step for initial pattern size : 0< delta < 1.
            :Example: If delta = 0.05, the pattern size will be 5% of the magnitude of x0 values.
        :param pen: penalisation factor to multiply the constraint value. The constraint is negative when unsatisfied.
            :Example: This algorythm minimizes the modified cost function : criterion + pen*constraint
        """

        # TODO
        raise NotImplementedError
        return




# -------------------- PLOT OPTIM RESULTS

def explore_design_space(ac, var, step, data, file, proc="mda"):

    aircraft = deepcopy(ac)

    res = eval_this(aircraft,var)

    slst_list = [res[0]*(1-1.5*step[0]), res[0]*(1-0.5*step[0]), res[0], res[0]*(1+0.5*step[0]), res[0]*(1+1.5*step[0])]
    area_list = [res[1]*(1-1.5*step[1]), res[1]*(1-0.5*step[1]), res[1], res[1]*(1+0.5*step[1]), res[1]*(1+1.5*step[1])]

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
        x = np.array(x)  # /10 # /10 to rescale units ... WARNING not very robust !!
        x = unit.convert_to(uni_[0], x)
        y = np.array(y)
        y = unit.convert_to(uni_[1], y)
        axe.scatter(x,y)
        axe.quiver(x[:-1],y[:-1] , x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy',angles='xy',scale=1,color='k')

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







