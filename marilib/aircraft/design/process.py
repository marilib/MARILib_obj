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

    if mass_mission_matching:
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

    def mdf(self,aircraft,var,var_bnd,cst,cst_mag,crt,method='trust-constr'):
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

        crt_mag, unused = self.eval_optim_data(start_value, aircraft, var, cst, cst_mag, crt, 1.)

        if method == 'trust-constr':
            res = self.scipy_trust_constraint(aircraft,start_value,var,var_bnd,cst,cst_mag,crt,crt_mag)

        elif method == 'custom':
            cost_const = lambda x_in: self.eval_optim_data(x_in, aircraft, var, cst, cst_mag, crt, 1.)
            res = self.custom_descent_search(cost_const,start_value)

        else:
            raise ValueError("Invalid method : should be 'trust-constr' or 'custom' but found '%s'" %str(method))

        print(res)


    def scipy_trust_constraint(self,aircraft,start_value,var,var_bnd,cst,cst_mag,crt,crt_mag):
        """
        Run the trust-constraint minimization procedure :func:`scipy.optimize.minimize` to minimize a given criterion
        and satisfy given constraints for a given aircraft.
        """

        res = minimize(lambda x,*args:self.eval_optim_data_checked(x,*args)[0],
                       start_value, args=(aircraft,var,cst,cst_mag,crt,crt_mag,), method="trust-constr",
                       jac="2-point", bounds=var_bnd,
                       constraints=NonlinearConstraint(fun=lambda x:self.eval_optim_data_checked(x,aircraft,var,cst,cst_mag,crt,crt_mag)[1],
                                                       lb=0., ub=np.inf, jac='2-point'),
                       options={'maxiter':500,'xtol': np.linalg.norm(start_value)*0.01,
                                'initial_tr_radius': np.linalg.norm(start_value)*0.05 })
        return res

    def custom_descent_search(self, cost_const, x0, relative_step_init=2e-2, relative_step_end=5e-3, c_tol=1e-2,
                              relative_finite_diff_step = 1e-3):
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

        :param cost_const: a function that returns the criterion to be minimized and the constraints value for given
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

        # intialize current position and stepsize
        crit,cst = cost_const(x0)
        current_point = {"xy":x0, "criterion":crit, "constraints":cst}
        relative_step = relative_step_init
        dxdy = relative_finite_diff_step*x0
        k = 0
        while relative_step >= relative_step_end: # the minimzation ends when the relative step reaches the desired value

            if any([c<0 for c in current_point["constraints"]]) : # a constraint is active -> find gradient of constraint and go to constraint=0
                print("Active constraint")
                grad, const_grad = self._gradient(current_point,cost_const,dxdy,const_grad=True)
                # Newton Raphson iterate : find dx so that f(x0) + grad(f)*dx = 0, with f(x) the sum of active constraints at x.
                # we make the assumption that constraints are linear:
                # TODO : find constraint = 0
                dx = -sum(c for c in current_point["constraints"] if c<0)/const_grad
                xy = current_point["xy"] + dx
                current_point["xy"] = xy
                current_point["criterion"], current_point["constraints"] = cost_const(xy)


            # update step size
            relative_step = relative_step/2**k
            step_size = relative_step * x0
            print("Iter %d, relative_step = %.2g" % (k, relative_step))
            crit_gradient, const_grad= self._gradient( cost_const, current_point, relative_finite_diff_step*x0)

            xy = current_point["xy"]
            crit = current_point["criterion"] -1  # set current_criter > criter to start loop
            k=0
            while crit<current_point["criterion"] :  # descent search
                if k == 0:
                    pass
                else: # update current point
                    current_point["xy"] = xy # update current position
                    current_point["criterion"] = crit
                    current_point["constraints"] = const

                xy = xy - crit_gradient*step_size
                crit,const = cost_const(xy)
                print("\tpoint", xy)
            k += 1

        res = "---- Custom descent search ----\n>Number of evaluations : %d" %len(points)
        return res

    def _gradient(self, cost_const, current_point, finite_diff_step, constraints_grad=False):
        """
        Compute the normalized gradient of a function f(x,y) using order 1 finite difference scheme at point xy_ref.
        If constraint_grad, is true, also compute the gradient of the constraint(s) returned by f.
        :param xy_ref: the point of gradient evaluation, 2 element array-like.
        :param finite_diff_step: [dx,dy], the finite diff step (array like)
        :return: the list of points:
        [[x    , y    , f(x,y)    ], -> current point
         [x+dx , y    , f(x+dx,y) ],
         [x    , y+dy , f(x,y+dy) ]]
        """
        grad_points = []  # store position, criterion and constraints for the two grad points
        for step in [(-1, 0), (0, -1)]:
            xy = current_point["xy"] + np.array(step) * finite_diff_step  # move current location by a step
            crit, cst = cost_const(xy)
            grad_points.append({"xy": xy, "criterion": crit, "constraints": cst})

        print(grad_points)
        df_dx = (current_point["criterion"] - grad_points[0]["criterion"])
        df_dy = (current_point["criterion"] - grad_points[1]["criterion"])
        crit_norm_gradient = np.array([df_dx, df_dy]) / np.sqrt(df_dx ** 2 + df_dy ** 2)
        if constraints_grad:
            df_dx = (current_point["constraint"] - grad_points[0]["constraints"])
            df_dy = (current_point["constraints"] - grad_points[1]["constraints"])
            const_norm_gradient = np.array([df_dx, df_dy]) / np.sqrt(df_dx ** 2 + df_dy ** 2)
        else:
            const_norm_gradient = []

        return crit_norm_gradient, const_norm_gradient

    def eval_optim_data(self,x_in,aircraft,var,cst,cst_mag,crt,crt_mag):
        """Compute criterion and constraints.
        """
        for k, key in enumerate(var):  # Put optimization variables in aircraft object
            exec(key + " = x_in[k]")

        mda(aircraft)  # Run MDA

        constraint = np.zeros(len(cst))
        for k, key in enumerate(cst):  # put optimization variables in aircraft object
            constraint[k] = eval(key) / eval(cst_mag[k])

        criterion = eval(crt) / crt_mag

        self.computed_points[tuple(x_in)] = [criterion, constraint]

        return criterion,constraint


    def eval_optim_data_checked(self,x_in,aircraft,var,cst,cst_mag,crt,crt_mag):
        """Compute criterion and constraints and check that it was not already computed.
        """
        in_key = tuple(x_in)
        if self.check_for_doublon:

            if in_key not in self.computed_points.keys(): # check if this point has not been already evaluated
                criterion,constraint = self.eval_optim_data(x_in,aircraft,var,cst,cst_mag,crt,crt_mag)

            else:
                criterion = self.computed_points[in_key][0]
                constraint = self.computed_points[in_key][1]

        else:
            criterion, constraint = self.eval_optim_data(x_in, aircraft, var, cst, cst_mag, crt, crt_mag)

        print("-->Design point:", x_in)
        print("Criterion :", criterion)
        print("Constraints :", constraint)
        return criterion,constraint


    def grid_minimum_search(self,cost_fun,x0,delta=0.02,pen=5e5):
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

def explore_design_space(ac, var, step, data, file):

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
                mda(aircraft)   # Perform MDA
            except Exception:
                print("WARNING: unable to perform MDA at : thrust = ",thrust/10.," daN    area = ",area, " m")
            #print("Done")

            res_list = []
            for j in range(len(data)):
                res_list.append([str(eval(val[j]))])
            res_array = np.array(res_list)
            txt = np.hstack([txt,res_array])

    np.savetxt(file,txt,delimiter=";",fmt='%14s')

    return res


def draw_design_space(file, mark, field, const, color, limit, bound, optim_points=None):
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

    # Add optim points if specified -------------------------------------- MICOLAS
    if optim_points is not None:
        x,y = zip(*optim_points)
        axe.scatter(np.array(x)/10,y)  # /10 to rescale units ... WARNING not very robust !!

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







