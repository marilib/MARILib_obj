#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np

from scipy.optimize import SR1, NonlinearConstraint, minimize, fsolve

import unit

import optim2d


def geometry_solver(aircraft):
    """Perform geometrical design
    """
    aircraft.cabin.eval_geometry()
    aircraft.fuselage.eval_geometry()
    aircraft.wing.eval_geometry()
    aircraft.tank.eval_geometry()
    aircraft.nacelles.eval_geometry()

    aircraft.htp.solve_area()                     # Solver inside
    aircraft.vtp.solve_area()                     # Solver inside

    aircraft.landing_gears.eval_geometry()
    aircraft.systems.eval_geometry()
    aircraft.geometry.eval_wet_area()



def mass_mission_adaptation(aircraft):
    """Perform Mass - Mission Adaptation
    """
    def fct(x):
        aircraft.mass.mtow = x[0]
        aircraft.mass.mzfw = x[1]
        aircraft.mass.mlw = x[2]

        aircraft.mass.eval_equiped_mass()
        aircraft.missions.eval_nominal_mission()
        aircraft.mass.eval_characteristic_mass()

        return [x[0]-aircraft.mass.mtow,
                x[1]-aircraft.mass.mzfw,
                x[2]-aircraft.mass.mlw]

    xini = [float(aircraft.mass.mtow), float(aircraft.mass.mzfw), float(aircraft.mass.mlw)]
    output_dict = fsolve(fct, x0=xini, args=(), full_output=True)
    if (output_dict[2]!=1): raise Exception("Convergence problem")

    aircraft.mass.mtow = output_dict[0][0]
    aircraft.mass.mzfw = output_dict[0][1]
    aircraft.mass.mlw = output_dict[0][2]

    aircraft.mass.eval_equiped_mass()
    aircraft.missions.eval_nominal_mission()
    aircraft.mass.eval_characteristic_mass()



def mda(aircraft):
    """Perform Multidsciplinary_Design_Analysis
    All coupling constraints are solved in a relevent order
    """
    geometry_solver(aircraft)                       # Solver inside

    mass_mission_adaptation(aircraft)               # Solver inside

    aircraft.missions.eval_payload_range_solver()   # Solver inside

    aircraft.operations.eval()

    aircraft.economics.eval()



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
        print("--------------------------------------------------------------")
        print("start_value = ", start_value)
        print("--------------------------------------------------------------")

        crt_mag, unused = self.eval_optim_data(start_value, aircraft, var, cst, cst_mag, crt, 1.)

        if method == 'trust-constr':

            res = self.scipy_trust_constraint(aircraft,start_value,var,var_bnd,cst,cst_mag,crt,crt_mag)

        elif method == 'optim2d':

            pass

            # xini =
            # dxini =
            # names =
            # units =
            # nzoom =
            # lwbs =
            # lwfc =
            # upbs =
            # upfc =
            # crfc =
            # fct_scitwod =
            # graph =
            #
            # my_zed_df, ref_cell_pts_on_grid_list, my_zed_df_cell, my_sol \
            # = optim2d.optim2d_(xini,dxini, names,units, nzoom, lwbs,lwfc, upbs,upfc, crfc, fct_scitwod, graph)





        print(res)


    def scipy_trust_constraint(self,aircraft,start_value,var,var_bnd,cst,cst_mag,crt,crt_mag):
        """
        Run the trust-constraint minimization procedure :func:`scipy.optimize.minimize` to minimize a given criterion
        and satisfy given constraints for a given aircraft.
        """
        def cost(x,*args):
            return self.eval_optim_data_checked(x,*args)[0]

        def constraints(x):
            return self.eval_optim_data_checked(x,aircraft,var,cst,cst_mag,crt,crt_mag)[1]

        res = minimize(cost, start_value, args=(aircraft,var,cst,cst_mag,crt,crt_mag,),
                       constraints=NonlinearConstraint(fun=constraints,
                                                       lb=0., ub=np.inf, jac='3-point'),
                       method="trust-constr", jac="3-point", bounds=var_bnd, hess=SR1(), hessp=None,
                       options={'maxiter':500,'xtol':1.e-6})
                       #options={'maxiter':500,'xtol': np.linalg.norm(start_value)*0.01,
                       #         'initial_tr_radius': np.linalg.norm(start_value)*0.05 })
        return res


