#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import numpy as np
from scipy.optimize import fsolve


#--------------------------------------------------------------------------------------------------------------------------------
class Airframe(object):
    """
    Logical aircraft description
    """

    def __init__(self, aircraft):
        self.aircraft = aircraft
        self.mass_analysis_order = []

    def mass_iter(self):
        component_list = []
        for name in self.mass_analysis_order:
            component_list.append(self.__dict__[name])
        return iter(component_list)

    def geometry_analysis(self):
        stab_architecture = self.aircraft.arrangement.stab_architecture

        self.aircraft.airframe.cabin.eval_geometry()
        self.aircraft.airframe.body.eval_geometry()
        self.aircraft.airframe.wing.eval_geometry()
        self.aircraft.airframe.cargo.eval_geometry()
        self.aircraft.airframe.nacelle.eval_geometry()

        if (self.aircraft.arrangement.stab_architecture in ["classic","t_tail"]):
            self.aircraft.airframe.vertical_stab.eval_geometry()
            self.aircraft.airframe.horizontal_stab.eval_geometry()
        elif (self.aircraft.arrangement.stab_architecture=="h_tail"):
            self.aircraft.airframe.horizontal_stab.eval_geometry()
            self.aircraft.airframe.vertical_stab.eval_geometry()

        self.aircraft.airframe.vertical_stab.eval_area()
        self.aircraft.airframe.horizontal_stab.eval_area()

        self.aircraft.airframe.tank.eval_geometry()
        self.aircraft.airframe.landing_gear.eval_geometry()
        self.aircraft.airframe.system.eval_geometry()

    def statistical_pre_design(self):
        """
        Solves strong coupling and compute tail areas using volume coefficients
        """
        self.aircraft.airframe.cabin.eval_geometry()
        self.aircraft.airframe.body.eval_geometry()
        self.aircraft.airframe.wing.eval_geometry()
        self.aircraft.airframe.cargo.eval_geometry()
        self.aircraft.airframe.nacelle.eval_geometry()

        def fct(x_in):
            self.aircraft.airframe.vertical_stab.area = x_in[0]                           # Coupling variable
            self.aircraft.airframe.horizontal_stab.area = x_in[1]                             # Coupling variable

            if (self.aircraft.arrangement.stab_architecture in ["classic","t_tail"]):
                self.aircraft.airframe.vertical_stab.eval_geometry()
                self.aircraft.airframe.horizontal_stab.eval_geometry()
            elif (self.aircraft.arrangement.stab_architecture=="h_tail"):
                self.aircraft.airframe.horizontal_stab.eval_geometry()
                self.aircraft.airframe.vertical_stab.eval_geometry()

            y_out = np.array([x_in[0] - self.aircraft.airframe.vertical_stab.area,
                              x_in[1] - self.aircraft.airframe.horizontal_stab.area])
            return y_out

        x_ini = np.array([self.aircraft.airframe.vertical_stab.area,
                          self.aircraft.airframe.horizontal_stab.area])

        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.aircraft.airframe.vertical_stab.area = output_dict[0][0]                           # Coupling variable
        self.aircraft.airframe.horizontal_stab.area = output_dict[0][1]                             # Coupling variable

        if (self.aircraft.arrangement.stab_architecture in ["classic","t_tail"]):
            self.aircraft.airframe.vertical_stab.eval_geometry()
            self.aircraft.airframe.horizontal_stab.eval_geometry()
        elif (self.aircraft.arrangement.stab_architecture=="h_tail"):
            self.aircraft.airframe.horizontal_stab.eval_geometry()
            self.aircraft.airframe.vertical_stab.eval_geometry()

        self.aircraft.airframe.tank.eval_geometry()
        self.aircraft.airframe.landing_gear.eval_geometry()
        self.aircraft.airframe.system.eval_geometry()


