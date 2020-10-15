#!/usr/bin/env python3
"""


:author: DRUOT Thierry, Nicolas Monrolin
"""

import numpy as np
from scipy.optimize import fsolve
from marilib.aircraft.airframe.component import Component, Nacelle, Tank, Pod

#--------------------------------------------------------------------------------------------------------------------------------
class Airframe(object):
    """Logical aircraft components
    """

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.engine_analysis_order = []
        self.mass_analysis_order = []

    def __iter__(self):
        public = [value for value in self.__dict__.values() if issubclass(type(value),Component)]
        return iter(public)

    def mass_iter(self):
        component_list = []
        for name in self.mass_analysis_order:
            component_list.append(self.__dict__[name])
        return iter(component_list)

    def geometry_analysis(self, hq_optim=False):
        """Crude geometry analysis of each component independantly by calling component.eval_geometry()
        """
        stab_architecture = self.aircraft.arrangement.stab_architecture
        number_of_engine = self.aircraft.arrangement.number_of_engine
        power_architecture = self.aircraft.arrangement.power_architecture

        self.aircraft.airframe.cabin.eval_geometry()
        self.aircraft.airframe.body.eval_geometry()
        self.aircraft.airframe.wing.eval_geometry(hq_optim)

        for comp in self.aircraft.airframe:
            if issubclass(type(comp),Tank):
                comp.eval_geometry()

        for comp in self.aircraft.airframe:
            if issubclass(type(comp),Pod):
                comp.eval_geometry()

        self.aircraft.airframe.cargo.eval_geometry()

        for comp in self.aircraft.airframe:
            if issubclass(type(comp),Nacelle):
                comp.eval_geometry()

        if (stab_architecture in ["classic","t_tail"]):
            self.aircraft.airframe.vertical_stab.eval_geometry()
            self.aircraft.airframe.horizontal_stab.eval_geometry()
        elif (stab_architecture=="h_tail"):
            self.aircraft.airframe.horizontal_stab.eval_geometry()
            self.aircraft.airframe.vertical_stab.eval_geometry()
            self.aircraft.airframe.other_vertical_stab.eval_geometry()

        self.aircraft.airframe.landing_gear.eval_geometry()
        self.aircraft.airframe.system.eval_geometry()

    def statistical_pre_design(self):
        """Solves strong coupling and compute tail areas using volume coefficients
        """
        stab_architecture = self.aircraft.arrangement.stab_architecture
        number_of_engine = self.aircraft.arrangement.number_of_engine
        power_architecture = self.aircraft.arrangement.power_architecture

        self.aircraft.airframe.cabin.eval_geometry()
        self.aircraft.airframe.body.eval_geometry()
        self.aircraft.airframe.wing.eval_geometry()

        for comp in self.aircraft.airframe:
            if issubclass(type(comp),Tank):
                comp.eval_geometry()

        for comp in self.aircraft.airframe:
            if issubclass(type(comp),Pod):
                comp.eval_geometry()

        self.aircraft.airframe.cargo.eval_geometry()

        for comp in self.aircraft.airframe:
            if issubclass(type(comp),Nacelle):
                comp.eval_geometry()

        def fct(x_in):
            self.aircraft.airframe.vertical_stab.area = x_in[0]                           # Coupling variable
            self.aircraft.airframe.horizontal_stab.area = x_in[1]                             # Coupling variable

            if (stab_architecture in ["classic","t_tail"]):
                self.aircraft.airframe.vertical_stab.eval_geometry()
                self.aircraft.airframe.horizontal_stab.eval_geometry()
            elif (stab_architecture=="h_tail"):
                self.aircraft.airframe.horizontal_stab.eval_geometry()
                self.aircraft.airframe.vertical_stab.eval_geometry()
                self.aircraft.airframe.other_vertical_stab.eval_geometry()

            self.aircraft.airframe.horizontal_stab.eval_area()
            self.aircraft.airframe.vertical_stab.eval_area()

            y_out = np.array([x_in[0] - self.aircraft.airframe.vertical_stab.area,
                              x_in[1] - self.aircraft.airframe.horizontal_stab.area])
            return y_out

        x_ini = np.array([self.aircraft.airframe.vertical_stab.area,
                          self.aircraft.airframe.horizontal_stab.area])

        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        if (stab_architecture in ["classic","t_tail"]):
            self.aircraft.airframe.vertical_stab.area = output_dict[0][0]
            self.aircraft.airframe.vertical_stab.eval_geometry()
            self.aircraft.airframe.horizontal_stab.area = output_dict[0][1]
            self.aircraft.airframe.horizontal_stab.eval_geometry()
        elif (stab_architecture=="h_tail"):
            self.aircraft.airframe.horizontal_stab.area = output_dict[0][1]
            self.aircraft.airframe.horizontal_stab.eval_geometry()
            self.aircraft.airframe.vertical_stab.area = output_dict[0][0]
            self.aircraft.airframe.vertical_stab.eval_geometry()
            self.aircraft.airframe.other_vertical_stab.area = output_dict[0][0]
            self.aircraft.airframe.other_vertical_stab.eval_geometry()

        self.aircraft.airframe.landing_gear.eval_geometry()
        self.aircraft.airframe.system.eval_geometry()


