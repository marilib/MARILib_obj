#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np
from context import unit

data_dict = {
    "n_panel": {"unit":"int", "mag":1e4, "txt":"Number of individual panels"},
    "panel_area": {"unit":"m2", "mag":1e1, "txt":"Panel area"},
    "panel_mass": {"unit":"kg", "mag":1e1, "txt":"Panel mass"},
    "ground_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Required ground area over panel area"},
    "grey_energy_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Fraction of produced energy needed to build, maintain and recycle the production device"},
    "life_time": {"unit":"year", "mag":1e0, "txt":"Power plant reference life time"},
    "gross_power_efficiency": {"unit":"no_dim", "mag":1e0, "txt":"Ratio of produced power over solar input power"},
    "ref_yearly_sun_power": {"unit":"W/m2", "mag":1e0, "txt":"Yearly mean sun power at power plant location"},
    "load_factor": {"unit":"no_dim", "mag":1e0, "txt":"Ratio of yearly mean power over peak power"},
    "total_panel_area": {"unit":"m2", "mag":1e6, "txt":"Total area of the solar panels"},
    "foot_print": {"unit":"m2", "mag":1e6, "txt":"Ground footprint area of the power plant"},
    "net_power_efficiency": {"unit":"no_dim", "mag":1e0, "txt":"Power efficiency including grey energy spread over life time"},
    "nominal_peak_power": {"unit":"MW", "mag":1e2, "txt":"Output power with ref_yearly_sun_power is input"},
    "nominal_gross_power": {"unit":"MW", "mag":1e2, "txt":"Mean yearly output power"},
    "nominal_net_power": {"unit":"MW", "mag":1e2, "txt":"Mean yearly output power including grey energy spread over life time"},
    "gross_yearly_enrg": {"unit":"GWh", "mag":1e2, "txt":"Mean yearly energy production"},
    "net_yearly_enrg": {"unit":"GWh", "mag":1e2, "txt":"Mean yearly energy production including grey energy spread over life time"},
    "total_grey_enrg": {"unit":"GWh", "mag":1e3, "txt":"Total required grey energy over life time"},
}


one_year = 3600. * 24. * 365.


class PV_power_plant(object):

    def __init__(self, n_panel, ref_sun_pw):
        self.n_panel = n_panel
        self.panel_area = 1.
        self.ground_ratio = 2.6
        self.life_time = 20.

        self.grey_energy_ratio = 0.11
        self.gross_power_efficiency = 0.15

        self.ref_yearly_sun_power = ref_sun_pw
        self.load_factor = 0.13

        self.total_panel_area = self.panel_area * self.n_panel
        self.total_foot_print = self.total_panel_area * self.ground_ratio

        self.net_power_efficiency = self.gross_power_efficiency * (1.-self.grey_energy_ratio)

        self.nominal_peak_power = self.ref_yearly_sun_power * self.total_panel_area * self.gross_power_efficiency
        self.nominal_gross_power = self.ref_yearly_sun_power * self.load_factor * self.total_panel_area * self.gross_power_efficiency
        self.nominal_net_power = self.ref_yearly_sun_power * self.load_factor * self.total_panel_area * self.net_power_efficiency

        self.gross_yearly_enrg = self.nominal_gross_power * one_year
        self.net_yearly_enrg = self.nominal_net_power * one_year

        self.total_grey_enrg = self.grey_energy_ratio * self.gross_yearly_enrg * self.life_time

    def update(self, ):
        self.net_power_efficiency = self.gross_power_efficiency * (1.-self.grey_energy_ratio)
        self.total_panel_area = self.panel_area * self.n_panel
        self.total_foot_print = self.total_panel_area * self.ground_ratio
        self.nominal_gross_power = self.ref_yearly_sun_power * self.load_factor * self.total_panel_area * self.gross_power_efficiency
        self.nominal_net_power = self.ref_yearly_sun_power * self.load_factor * self.total_panel_area * self.net_power_efficiency
        self.gross_yearly_enrg = self.nominal_gross_power * (3600.*24.*365.)
        self.net_yearly_enrg = self.nominal_net_power * (3600.*24.*365.)
        self.total_grey_enrg = self.grey_energy_ratio * self.gross_yearly_enrg * self.life_time



pv1 = PV_power_plant(1e6, 1e3)

print("Nominal peak power = ", "%8.1f" % unit.MW_W(pv1.nominal_peak_power), " MW")
print("Nominal mean power = ", "%8.1f" % unit.MW_W(pv1.nominal_gross_power), " MW")
print("Yearly production = ", "%8.1f" % unit.GWh_J(pv1.gross_yearly_enrg), " GWh")
print("Total grey energy = ", "%8.1f" % unit.GWh_J(pv1.total_grey_enrg), " GWh")
print("Total footprint = ", "%8.1f" % unit.km2_m2(pv1.total_foot_print), " km2")





class EOL_power_plant(object):

    def __init__(self, n_rotor, location):
        self.location = location
        self.n_rotor = n_rotor
        self.rotor_width = 90.
        self.rotor_peak_power = 10.e6
        self.life_time = 20.

        self.load_factor = {"onshore":0.30,
                            "offshore":0.40
                            }.get(self.location, "Error, location is unknown")

        self.rotor_footprint = 0.25e6*(self.rotor_width/90.)

        self.rotor_area = 0.25*np.pi*self.rotor_width**2

        self.rotor_grey_enrg = {"onshore": unit.J_GWh(1.27) * (self.rotor_area / 6362.),
                                "offshore": unit.J_GWh(2.28) * (self.rotor_area / 6362.)
                                }.get(self.location, "Error, location is unknown")

        self.total_foot_print = self.rotor_footprint * self.n_rotor

        self.total_grey_enrg = self.rotor_grey_enrg * self.n_rotor

        self.nominal_peak_power = self.rotor_peak_power * self.n_rotor
        self.nominal_gross_power = self.nominal_peak_power * self.load_factor
        self.gross_yearly_enrg = self.nominal_gross_power * one_year

        self.net_yearly_enrg = (self.gross_yearly_enrg * self.life_time - self.total_grey_enrg) / self.life_time
        self.nominal_net_power = self.net_yearly_enrg / one_year

    def update(self, ):
        self.load_factor = {"onshore":0.30,
                            "offshore":0.40
                            }.get(self.location, "Error, location is unknown")
        self.rotor_footprint = 0.25e6*(self.rotor_width/90.)
        self.rotor_area = 0.25*np.pi*self.rotor_width**2
        self.rotor_grey_enrg = {"onshore": unit.J_GWh(1.27) * (self.rotor_area / 6362.),
                                "offshore": unit.J_GWh(2.28) * (self.rotor_area / 6362.)
                                }.get(self.location, "Error, location is unknown")
        self.total_foot_print = self.rotor_footprint * self.n_rotor
        self.total_grey_enrg = self.rotor_grey_energy * self.n_rotor
        self.nominal_peak_power = self.rotor_peak_power * self.n_rotor
        self.nominal_gross_power = self.nominal_peak_power * self.load_factor
        self.gross_yearly_enrg = self.nominal_gross_power * one_year
        self.net_yearly_enrg = (self.gross_yearly_enrg * self.life_time - self.total_grey_enrg) / self.life_time
        self.nominal_net_power = self.net_yearly_enrg / one_year



eol1 = EOL_power_plant(25, "onshore")

print("")
print("Nominal peak power = ", "%8.1f" % unit.MW_W(eol1.nominal_peak_power), " MW")
print("Nominal mean power = ", "%8.1f" % unit.MW_W(eol1.nominal_gross_power), " MW")
print("Yearly production = ", "%8.1f" % unit.GWh_J(eol1.gross_yearly_enrg), " GWh")
print("Total grey energy = ", "%8.1f" % unit.GWh_J(eol1.total_grey_enrg), " GWh")
print("Total footprint = ", "%8.1f" % unit.km2_m2(eol1.total_foot_print), " km2")
