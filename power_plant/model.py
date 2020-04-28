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

one_hour = 3600.
one_day = one_hour * 24.
one_year = one_day * 365.




class Material(object):

    def __init__(self):
        self.concrete = None
        self.steel = None
        self.aluminium = None
        self.copper = None
        self.lead = None
        self.plastic = None
        self.silicon = None
        self.glass = None



class PowerPlant(object):

    def __init__(self):
        self.total_footprint = None
        self.total_grey_enrg = None

        self.nominal_peak_power = None
        self.nominal_mean_power = None
        self.mean_dayly_energy = None

        self.regulation_time = None
        self.retrieval_time = None
        self.regulated_power = None
        self.storage_capacity = None
        self.regulation_power_efficiency = None

        self.gross_yearly_enrg = None
        self.net_yearly_enrg = None

        self.marginal_efficiency = None
        self.net_power_efficiency = None

        self.material = Material()

    def update(self):
        raise NotImplementedError

    def print(self, *kwargs):
        print("")
        if len(kwargs)>0: print("Info = "+kwargs[0])
        print("Power plant type = ",self.type)
        print("--------------------------------------------------")
        print("Nominal peak power = ", "%8.1f" % unit.MW_W(self.nominal_peak_power), " MW")
        print("Nominal mean power = ", "%8.1f" % unit.MW_W(self.nominal_mean_power), " MW")
        print("Regulated power = ", "%8.1f" % unit.MW_W(self.regulated_power), " MW")
        print("Regulation time = ", "%8.1f" % unit.h_s(self.regulation_time), " h")
        print("Retrieval time = ", "%8.1f" % unit.h_s(self.retrieval_time), " h")
        print("Storage capacity = ", "%8.1f" % unit.MWh_J(self.storage_capacity), " MWh")
        print("Yearly gross production = ", "%8.1f" % unit.GWh_J(self.gross_yearly_enrg), " GWh")
        print("Yearly net production = ", "%8.1f" % unit.GWh_J(self.net_yearly_enrg), " GWh")
        print("Marginal efficiency = ", "%8.3f" % self.marginal_efficiency)
        print("Net power efficiency = ", "%8.3f" % self.net_power_efficiency)
        print("Total grey energy = ", "%8.1f" % unit.GWh_J(self.total_grey_enrg), " GWh")
        print("Total footprint = ", "%8.1f" % unit.km2_m2(self.total_footprint), " km2")
        print("   Concrete = ", "%8.0f" % (self.material.concrete*1.e-3), " t")
        print("      Steel = ", "%8.0f" % (self.material.steel*1.e-3), " t")
        print("  Aluminium = ", "%8.0f" % (self.material.aluminium*1.e-3), " t")
        print("     Copper = ", "%8.0f" % (self.material.copper*1.e-3), " t")
        print("       Lead = ", "%8.0f" % (self.material.lead*1.e-3), " t")
        print("    Plastic = ", "%8.0f" % (self.material.plastic*1.e-3), " t")
        print("    Silicon = ", "%8.0f" % (self.material.silicon*1.e-3), " t")
        print("      Glass = ", "%8.0f" % (self.material.glass*1.e-3), " t")


class StPowerPlant(PowerPlant):

    def __init__(self, n_mirror, mean_sun_pw,
                 reg_factor = 0.,
                 mirror_area = 68.,
                 ground_ratio = 4.,
                 life_time = 25.,
                 gross_power_efficiency = 0.39,
                 storage_efficiency = 0.99,
                 grey_energy_ratio = 0.11,
                 load_factor = 0.38):
        super(StPowerPlant, self).__init__()

        self.type = "Cylindroparabolic mirror"
        self.regulation_factor = reg_factor
        self.n_mirror = n_mirror
        self.mirror_area = mirror_area
        self.ground_ratio = ground_ratio
        self.life_time = life_time

        self.gross_power_efficiency = gross_power_efficiency
        self.storage_efficiency = storage_efficiency
        self.grey_energy_ratio = grey_energy_ratio

        self.mean_yearly_sun_power = mean_sun_pw
        self.load_factor = load_factor

        self.total_mirror_area = None

        self.update()

    def update(self):
        self.total_mirror_area = self.mirror_area * self.n_mirror
        self.total_footprint = self.total_mirror_area * self.ground_ratio

        self.nominal_peak_power = self.mean_yearly_sun_power * self.total_mirror_area * self.gross_power_efficiency
        self.nominal_mean_power = self.nominal_peak_power * self.load_factor
        self.mean_dayly_energy = self.nominal_mean_power * one_day

        self.regulation_time = one_day * (self.load_factor + (1. - self.load_factor)*self.regulation_factor)
        self.retrieval_time = one_day * (1. - self.load_factor)*self.regulation_factor
        self.regulated_power = self.nominal_peak_power / (1. + self.regulation_factor*(1.-self.load_factor)/(self.load_factor*self.storage_efficiency))
        self.storage_capacity = (self.nominal_peak_power - self.regulated_power) * self.load_factor * self.storage_efficiency * one_day
        self.regulation_power_efficiency = self.regulated_power * self.regulation_time / self.mean_dayly_energy

        self.marginal_efficiency = self.regulation_power_efficiency * (1.-self.grey_energy_ratio)
        self.net_power_efficiency = self.gross_power_efficiency * self.marginal_efficiency

        self.gross_yearly_enrg = self.regulated_power * self.regulation_time * 365.
        self.net_yearly_enrg = self.gross_yearly_enrg - self.gross_yearly_enrg * self.grey_energy_ratio

        self.total_grey_enrg = self.gross_yearly_enrg * self.grey_energy_ratio * self.life_time

        self.material.concrete = np.nan
        self.material.steel = np.nan
        self.material.aluminium = np.nan
        self.material.copper = np.nan
        self.material.lead = np.nan
        self.material.plastic = np.nan
        self.material.silicon = np.nan
        self.material.glass = np.nan


class PvPowerPlant(PowerPlant):

    def __init__(self, n_panel, mean_sun_pw,
                 ref_sun_power = 1000.,
                 reg_factor = 0.,
                 panel_area = 2.,
                 ground_ratio = 2.6,
                 life_time = 25.,
                 gross_power_efficiency = 0.15,
                 storage_efficiency = 0.80,
                 grey_energy_ratio = 0.11,
                 load_factor = 0.14):
        super(PvPowerPlant, self).__init__()

        self.type = "Photovoltaïc panel"
        self.ref_sun_power = ref_sun_power
        self.regulation_factor = reg_factor
        self.n_panel = n_panel
        self.panel_area = panel_area
        self.ground_ratio = ground_ratio
        self.life_time = life_time

        self.gross_power_efficiency = gross_power_efficiency
        self.storage_efficiency = storage_efficiency
        self.grey_energy_ratio = grey_energy_ratio

        self.mean_yearly_sun_power = mean_sun_pw
        self.ref_yearly_sun_power = self.ref_sun_power * (mean_sun_pw/250.)
        self.load_factor = load_factor

        self.total_panel_area = None

        self.update()

    def update(self):
        self.total_panel_area = self.panel_area * self.n_panel
        self.total_footprint = self.total_panel_area * self.ground_ratio

        self.nominal_peak_power = self.ref_yearly_sun_power * self.total_panel_area * self.gross_power_efficiency
        self.nominal_mean_power = self.nominal_peak_power * self.load_factor
        self.mean_dayly_energy = self.nominal_mean_power * one_day

        self.regulation_time = one_day * (self.load_factor + (1. - self.load_factor)*self.regulation_factor)
        self.retrieval_time = one_day * (1. - self.load_factor)*self.regulation_factor
        self.regulated_power = self.nominal_peak_power / (1. + self.regulation_factor*(1.-self.load_factor)/(self.load_factor*self.storage_efficiency))
        self.storage_capacity = (self.nominal_peak_power - self.regulated_power) * self.load_factor * self.storage_efficiency * one_day
        self.regulation_power_efficiency = self.regulated_power * self.regulation_time / self.mean_dayly_energy

        self.marginal_efficiency = self.regulation_power_efficiency * (1.-self.grey_energy_ratio)
        self.net_power_efficiency = self.gross_power_efficiency * self.marginal_efficiency

        self.gross_yearly_enrg = self.regulated_power * self.regulation_time * 365.
        self.net_yearly_enrg = self.gross_yearly_enrg * (1. - self.grey_energy_ratio)

        self.total_grey_enrg = self.gross_yearly_enrg * self.grey_energy_ratio * self.life_time

        self.material.concrete = 0.
        self.material.steel = 16.5 * self.total_panel_area
        self.material.aluminium = 0.6 * self.total_panel_area
        self.material.copper = 0.011 * self.total_panel_area
        self.material.lead = 0.
        self.material.plastic = 0.12 * self.total_panel_area
        self.material.silicon = 2.15 * self.total_panel_area
        self.material.glass = 8.5 * self.total_panel_area



class EolPowerPlant(PowerPlant):

    def __init__(self, n_rotor, location,
                 reg_factor = 0.,
                 rotor_width = 100.,
                 rotor_peak_power = 2.5e6,
                 load_factor = None,
                 storage_efficiency = 0.8,
                 life_time = 25.):
        super(EolPowerPlant, self).__init__()

        self.type = "Wind turbine " + location
        self.location = location
        self.regulation_factor = reg_factor
        self.n_rotor = n_rotor
        self.rotor_width = rotor_width
        self.rotor_peak_power = rotor_peak_power
        self.load_factor = load_factor
        self.storage_efficiency = storage_efficiency
        self.life_time = life_time

        self.rotor_area = None
        self.rotor_footprint = None
        self.rotor_grey_enrg = None

        self.update()

    def update(self):
        self.rotor_area = 0.25*np.pi*self.rotor_width**2

        self.rotor_footprint = {"onshore":0.40e6*(self.rotor_width/90.),
                                "offshore":1.00e6*(self.rotor_width/90.)
                                }.get(self.location, "Error, location is unknown")

        self.total_footprint = self.rotor_footprint * self.n_rotor

        if (self.load_factor is None):
            self.load_factor = {"onshore":0.25,
                                "offshore":0.50
                                }.get(self.location, "Error, location is unknown")

        self.rotor_grey_enrg = {"onshore": unit.J_GWh(1.27) * (self.rotor_area / 6362.),
                                "offshore": unit.J_GWh(2.28) * (self.rotor_area / 6362.)
                                }.get(self.location, "Error, location is unknown")

        self.total_grey_enrg = self.rotor_grey_enrg * self.n_rotor

        self.nominal_peak_power = self.rotor_peak_power * self.n_rotor
        self.nominal_mean_power = self.nominal_peak_power * self.load_factor
        self.mean_dayly_energy = self.nominal_mean_power * one_day

        self.regulation_time = one_day * (self.load_factor + (1. - self.load_factor)*self.regulation_factor)
        self.retrieval_time = one_day * (1. - self.load_factor)*self.regulation_factor
        self.regulated_power = self.nominal_peak_power / (1. + self.regulation_factor*(1.-self.load_factor)/(self.load_factor*self.storage_efficiency))
        self.storage_capacity = (self.nominal_peak_power - self.regulated_power) * self.load_factor * self.storage_efficiency * one_day
        self.regulation_power_efficiency = self.regulated_power * self.regulation_time / self.mean_dayly_energy

        self.gross_yearly_enrg = self.regulated_power * self.regulation_time * 365.
        self.net_yearly_enrg = self.gross_yearly_enrg - self.total_grey_enrg / self.life_time

        self.marginal_efficiency = self.net_yearly_enrg / (self.nominal_mean_power * one_year)
        self.net_power_efficiency =  np.nan

        self.material.concrete = {"onshore":320000., "offshore":640000.}.get(self.location) * self.nominal_peak_power * 1e-6
        self.material.steel = {"onshore":61500., "offshore":91500.}.get(self.location) * self.nominal_peak_power * 1e-6
        self.material.aluminium = {"onshore":1600., "offshore":1600.}.get(self.location) * self.nominal_peak_power * 1e-6
        self.material.copper = {"onshore":400., "offshore":3400.}.get(self.location) * self.nominal_peak_power * 1e-6
        self.material.lead = {"onshore":0., "offshore":4000.}.get(self.location) * self.nominal_peak_power * 1e-6
        self.material.plastic = {"onshore":2200., "offshore":2900.}.get(self.location) * self.nominal_peak_power * 1e-6
        self.material.silicon = {"onshore":0., "offshore":0.}.get(self.location) * self.nominal_peak_power * 1e-6
        self.material.glass = {"onshore":0., "offshore":0.}.get(self.location) * self.nominal_peak_power * 1e-6




class NuclearPowerPlant(PowerPlant):

    def __init__(self, n_unit,
                 unit_peak_power = 1.0e9,
                 load_factor = 0.75,
                 life_time = 50.,
                 grey_energy_ratio = 0.2,
                 unit_footprint = 0.15e6):
        super(NuclearPowerPlant, self).__init__()

        self.type = "Nuclear reactor"
        self.n_unit = n_unit
        self.unit_peak_power = unit_peak_power
        self.load_factor = load_factor
        self.life_time = life_time
        self.grey_energy_ratio = grey_energy_ratio
        self.unit_footprint = unit_footprint

        self.update()

    def update(self):
        self.nominal_peak_power = self.unit_peak_power * self.n_unit
        self.nominal_mean_power = self.nominal_peak_power * self.load_factor
        self.total_footprint = self.unit_footprint * self.n_unit

        self.regulation_time = one_day
        self.retrieval_time = 0.
        self.regulated_power = self.nominal_mean_power
        self.storage_capacity = 0.
        self.regulation_power_efficiency = 1.0

        self.mean_dayly_energy = self.nominal_mean_power * one_day
        self.gross_yearly_enrg = self.nominal_mean_power * one_year

        self.total_grey_enrg = self.gross_yearly_enrg * self.life_time * self.grey_energy_ratio
        self.net_yearly_enrg = self.gross_yearly_enrg * self.life_time * (1. - self.grey_energy_ratio) / self.life_time

        self.marginal_efficiency = self.net_yearly_enrg / (self.nominal_mean_power * one_year)
        self.net_power_efficiency =  np.nan

        # ref : for waste storage
        # Life cycle energy and greenhouse gas emissions of nuclear energy:
        # A review
        # Manfred Lenzen
        # *
        # ISA, Centre for Integrated Sustainability Analysis, The University of Sydney, Physics Building A28, Sydney, NSW 2006, Australia
        # Received 13 June 2007; accepted 31 January 2008
        # Available online 8 April 2008
        self.material.concrete =   320.e3 * self.nominal_peak_power * 1e-6 \
                                 + (373.e3/177.) * self.mean_dayly_energy * self.life_time * 1e-9   # Power plant + waste storage
        self.material.steel =  60.e3 * self.nominal_peak_power * 1e-6
        self.material.aluminium = 0.14e3 * self.nominal_peak_power * 1e-6
        self.material.copper = 1.51e3 * self.nominal_peak_power * 1e-6
        self.material.lead = np.nan
        self.material.plastic = np.nan
        self.material.silicon = np.nan
        self.material.glass = np.nan



def max_solar_input(latt,long,pamb,day,gmt):
    """Compute max solar radiation input from location and time on Earth

    :param latt: Lattitude in radians
    :param long: Longitude in radians
    :param day: Day of the year, from 1 to 365
    :param gmt: GMT time in the day, from 0. to 24.
    :return:
    """
    delta = unit.rad_deg(23.45 * np.sin(2.*np.pi*(284.+day)/365.))
    equ = 0. # output of time equation, neglected here
    solar_time = gmt + (4.*latt/unit.rad_deg(60.)) - equ
    eta = unit.rad_deg(15.*(12.-solar_time))
    sin_a = np.sin(latt) * np.sin(delta) + np.cos(latt)*np.cos(delta)*np.cos(eta)
    r_out = 1367. * (1. + 0.034*np.cos(2.*np.pi*(day/365.)))
    m0 = np.sqrt(1229. + (614.*sin_a)**2) - 614.*sin_a
    p0 = 101325.
    m = m0*(pamb/p0)
    tau = 0.6
    r_direct = r_out + tau**m * sin_a
    r_diffus = r_out * (0.271 - 0.294*tau**m) * sin_a
    r_total = r_direct + r_diffus
    return r_total



st1 = StPowerPlant(7500., 250., reg_factor=0.51)
st1.print("Andasol 3")

pv1 = PvPowerPlant(1e6, 250.)
pv1.print("Cestas")

eol1 = EolPowerPlant(240., "onshore")
eol1.print("Fantanele Cogealav")

eol2 = EolPowerPlant(175., "offshore", rotor_width=120., rotor_peak_power=3.5e6, load_factor=0.36)
eol2.print("London Array")

atom1 = NuclearPowerPlant(4)
atom1.print()


latt = unit.rad_deg(43.668731)
long = unit.rad_deg(1.497691)
pamb = 101325.
day = 182 #31+29+31+28
gmt = 5.

pw = max_solar_input(latt,long,pamb,day,gmt)

print("Solar power = ",pw)
