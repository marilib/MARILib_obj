#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np
from context import unit

# TODO : comlete dictionnary
# TODO : add data for heat and elec storage


one_hour = 3600.
one_day = one_hour * 24.
one_year = one_day * 365.


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
    "total_footprint": {"unit":"km2", "mag":1e1, "txt":"Plant total footprint"},
    "total_grey_enrg": {"unit":"TWh", "mag":1e0, "txt":"Plant total embodied energy"},
    "nominal_peak_power": {"unit":"MW", "mag":1e2, "txt":"Plant peak power"},
    "nominal_mean_power": {"unit":"MW", "mag":1e2, "txt":"Mean gross power during one day"},
    "mean_dayly_energy": {"unit":"GWh", "mag":1e2, "txt":"Mean gross energy pruduced in one year"},
    "regulation_time": {"unit":"h", "mag":1e1, "txt":"Regulated period (period during which regulated power can be maintained)"},
    "retrieval_time": {"unit":"h", "mag":1e1, "txt":"Retrieving period (power is coming from stored energy)"},
    "regulated_power": {"unit":"MW", "mag":1e2, "txt":"Mean regulated output power during regulated period"},
    "storage_capacity": {"unit":"GWh", "mag":1e2, "txt":"Retrievable stored energy"},
    "regulation_power_efficiency": {"unit":"", "mag":1e3, "txt":"Efficiency of regulated power (including storage losses)"},
    "gross_yearly_enrg": {"unit":"TWh", "mag":1e2, "txt":"Mean gross energy produced in one year"},
    "net_yearly_enrg": {"unit":"TWh", "mag":1e2, "txt":"Mean net energy produced in one year : gross minus grey"},
    "marginal_efficiency": {"unit":"no_dim", "mag":1e0, "txt":"efficiency which includes grey energy and storage if any"},
    "net_power_efficiency": {"unit":"no_dim", "mag":1e0, "txt":"overall efficiency : captation, transformation and marginal"},
    "type": {"unit":"string", "mag":32, "txt":"Power plant technology"},
    "regulation_factor": {"unit":"no_dim", "mag":1e0, "txt":"Regulation factor, 0.:no storage, 1.:regulation period is 24h"},
    "n_mirror": {"unit":"", "mag":1e3, "txt":"Number of mirror unit"},
    "n_panel": {"unit":"int", "mag":1e6, "txt":"Number of photovoltaïc panel unit"},
    "n_rotor": {"unit":"int", "mag":1e2, "txt":"Number of rotor"},
    "n_core": {"unit":"int", "mag":1e0, "txt":"Number of nuclear core"},
    "n_unit": {"unit":"int", "mag":1e3, "txt":"Number of production unit"},
    "mirror_area": {"unit":"m2", "mag":1e0, "txt":"Area of a mirror unit"},
    "panel_area": {"unit":"m2", "mag":1e0, "txt":"Area of photovoltaïc panel unit"},
    "total_mirror_area": {"unit":"m2", "mag":1e5, "txt":"Total mirror area of the plant"},
    "total_panel_area": {"unit":"m2", "mag":1e5, "txt":"Total panel area of the plant"},
    "ground_ratio": {"unit":"", "mag":1e3, "txt":"Required footprint area per unit area"},
    "life_time": {"unit":"year", "mag":1e1, "txt":"Plant life time"},
    "gross_power_efficiency": {"unit":"", "mag":1e3, "txt":"Gross output power over available input power"},
    "storage_medium": {"unit":"string", "mag":32, "txt":"Type of storage medium depending on plant technology"},
    "grey_energy_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Required embodied energy over gros output energy"},
    "mean_yearly_sun_power": {"unit":"W/m2", "mag":1e2, "txt":"Mean input sun radiative power over a year"},
    "load_factor": {"unit":"", "mag":1e0, "txt":"Yearl mean output power over device peak power"},
    "ref_sun_power": {"unit":"", "mag":1e3, "txt":"Position free reference input radiative sun power, generally 1000W/m2"},
    "mean_sun_power": {"unit":"", "mag":1e2, "txt":"Effective mean input radiative sun power"},
    "rotor_width": {"unit":"m", "mag":1e2, "txt":"Rotor diameter"},
    "rotor_area": {"unit":"m2", "mag":1e4, "txt":"Rotor disk area"},
    "rotor_footprint": {"unit":"km2", "mag":1e3, "txt":"Required footprint for one rotor"},
    "rotor_grey_enrg": {"unit":"GWh", "mag":1e2, "txt":"Embodied energy for one rotor"}
}



class Material(object):

    def __init__(self):
        self.concrete = 0.
        self.iron = 0.
        self.steel = 0.
        self.aluminium = 0.
        self.copper = 0.
        self.lead = 0.
        self.silicon = 0.
        self.plastic = 0.
        self.fiber_glass = 0.
        self.glass = 0.
        self.oil = 0.
        self.Na2CO3 = 0.
        self.KNO3 = 0.



class GreyEnergy(object):
    """Embodied energy of materials  (in J/kg)
    Source:
    G.P.Hammond and C.I.Jones (2006) Embodied energy and carbon footprint database, Department of Mechanical Engineering, University of Bath, United Kingdom
    Embodied energy in thermal energy storage (TES) systems for high temperature applications
    """
    def __init__(self):
        self.concrete_ge = 1.11e6
        self.iron_ge = 25.0e6
        self.steel_ge = 20.1e6
        self.aluminium_ge = 155.0e6
        self.copper_ge = 42.0e6
        self.lead_ge = 25.2e6
        self.silicon_ge = 15.0e6
        self.plastic_ge = 75.0e6
        self.fiber_glass_ge = 28.0e6
        self.glass_ge = 15.0e6
        self.oil_ge = 50.0e6
        self.Na2CO3_ge = 16.0e6
        self.KNO3_ge = 16.0e6



class Scarcity(object):
    """World wide stocks of some minerals (in kg)
    """
    def __init__(self):
        self.sand = 192.e18     # ref: Wikipedia
        self.iron = 87.e12      # ref Wikipedia
        self.aluminium = 28.e12 # ref: Wikipedia
        self.copper = 630.e9    # ref: Wikipedia



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
        if (self.material.concrete > 0.):     print("    Concrete = ", "%8.0f" % (self.material.concrete*1.e-3), " t")
        if (self.material.iron > 0.):         print("        Iron = ", "%8.0f" % (self.material.iron*1.e-3), " t")
        if (self.material.steel > 0.):        print("       Steel = ", "%8.0f" % (self.material.steel*1.e-3), " t")
        if (self.material.aluminium > 0.):    print("   Aluminium = ", "%8.0f" % (self.material.aluminium*1.e-3), " t")
        if (self.material.copper > 0.):       print("      Copper = ", "%8.0f" % (self.material.copper*1.e-3), " t")
        if (self.material.lead > 0.):         print("        Lead = ", "%8.0f" % (self.material.lead*1.e-3), " t")
        if (self.material.silicon > 0.):      print("     Silicon = ", "%8.0f" % (self.material.silicon*1.e-3), " t")
        if (self.material.plastic > 0.):      print("     Plastic = ", "%8.0f" % (self.material.plastic*1.e-3), " t")
        if (self.material.fiber_glass > 0.):  print(" Fiber_glass = ", "%8.0f" % (self.material.fiber_glass*1.e-3), " t")
        if (self.material.glass > 0.):        print("       Glass = ", "%8.0f" % (self.material.glass*1.e-3), " t")
        if (self.material.oil > 0.):          print("         Oil = ", "%8.0f" % (self.material.oil*1.e-3), " t")
        if (self.material.Na2CO3 > 0.):        print("       Na2CO3 = ", "%8.0f" % (self.material.Na2CO3*1.e-3), " t")
        if (self.material.KNO3 > 0.):         print("        KNO3 = ", "%8.0f" % (self.material.KNO3*1.e-3), " t")



def elec_storage(storage_type, *kwargs):
    """Provide storage efficiency or add materials required for heat storage

    :param storage_type: "electrolysis", "flow_battery"
    :param energy_capacity:
    :param material:
    :return:
    """
    if (storage_type=="electrolysis"):
        storage_efficiency = 0.70
    elif (storage_type=="flow_battery"):
        storage_efficiency = 0.80
    else:
        raise Exception("Type of electricity storage is unknown")

    if (len(kwargs)==0):
        return storage_efficiency
    else:
        energy_capacity = kwargs[0]
        material = kwargs[1]

    if (storage_type=="electrolysis"):
        pass
    elif (storage_type=="flow_battery"):
        pass




def heat_storage(storage_type, *kwargs):
    """Provide storage efficiency or add materials required for heat storage

    :param storage_type: "molten_salt", "concrete"
    :param energy_capacity:
    :param material:
    :return:
    """
    if (storage_type=="molten_salt"):
        storage_efficiency = 0.99
    elif (storage_type=="concrete"):
        storage_efficiency = 0.99
    else:
        raise Exception("Type of heat storage is unknown")

    if (len(kwargs)==0):
        return storage_efficiency
    else:
        energy_capacity = kwargs[0]
        material = kwargs[1]

    if (storage_type=="molten_salt"):
        material.Na2CO3 += 45.6e3 * energy_capacity*1e-6
        material.KNO3 += 30.4e3 * energy_capacity*1e-6
        material.concrete += 10.1e3 * energy_capacity*1e-6
        material.steel += 4.6e3 * energy_capacity*1e-6
        material.fiber_glass += 0.16e3 * energy_capacity*1e-6
    elif (storage_type=="concrete"):
        pass



class CspPowerPlant(PowerPlant):

    def __init__(self, n_mirror, mean_sun_pw,
                 reg_factor = 0.,
                 mirror_area = 68.,
                 ground_ratio = 4.,
                 life_time = 25.,
                 gross_power_efficiency = 0.39,
                 storage_medium = "molten_salt",
                 grey_energy_ratio = 0.11,
                 load_factor = 0.38):
        super(CspPowerPlant, self).__init__()

        self.type = "Cylindroparabolic mirror"
        self.regulation_factor = reg_factor
        self.n_mirror = n_mirror
        self.mirror_area = mirror_area
        self.ground_ratio = ground_ratio
        self.life_time = life_time

        self.gross_power_efficiency = gross_power_efficiency
        self.storage_medium = storage_medium
        self.grey_energy_ratio = grey_energy_ratio

        self.mean_yearly_sun_power = mean_sun_pw
        self.load_factor = load_factor

        self.total_mirror_area = None

        self.update()

    def update(self):
        self.storage_efficiency = heat_storage(self.storage_medium)

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

        self.material.concrete = 760.e3 * self.nominal_peak_power*1e-6
        self.material.steel = 186.e3 * self.nominal_peak_power*1e-6
        self.material.iron = 4.5e3 * self.nominal_peak_power*1e-6
        self.material.copper = 2.72e3 * self.nominal_peak_power*1e-6
        self.material.plastic = 3.3e3 * self.nominal_peak_power*1e-6
        self.material.fiber_glass = 0.03e3 * self.nominal_peak_power*1e-6
        self.material.glass = 112.e3 * self.nominal_peak_power*1e-6
        self.material.oil = 2.82e3 * self.nominal_peak_power*1e-6

        heat_storage(self.storage_medium, self.storage_capacity, self.material)



class PvPowerPlant(PowerPlant):

    def __init__(self, n_panel, mean_sun_pw,
                 ref_sun_power = 1000.,
                 reg_factor = 0.,
                 panel_area = 2.,
                 ground_ratio = 2.6,
                 life_time = 25.,
                 gross_power_efficiency = 0.15,
                 storage_medium = "flow_battery",
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
        self.storage_medium = storage_medium
        self.grey_energy_ratio = grey_energy_ratio

        self.mean_yearly_sun_power = mean_sun_pw
        self.load_factor = load_factor

        self.total_panel_area = None

        self.update()

    def update(self):
        self.ref_yearly_sun_power = self.ref_sun_power * (self.mean_yearly_sun_power/250.)

        self.storage_efficiency = elec_storage(self.storage_medium)

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

        self.material.steel = 16.5 * self.total_panel_area
        self.material.aluminium = 0.6 * self.total_panel_area
        self.material.copper = 0.011 * self.total_panel_area
        self.material.plastic = 0.12 * self.total_panel_area
        self.material.silicon = 2.15 * self.total_panel_area
        self.material.glass = 8.5 * self.total_panel_area

        elec_storage(self.storage_medium, self.storage_capacity, self.material)



class EolPowerPlant(PowerPlant):

    def __init__(self, n_rotor, location,
                 reg_factor = 0.,
                 rotor_width = 100.,
                 rotor_peak_power = 2.5e6,
                 load_factor = None,
                 storage_medium = "flow_battery",
                 life_time = 25.):
        super(EolPowerPlant, self).__init__()

        self.type = "Wind turbine " + location
        self.location = location
        self.regulation_factor = reg_factor
        self.n_rotor = n_rotor
        self.rotor_width = rotor_width
        self.rotor_peak_power = rotor_peak_power
        self.load_factor = load_factor
        self.storage_medium = storage_medium
        self.storage_efficiency = elec_storage(storage_medium)
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
        self.material.aluminium = {"onshore":1600., "offshore":2500.}.get(self.location) * self.nominal_peak_power * 1e-6
        self.material.copper = {"onshore":400., "offshore":400.}.get(self.location) * self.nominal_peak_power * 1e-6
        self.material.lead = {"onshore":0., "offshore":4000.}.get(self.location) * self.nominal_peak_power * 1e-6
        self.material.plastic = {"onshore":2200., "offshore":2900.}.get(self.location) * self.nominal_peak_power * 1e-6
        self.material.silicon = {"onshore":0., "offshore":0.}.get(self.location) * self.nominal_peak_power * 1e-6
        self.material.glass = {"onshore":0., "offshore":0.}.get(self.location) * self.nominal_peak_power * 1e-6

        elec_storage(self.storage_medium, self.storage_capacity, self.material)



class NuclearPowerPlant(PowerPlant):

    def __init__(self, n_unit,
                 unit_peak_power = 1.0e9,
                 load_factor = 0.75,
                 life_time = 50.,
                 grey_energy_ratio = 0.2,
                 unit_footprint = 0.25e6):
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



class MixEnergetic(object):

    def __init__(self, sun_pw=250., mix={}):
        self.total_footprint = 0.
        self.total_grey_energy = 0.
        self_total_material_index = None
        self_total_material_grey_energy = None

        self.pv_unit = 1.e6     # reference number of pv panels
        self.pv = PvPowerPlant(self.pv_unit, sun_pw, reg_factor=0.)
        self.pv_power = self.pv.nominal_peak_power    # reference peak power of one plant

        self.csp_cp_unit = 1.e3     # reference number of mirror lines per plant
        self.csp_cp = CspPowerPlant(self.csp_cp_unit, sun_pw, reg_factor=0.)
        self.csp_cp_power = self.csp_cp.nominal_peak_power    # reference peak power of one plant

        rppw = 2.5e6    # peak power of each rotor
        loadf = 0.25    # load factor
        self.eol_onsh_unit = 20.     # reference number of rotors per farm
        self.eol_onsh = EolPowerPlant(self.eol_onsh_unit, "onshore", rotor_peak_power=rppw, load_factor=loadf)
        self.eol_onsh_power = self.eol_onsh.nominal_peak_power    # reference peak power of one plant

        rppw = 3.5e6    # peak power of each rotor
        loadf = 0.50    # load factor
        self.eol_offsh_unit = 2.e2     # reference number of rotors per farm
        self.eol_offsh = EolPowerPlant(self.eol_offsh_unit, "offshore", rotor_peak_power=rppw, load_factor=loadf)
        self.eol_offsh_power = self.eol_offsh.nominal_peak_power    # reference peak power of one plant

        uppw = 1.0e9    # peak power of each core
        loadf = 0.75    # load factor
        self.nuclear_unit = 4.     # reference number of cores per plant
        self.nuclear = NuclearPowerPlant(self.nuclear_unit, unit_peak_power = uppw, load_factor=loadf)
        self.nuclear_power = self.nuclear.nominal_peak_power    # reference peak power of one plant

        self.update(mix)

    def __iter__(self):
        public = [value for value in self.__dict__.values() if issubclass(type(value),PowerPlant)]
        return iter(public)

    def update(self, mix):
        self.pv_plant = mix["pv"] / self.pv_power
        self.pv.n_panel = mix["pv"] / (self.pv.nominal_peak_power/self.pv.n_panel)
        self.pv.update()

        self.csp_cp_plant = mix["csp_cp"] / self.csp_cp_power
        self.csp_cp.n_mirror = mix["csp_cp"] / (self.csp_cp.nominal_peak_power/self.csp_cp.n_mirror)
        self.csp_cp.update()

        self.eol_onsh_plant = mix["eol_onsh"] / self.eol_onsh_power
        self.eol_onsh.n_rotor = mix["eol_onsh"] / (self.eol_onsh.nominal_peak_power/self.eol_onsh.n_rotor)
        self.eol_onsh.update()

        self.eol_offsh_plant = mix["eol_offsh"] / self.eol_offsh_power
        self.eol_offsh.n_rotor = mix["eol_offsh"] / (self.eol_offsh.nominal_peak_power/self.eol_offsh.n_rotor)
        self.eol_offsh.update()

        self.nuclear_plant = mix["nuclear"] / self.nuclear_power
        self.nuclear.n_unit = mix["nuclear"] / (self.nuclear.nominal_peak_power/self.nuclear.n_unit)
        self.nuclear.update()

        for plant in self:
            self.total_footprint += plant.total_footprint
            self.total_grey_energy += plant.total_grey_enrg

    def print(self):
        print("Number of PV plant = ",int(np.round(self.pv_plant)))
        print("Footprint of ALL PV plant = ","%8.1f" % (self.pv.total_footprint*1e-6)," km2")
        print("")
        print("Number of CSP plant = ",int(np.round(self.csp_cp_plant)))
        print("Footprint of ALL CSP plant = ","%8.1f" % (self.csp_cp.total_footprint*1e-6)," km2")
        print("")
        print("Number of WTP onshore = ",int(np.round(self.eol_onsh_plant)))
        print("Footprint of ALL WTP onshore = ","%8.1f" % (self.eol_onsh.total_footprint*1e-6)," km2")
        print("")
        print("Number of WTP offshore = ",int(np.round(self.eol_offsh_plant)))
        print("Footprint of ALL WTP offshore = ","%8.1f" % (self.eol_offsh.total_footprint*1e-6)," km2")
        print("")
        print("Number of nuclear plant = ",int(np.round(self.nuclear_plant)))
        print("Footprint of ALL nuclear plant = ","%8.0f" % (self.nuclear.total_footprint*1e-6)," km2")
        print("")
        print("Total footprint of the mix = ","%8.0f" % (self.total_footprint*1e-6)," km2")
        print("Total grey energy = ","%8.3f" % (self.total_grey_energy*1e-18)," EWh (1e18 Wh)")




def max_solar_power(latt,long,pamb,day,gmt):
    """Compute max solar radiative power from location and time on Earth

    :param latt: Lattitude in radians
    :param long: Longitude in radians
    :param day: Day of the year, from 1 to 365
    :param gmt: GMT time in the day, from 0. to 24.
    :return:
    """
    delta = unit.rad_deg(23.45 * np.sin(unit.rad_deg((284.+day)*(360./365.))))
    equ = 0. # output of time equation, neglected here
    solar_time = gmt + (unit.deg_rad(long)*(4./60.)) - equ
    eta = unit.rad_deg((360./24.)*(solar_time - 12.))
    sin_a = np.sin(latt) * np.sin(delta) + np.cos(latt)*np.cos(delta)*np.cos(eta)
    alpha = np.arcsin(sin_a)    # Sun elevation
    ref_solar_pw = 1367.        # Reference solar power
    pw_out = ref_solar_pw * (1. + 0.034*np.cos(unit.rad_deg(day*(360./365.))))
    m0 = np.sqrt(1229. + (614.*sin_a)**2) - 614.*sin_a      # Absorbtion coefficient
    p0 = 101325.                # Sea level reference pressure
    m = m0*(pamb/p0)            # Influence of altitude on the absorbtion coefficient
    tau = 0.6                   # Transmission coefficient
    pw_direct = pw_out * tau**m * sin_a
    pw_diffus = pw_out * (0.271 - 0.294*tau**m) * sin_a
    if (alpha>unit.rad_deg(3.)):
        pw_total = pw_direct + pw_diffus
    else:
        pw_total = 0.
    return pw_total



st1 = CspPowerPlant(7500., 250., reg_factor=0.51)
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
day = 31+29+31+28
gmt = 17.

pw = max_solar_power(latt,long,pamb,day,gmt)

print("")
print("Solar power = ",pw)


mix = {"pv":5.e9, "csp_cp":5.e9, "eol_onsh":20.e9, "eol_offsh":30.e9, "nuclear":40.e9}

mix = MixEnergetic(sun_pw=250., mix=mix)

print("")
mix.print()
