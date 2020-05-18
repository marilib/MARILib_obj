#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np

from marilib.utils import unit
from marilib.power_plant.tool import Material

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
    "rotor_grey_enrg": {"unit":"GWh", "mag":1e2, "txt":"Embodied energy for one rotor"},
    "er_o_ei": {"unit":"no_dim", "mag":1e0, "txt":"Energy Returned over Energy Invested (ERoEI)"},
    "mix_peak_power": {"unit":"GW", "mag":1e2, "txt":"Total peak power of the mix"},
    "mix_mean_power": {"unit":"GW", "mag":1e2, "txt":"Total mean power of the mix"},
    "mix_er_o_ei": {"unit":"no_dim", "mag":1e0, "txt":"Total Energy Returned over Energy Invested of the mix (ERoEI)"},
    "mix_footprint": {"unit":"km2", "mag":1e5, "txt":"Total footprint of the mix"},
    "mix_grey_energy": {"unit":"EWh", "mag":1e1, "txt":"Total embodied energy of the mix"},
    "mix_material_grey_energy": {"unit":"EWh", "mag":1e0, "txt":"Total material embodied energy ofthe mix"},
    "mix_mean_dayly_energy": {"unit":"TWh", "mag":1e1, "txt":"Total mean dayly production of the mix"},
    "mix_potential_energy_default": {"unit":"GWh", "mag":1e1, "txt":"Potential dayly regulation default of the mix"},
    "mix_potential_default_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Potential energy default over mean dayly production"}
}

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
        self.potential_energy_default = None

        self.gross_yearly_enrg = None
        self.net_yearly_enrg = None

        self.marginal_efficiency = None
        self.net_power_efficiency = None
        self.er_o_ei = None

        self.material = Material()
        self.material_grey_enrg = None

    def get_n_unit(self):
        raise NotImplementedError

    def set_n_unit(self, n_unit):
        raise NotImplementedError

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
        print("Energy returned over energy invested = ", "%8.1f" % self.er_o_ei)
        print("Total material embodied energy = ", "%8.1f" % unit.GWh_J(self.material_grey_enrg), " GWh")
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
        if (self.material.quinone > 0.):      print("     quinone = ", "%8.0f" % (self.material.quinone*1.e-3), " t")

    def elec_storage(self, storage_type, *kwargs):
        """Provide storage efficiency or add materials required for heat storage

        :param storage_type: "electrolysis", "flow_battery"
        :param energy_capacity:
        :param material:
        :return:
        """
        if (storage_type=="electrolysis"):
            storage_efficiency = 0.70
        elif (storage_type=="flow_battery"):
            storage_efficiency = 0.85
        else:
            raise Exception("Type of electricity storage is unknown")

        if (len(kwargs)==0):
            return storage_efficiency
        else:
            energy_capacity = kwargs[0]
            material = kwargs[1]

        if (storage_type=="electrolysis"):
            print("Electric energy storage using electrolysis not implemented")
        elif (storage_type=="flow_battery"):
            material.quinone = energy_capacity / unit.J_Wh(50.)   # 50 Wh/kg

            material.concrete += 0.15 * material.quinone
            material.steel += 0.01 * material.quinone
            material.fiber_glass += 0.01 * material.quinone

    def heat_storage(self, storage_type, *kwargs):
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
            material.Na2CO3 = 45.6e3 * unit.MWh_J(energy_capacity)
            material.KNO3 = 30.4e3 * unit.MWh_J(energy_capacity)

            material.concrete += 10.1e3 * unit.MWh_J(energy_capacity)
            material.steel += 4.6e3 * unit.MWh_J(energy_capacity)
            material.fiber_glass += 0.16e3 * unit.MWh_J(energy_capacity)
        elif (storage_type=="concrete"):
            print("Thermal energy storage using concrete not implemented")



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

        self.type = "Cylindroparabolic solar concentrated"
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

    def get_n_unit(self):
        return self.n_mirror

    def set_n_unit(self, n_unit):
        self.n_mirror = n_unit

    def update(self):
        self.storage_efficiency = self.heat_storage(self.storage_medium)

        self.total_mirror_area = self.mirror_area * self.n_mirror
        self.total_footprint = self.total_mirror_area * self.ground_ratio

        self.nominal_peak_power = self.mean_yearly_sun_power * self.total_mirror_area * self.gross_power_efficiency
        self.nominal_mean_power = self.nominal_peak_power * self.load_factor
        self.mean_daily_energy = self.nominal_mean_power * one_day

        self.regulation_time = one_day * (self.load_factor + (1. - self.load_factor)*self.regulation_factor)
        self.retrieval_time = one_day * (1. - self.load_factor)*self.regulation_factor
        self.regulated_power = self.nominal_peak_power / (1. + self.regulation_factor*(1.-self.load_factor)/(self.load_factor*self.storage_efficiency))
        self.storage_capacity = (self.nominal_peak_power - self.regulated_power) * self.load_factor * self.storage_efficiency * one_day
        self.regulation_power_efficiency = self.regulated_power * self.regulation_time / self.mean_daily_energy
        self.potential_energy_default = self.nominal_mean_power * (one_day - self.regulation_time)

        self.marginal_efficiency = self.regulation_power_efficiency * (1.-self.grey_energy_ratio)
        self.net_power_efficiency = self.gross_power_efficiency * self.marginal_efficiency

        self.gross_yearly_enrg = self.regulated_power * self.regulation_time * 365.
        self.net_yearly_enrg = self.gross_yearly_enrg - self.gross_yearly_enrg * self.grey_energy_ratio

        self.total_grey_enrg = self.gross_yearly_enrg * self.grey_energy_ratio * self.life_time
        self.er_o_ei = self.net_yearly_enrg / (self.gross_yearly_enrg * self.grey_energy_ratio)

        self.material.concrete = 760.e3 * unit.MW_W(self.nominal_peak_power)
        self.material.steel = 186.e3 * unit.MW_W(self.nominal_peak_power)
        self.material.iron = 4.5e3 * unit.MW_W(self.nominal_peak_power)
        self.material.copper = 2.72e3 * unit.MW_W(self.nominal_peak_power)
        self.material.plastic = 3.3e3 * unit.MW_W(self.nominal_peak_power)
        self.material.fiber_glass = 0.03e3 * unit.MW_W(self.nominal_peak_power)
        self.material.glass = 112.e3 * unit.MW_W(self.nominal_peak_power)
        self.material.oil = 2.82e3 * unit.MW_W(self.nominal_peak_power)

        self.heat_storage(self.storage_medium, self.storage_capacity, self.material)

        self.material_grey_enrg = self.material.grey_energy()



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

        self.type = "Photovoltaïc"
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

    def get_n_unit(self):
        return self.n_panel

    def set_n_unit(self, n_unit):
        self.n_panel = n_unit

    def update(self):
        self.ref_yearly_sun_power = self.ref_sun_power * (self.mean_yearly_sun_power/250.)

        self.storage_efficiency = self.elec_storage(self.storage_medium)

        self.total_panel_area = self.panel_area * self.n_panel
        self.total_footprint = self.total_panel_area * self.ground_ratio

        self.nominal_peak_power = self.ref_yearly_sun_power * self.total_panel_area * self.gross_power_efficiency
        self.nominal_mean_power = self.nominal_peak_power * self.load_factor
        self.mean_daily_energy = self.nominal_mean_power * one_day

        self.regulation_time = one_day * (self.load_factor + (1. - self.load_factor)*self.regulation_factor)
        self.retrieval_time = one_day * (1. - self.load_factor)*self.regulation_factor
        self.regulated_power = self.nominal_peak_power / (1. + self.regulation_factor*(1.-self.load_factor)/(self.load_factor*self.storage_efficiency))
        self.storage_capacity = (self.nominal_peak_power - self.regulated_power) * self.load_factor * self.storage_efficiency * one_day
        self.regulation_power_efficiency = self.regulated_power * self.regulation_time / self.mean_daily_energy
        self.potential_energy_default = self.nominal_mean_power * (one_day - self.regulation_time)

        self.marginal_efficiency = self.regulation_power_efficiency * (1.-self.grey_energy_ratio)
        self.net_power_efficiency = self.gross_power_efficiency * self.marginal_efficiency

        self.gross_yearly_enrg = self.regulated_power * self.regulation_time * 365.
        self.net_yearly_enrg = self.gross_yearly_enrg * (1. - self.grey_energy_ratio)

        self.total_grey_enrg = self.gross_yearly_enrg * self.grey_energy_ratio * self.life_time
        self.er_o_ei = self.net_yearly_enrg / (self.gross_yearly_enrg * self.grey_energy_ratio)

        self.material.steel = 16.5 * self.total_panel_area
        self.material.aluminium = 0.6 * self.total_panel_area
        self.material.copper = 0.011 * self.total_panel_area
        self.material.plastic = 0.12 * self.total_panel_area
        self.material.silicon = 2.15 * self.total_panel_area
        self.material.glass = 8.5 * self.total_panel_area

        self.elec_storage(self.storage_medium, self.storage_capacity, self.material)

        self.material_grey_enrg = self.material.grey_energy()



class EolPowerPlant(PowerPlant):

    def __init__(self, n_rotor, location,
                 reg_factor = 0.,
                 rotor_width = None,
                 rotor_peak_power = 2.5e6,
                 load_factor = None,
                 storage_medium = "flow_battery",
                 life_time = 25.):
        super(EolPowerPlant, self).__init__()

        self.type = location+" Wind turbine"
        self.location = location
        self.regulation_factor = reg_factor
        self.n_rotor = n_rotor
        self.rotor_width = rotor_width
        self.rotor_peak_power = rotor_peak_power
        self.load_factor = load_factor
        self.storage_medium = storage_medium
        self.storage_efficiency = self.elec_storage(storage_medium)
        self.life_time = life_time

        self.rotor_area = None
        self.rotor_footprint = None
        self.rotor_grey_enrg = None

        self.update()

    def get_n_unit(self):
        return self.n_rotor

    def set_n_unit(self, n_unit):
        self.n_rotor = n_unit

    def update(self):
        if (self.rotor_width is None):
            self.rotor_width = 100. + 12.5*(self.rotor_peak_power*1e-6 - 2.5)

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
        self.mean_daily_energy = self.nominal_mean_power * one_day

        self.regulation_time = one_day * (self.load_factor + (1. - self.load_factor)*self.regulation_factor)
        self.retrieval_time = one_day * (1. - self.load_factor)*self.regulation_factor
        self.regulated_power = self.nominal_peak_power / (1. + self.regulation_factor*(1.-self.load_factor)/(self.load_factor*self.storage_efficiency))
        self.storage_capacity = (self.nominal_peak_power - self.regulated_power) * self.load_factor * self.storage_efficiency * one_day
        self.regulation_power_efficiency = self.regulated_power * self.regulation_time / self.mean_daily_energy
        self.potential_energy_default = self.nominal_mean_power * (one_day - self.regulation_time)

        self.gross_yearly_enrg = self.regulated_power * self.regulation_time * 365.
        self.net_yearly_enrg = self.gross_yearly_enrg - self.total_grey_enrg / self.life_time
        self.er_o_ei = self.net_yearly_enrg / (self.total_grey_enrg/self.life_time)

        self.gross_power_efficiency = 0.35
        self.marginal_efficiency = self.net_yearly_enrg / (self.nominal_mean_power * one_year)
        self.net_power_efficiency =  self.gross_power_efficiency * self.marginal_efficiency

        self.material.concrete = {"onshore":320000., "offshore":640000.}.get(self.location) * unit.MW_W(self.nominal_peak_power)
        self.material.steel = {"onshore":61500., "offshore":91500.}.get(self.location) * unit.MW_W(self.nominal_peak_power)
        self.material.aluminium = {"onshore":1600., "offshore":2500.}.get(self.location) * unit.MW_W(self.nominal_peak_power)
        self.material.copper = {"onshore":400., "offshore":400.}.get(self.location) * unit.MW_W(self.nominal_peak_power)
        self.material.lead = {"onshore":0., "offshore":4000.}.get(self.location) * unit.MW_W(self.nominal_peak_power)
        self.material.plastic = {"onshore":2200., "offshore":2900.}.get(self.location) * unit.MW_W(self.nominal_peak_power)
        self.material.silicon = {"onshore":0., "offshore":0.}.get(self.location) * unit.MW_W(self.nominal_peak_power)
        self.material.glass = {"onshore":0., "offshore":0.}.get(self.location) * unit.MW_W(self.nominal_peak_power)

        self.elec_storage(self.storage_medium, self.storage_capacity, self.material)

        self.material_grey_enrg = self.material.grey_energy()



class NuclearPowerPlant(PowerPlant):

    def __init__(self, n_core,
                 core_peak_power = 1.0e9,
                 load_factor = 0.80,
                 life_time = 50.,
                 grey_energy_ratio = 0.2,
                 unit_footprint = 0.25e6):
        super(NuclearPowerPlant, self).__init__()

        self.type = "Nuclear"
        self.n_core = n_core
        self.core_peak_power = core_peak_power
        self.load_factor = load_factor
        self.life_time = life_time
        self.grey_energy_ratio = grey_energy_ratio
        self.unit_footprint = unit_footprint

        self.update()

    def get_n_unit(self):
        return self.n_core

    def set_n_unit(self, n_unit):
        self.n_core = n_unit

    def update(self):
        self.nominal_peak_power = self.core_peak_power * self.n_core
        self.nominal_mean_power = self.nominal_peak_power * self.load_factor
        self.total_footprint = self.unit_footprint * self.n_core

        self.regulation_time = one_day
        self.retrieval_time = 0.
        self.regulated_power = self.nominal_mean_power
        self.storage_capacity = 0.
        self.regulation_power_efficiency = 1.0
        self.potential_energy_default = self.nominal_mean_power * (one_day - self.regulation_time)

        self.mean_daily_energy = self.nominal_mean_power * one_day
        self.gross_yearly_enrg = self.nominal_mean_power * one_year

        self.total_grey_enrg = self.gross_yearly_enrg * self.life_time * self.grey_energy_ratio
        self.net_yearly_enrg = self.gross_yearly_enrg * self.life_time * (1. - self.grey_energy_ratio) / self.life_time
        self.er_o_ei = self.net_yearly_enrg / (self.gross_yearly_enrg * self.grey_energy_ratio)

        self.gross_power_efficiency = 0.37
        self.marginal_efficiency = self.net_yearly_enrg / (self.nominal_mean_power * one_year)
        self.net_power_efficiency =  self.gross_power_efficiency * self.marginal_efficiency

        # ref : for waste storage
        # Life cycle energy and greenhouse gas emissions of nuclear energy:
        # A review
        # Manfred Lenzen
        # *
        # ISA, Centre for Integrated Sustainability Analysis, The University of Sydney, Physics Building A28, Sydney, NSW 2006, Australia
        # Received 13 June 2007; accepted 31 January 2008
        # Available online 8 April 2008
        self.material.concrete = 320.e3 * unit.MW_W(self.nominal_peak_power) \
                                 + (373.e3/177.) * unit.GWh_J(self.mean_daily_energy * self.life_time)   # Power plant + waste storage
        self.material.steel = 60.e3 * unit.MW_W(self.nominal_peak_power)
        self.material.aluminium = 0.14e3 * unit.MW_W(self.nominal_peak_power)
        self.material.copper = 1.51e3 * unit.MW_W(self.nominal_peak_power)
# TODO        self.material.lead = np.nan

        self.material_grey_enrg = self.material.grey_energy()



class EnergyMix(object):

    def __init__(self, tech_mix, power_mix):
        self.tech_mix = tech_mix
        self.power_mix = None
        self.n_unit = [pp.get_n_unit() for pp in self.tech_mix]
        self.plant_power = [pp.nominal_peak_power for pp in self.tech_mix]
        self.n_plant = None

        self.mix_peak_power = 0.
        self.mix_mean_power = 0.
        self.mix_er_o_ei = 0.
        self.mix_footprint = 0.
        self.mix_grey_energy = 0.
        self.mix_material_grey_energy = 0.

        self.mix_mean_daily_energy = 0.
        self.mix_potential_energy_default = 0.
        self.mix_potential_default_ratio = 0.

        self.update(power_mix)

    def update(self, power_mix):
        self.power_mix = power_mix
        self.n_plant = [np.ceil(tpw/upw) for tpw,upw in zip(power_mix, self.plant_power)]

        for j,pw in enumerate(self.power_mix):
            plt = self.tech_mix[j]
            upw = self.plant_power[j] / self.n_unit[j]
            plt.set_n_unit(pw/upw)
            plt.update()

        self.mix_peak_power = 0.
        self.mix_mean_power = 0.
        self.mix_er_o_ei = 0.
        self.mix_footprint = 0.
        self.mix_grey_energy = 0.
        self.mix_material_grey_energy = 0.
        self.mix_mean_daily_energy = 0.
        self.mix_potential_energy_default = 0.
        for plt in self.tech_mix:
            self.mix_peak_power += plt.nominal_peak_power
            self.mix_mean_power += plt.nominal_mean_power
            self.mix_er_o_ei += plt.er_o_ei * plt.nominal_peak_power
            self.mix_footprint += plt.total_footprint
            self.mix_grey_energy += plt.total_grey_enrg
            self.mix_material_grey_energy += plt.material_grey_enrg
            self.mix_mean_daily_energy += plt.mean_daily_energy
            self.mix_potential_energy_default += plt.potential_energy_default

        self.mix_er_o_ei = self.mix_er_o_ei / self.mix_peak_power
        self.mix_potential_default_ratio = self.mix_potential_energy_default / self.mix_mean_daily_energy

    def print(self):
        for j,plt in enumerate(self.tech_mix):
            print("")
            print("Number of "+plt.type+" plants = ","%5.0f" % self.n_plant[j])
            print("Footprint of all these plants = ","%8.1f" % (plt.total_footprint*1e-6)," km2")
            print("Peak power of all these plants = ","%8.3f" % (plt.nominal_peak_power*1e-9)," GWh")
            print("ERoEI of all these plants = ","%8.1f" % plt.er_o_ei)
        print("")
        print("Total mix peak power = ","%8.2f" % (self.mix_peak_power*1e-9)," GW")
        print("Total mix mean power = ","%8.2f" % (self.mix_mean_power*1e-9)," GW")
        print("Total mix ERoEI = ","%8.2f" % (self.mix_er_o_ei))
        print("Total mix footprint = ","%8.0f" % (self.mix_footprint*1e-6)," km2")
        print("Total mix grey energy = ","%8.3f" % (self.mix_grey_energy*1e-18)," EWh (1e18 Wh)")
        print("Total mix potential default ratio = ","%8.3f" % self.mix_potential_default_ratio)

    def param_iso_power(self,param):
        """Compute a set of of power values which sum is exactly total_power

        :param total_power:The total power of the mix
        :param param: Set of input parameters, (n_plant-1) values within 0 <= v <= 1
        :return: aset of power values which sum is exactly total_power
        """
        if len(param)!=len(self.power_mix)-1:
            raise Exception("Number of parameter must be :",len(self.power_mix)-1)
        pw = []
        total_power = self.mix_peak_power
        for i,p in enumerate(param):
            pw.append(total_power*p)
            for q in param[0:i]:
                pw[-1] *= (1.-q)
        pw.append(total_power)
        for q in param:
            pw[-1] *= (1.-q)
        return pw



if __name__ == "__main__":

    # ======================================================================================================
    # Identify existing plants
    # ------------------------------------------------------------------------------------------------------

    st1 = CspPowerPlant(7500., 250., reg_factor=0.51)
    st1.print("Andasol 3")

    pv1 = PvPowerPlant(1e6, 250., reg_factor=0.0)
    pv1.print("Cestas")

    eol1 = EolPowerPlant(240., "onshore", rotor_peak_power=2.5e6, load_factor=0.25)
    eol1.print("Fantanele Cogealav")

    eol2 = EolPowerPlant(175., "offshore", rotor_peak_power=3.5e6, load_factor=0.50)
    eol2.print("London Array")

    atom1 = NuclearPowerPlant(4)
    atom1.print()


    # ======================================================================================================
    # Build a mix
    # ------------------------------------------------------------------------------------------------------

    pv1 = PvPowerPlant(1e6, 250., reg_factor=0.50)

    st1 = CspPowerPlant(1e4, 250., reg_factor=0.50)

    eol1 = EolPowerPlant(20., "onshore")

    eol2 = EolPowerPlant(200., "offshore", rotor_peak_power=3.5e6, load_factor=0.50)

    atom1 = NuclearPowerPlant(4)

    tech_mix =  [atom1,  eol2,  eol1,  st1,  pv1]
    power_mix = [40.e9, 30.e9, 20.e9, 5.e9, 5.e9]

    mix = EnergyMix(tech_mix, power_mix)

    print("")
    mix.print()

    print("")
    pw = mix.param_iso_power([0.4, 0.5, 0.6666, 0.5])
    print([p*1.e-9 for p in pw])
    print(sum(pw)*1.e-9)

    print("--------------------------------------------------")
    pw_mix = mix.param_iso_power([0.5, 0.5, 0.5, 0.5])
    mix.update(pw_mix)
    mix.print()
