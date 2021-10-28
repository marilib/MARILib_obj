#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Camille LESCUYER, DRUOT Thierry
"""

import numpy as np

from marilib.utils import unit

# TODO : comlete dictionnary
# TODO : add data for heat and elec storage


one_hour = 3600.
one_day = one_hour * 24.
one_year = one_day * 365.


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
        self.quinone = 0.

    def grey_energy(self):
        """Embodied energy of materials  (in J/kg)
        Source:
        G.P.Hammond and C.I.Jones (2006) Embodied energy and carbon footprint database, Department of Mechanical Engineering, University of Bath, United Kingdom
        Embodied energy in thermal energy storage (TES) systems for high temperature applications
        Given data are in standard units (J/kg)
        """
        data = {"concrete":1.11e6,
                "iron":25.0e6,
                "steel":20.1e6,
                "aluminium":155.0e6,
                "copper":42.0e6,
                "lead":25.2e6,
                "silicon":15.0e6,
                "plastic":75.0e6,
                "fiber_glass":28.0e6,
                "glass":15.0e6,
                "oil":50.0e6,
                "Na2CO3":16.0e6,
                "KNO3":16.0e6,
                "quinone":10.0e6
                }
        grey_energy = 0.
        for m in data.keys():
            grey_energy += data[m] * getattr(self, m, 0.)
        return grey_energy

    def Scarcity(self, type):
        """World wide stocks of some minerals (in kg)
        Source : Wikipedia
        """
        data = {"sand": 192.e18,
                "iron": 87.e12,
                "aluminium": 28.e12,
                "copper": 630.e9
                }
        return data.get(type,None)


class PowerPlant(object):

    def __init__(self):
        self.total_footprint = None
        self.total_grey_enrg = None

        self.nominal_peak_power = None
        self.nominal_mean_power = None
        self.mean_dayly_energy = None

        self.production_time = None
        self.retrieval_time = None
        self.regulated_power = None
        self.storage_capacity = None
        self.production_power_efficiency = None
        self.potential_energy_default = None

        self.gross_yearly_enrg = None
        self.net_yearly_enrg = None

        self.marginal_efficiency = None
        self.net_power_efficiency = None
        self.er_o_ei = None
        self.enrg_pay_back_time = None

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
        print("Production time = ", "%8.1f" % unit.h_s(self.production_time), " h")
        print("Retrieval time = ", "%8.1f" % unit.h_s(self.retrieval_time), " h")
        print("Storage capacity = ", "%8.1f" % unit.MWh_J(self.storage_capacity), " MWh")
        print("Yearly gross production = ", "%8.1f" % unit.GWh_J(self.gross_yearly_enrg), " GWh")
        print("Yearly net production = ", "%8.1f" % unit.GWh_J(self.net_yearly_enrg), " GWh")
        print("Marginal efficiency = ", "%8.3f" % self.marginal_efficiency)
        print("Net power efficiency = ", "%8.3f" % self.net_power_efficiency)
        print("Total grey energy = ", "%8.1f" % unit.GWh_J(self.total_grey_enrg), " GWh")
        print("Total footprint = ", "%8.1f" % unit.km2_m2(self.total_footprint), " km2")
        print("Energy returned over energy invested = ", "%8.1f" % self.er_o_ei)
        print("Energy pay back time = ", "%8.1f" % self.enrg_pay_back_time)
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
        """Provide storage efficiency or add materials required for storage

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

        # Materials analysis
        if (storage_type=="electrolysis"):
            print("Materials for energy storage using electrolysis not implemented")
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

        # Materials analysis
        if (storage_type=="molten_salt"):
            material.Na2CO3 = 45.6e3 * unit.MWh_J(energy_capacity)
            material.KNO3 = 30.4e3 * unit.MWh_J(energy_capacity)

            material.concrete += 10.1e3 * unit.MWh_J(energy_capacity)
            material.steel += 4.6e3 * unit.MWh_J(energy_capacity)
            material.fiber_glass += 0.16e3 * unit.MWh_J(energy_capacity)
        elif (storage_type=="concrete"):
            print("Materials for thermal energy storage using concrete not implemented")



class PvPowerPlant(PowerPlant):

    def __init__(self, n_panel,
                 ref_sun_pw = 1000.,
                 load_factor = 0.14,
                 panel_area = 2.,
                 ground_ratio = 2.6,
                 life_time = 25.,
                 gross_pw_eff = 0.15,
                 specific_grey_enrg = unit.J_kWh(350.),
                 reg_factor = 0.,
                 storage = "flow_battery"):
        super(PvPowerPlant, self).__init__()

        self.type = "Photovoltaïc"
        self.ref_sun_power = ref_sun_pw
        self.load_factor = load_factor

        self.n_panel = n_panel
        self.panel_area = panel_area
        self.total_panel_area = panel_area*n_panel
        self.ground_ratio = ground_ratio
        self.life_time = life_time

        self.regulation_factor = reg_factor
        self.storage_medium = storage

        self.gross_power_efficiency = gross_pw_eff
        self.specific_grey_enrg = specific_grey_enrg    # J/m2, Energy per square meter installed, at system level
        self.total_grey_enrg = specific_grey_enrg*panel_area*n_panel

        self.update()

    def get_n_unit(self):
        return self.n_panel

    def set_n_unit(self, n_unit):
        self.n_panel = n_unit

    def update(self):
        self.storage_efficiency = self.elec_storage(self.storage_medium)

        self.total_panel_area = self.panel_area * self.n_panel
        self.total_footprint = self.total_panel_area * self.ground_ratio

        self.nominal_peak_power = self.ref_sun_power * self.total_panel_area * self.gross_power_efficiency
        self.nominal_mean_power = self.nominal_peak_power * self.load_factor
        self.mean_daily_energy = self.nominal_mean_power * one_day

        self.production_time = one_day * (self.load_factor + (1. - self.load_factor)*self.regulation_factor)
        self.retrieval_time = one_day * (1. - self.load_factor)*self.regulation_factor
        self.regulated_power = self.nominal_peak_power / (1. + self.regulation_factor*(1.-self.load_factor)/(self.load_factor*self.storage_efficiency))
        self.storage_capacity = (self.nominal_peak_power - self.regulated_power) * self.load_factor * self.storage_efficiency * one_day
        self.production_power_efficiency = self.regulated_power * self.production_time / self.mean_daily_energy
        self.potential_energy_default = self.nominal_mean_power * (one_day - self.production_time)

        self.gross_yearly_enrg = self.regulated_power * self.production_time * 365.
        self.total_grey_enrg = self.specific_grey_enrg * self.total_panel_area
        self.net_yearly_enrg = self.gross_yearly_enrg - self.total_grey_enrg / self.life_time

        self.marginal_efficiency = self.net_yearly_enrg / (self.nominal_mean_power * one_year)
        self.net_power_efficiency = self.gross_power_efficiency * self.marginal_efficiency
        self.er_o_ei = self.net_yearly_enrg / (self.total_grey_enrg/self.life_time)
        self.enrg_pay_back_time = self.total_grey_enrg / self.net_yearly_enrg


        self.material.steel = 16.5 * self.total_panel_area
        self.material.aluminium = 0.6 * self.total_panel_area
        self.material.copper = 0.011 * self.total_panel_area
        self.material.plastic = 0.12 * self.total_panel_area
        self.material.silicon = 2.15 * self.total_panel_area
        self.material.glass = 8.5 * self.total_panel_area

        self.elec_storage(self.storage_medium, self.storage_capacity, self.material)

        self.material_grey_enrg = self.material.grey_energy()



class EolPowerPlant(PowerPlant):

    def __init__(self, location, n_rotor,
                 load_factor = None,
                 rotor_width = None,
                 rotor_pk_pw = 2.5e6,
                 life_time = 25.,
                 gross_power_eff = 0.35,
                 specific_grey_enrg = 10e6,
                 reg_factor = 0.,
                 storage_medium = "flow_battery"):
        super(EolPowerPlant, self).__init__()

        self.type = location+" Wind turbine"
        self.location = location
        self.regulation_factor = reg_factor
        self.n_rotor = n_rotor
        self.rotor_width = rotor_width
        self.rotor_peak_power = rotor_pk_pw
        self.load_factor = load_factor
        self.gross_power_efficiency = gross_power_eff
        self.specific_grey_enrg = specific_grey_enrg    # J/W, Energy per installed rated power (peak power here), system level
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

        self.nominal_peak_power = self.rotor_peak_power * self.n_rotor
        self.nominal_mean_power = self.nominal_peak_power * self.load_factor
        self.mean_daily_energy = self.nominal_mean_power * one_day

        self.storage_efficiency = self.elec_storage(self.storage_medium)
        self.production_time = one_day * (self.load_factor + (1. - self.load_factor)*self.regulation_factor)
        self.retrieval_time = one_day * (1. - self.load_factor)*self.regulation_factor
        self.regulated_power = self.nominal_peak_power / (1. + self.regulation_factor*(1.-self.load_factor)/(self.load_factor*self.storage_efficiency))
        self.storage_capacity = (self.nominal_peak_power - self.regulated_power) * self.load_factor * self.storage_efficiency * one_day
        self.production_power_efficiency = self.regulated_power * self.production_time / self.mean_daily_energy
        self.potential_energy_default = self.nominal_mean_power * (one_day - self.production_time)

        self.rotor_grey_enrg = {"onshore": self.specific_grey_enrg * self.rotor_peak_power,
                                "offshore": 1.8 * self.specific_grey_enrg * self.rotor_peak_power
                                }.get(self.location, "Error, location is unknown")

        self.gross_yearly_enrg = self.regulated_power * self.production_time * 365.
        self.total_grey_enrg = self.rotor_grey_enrg * self.n_rotor
        self.net_yearly_enrg = self.gross_yearly_enrg - self.total_grey_enrg / self.life_time

        self.marginal_efficiency = self.net_yearly_enrg / (self.nominal_mean_power * one_year)
        self.net_power_efficiency =  self.gross_power_efficiency * self.marginal_efficiency
        self.er_o_ei = self.net_yearly_enrg / (self.total_grey_enrg/self.life_time)
        self.enrg_pay_back_time = self.total_grey_enrg / self.net_yearly_enrg

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






if __name__ == "__main__":

    # ======================================================================================================
    # Identify existing plants
    # ------------------------------------------------------------------------------------------------------

    pv1 = PvPowerPlant(1e6, reg_factor=0.0)
    pv1.print("Cestas")



