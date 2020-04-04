#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import numpy as np

import earth

from aircraft.tool import unit


class Economics():

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.irp = 10.              # 10 years
        self.period = 15.           # 15 years
        self.interest_rate = 0.04   # 4%
        self.labor_cost = 120.      # 120 $/h
        self.utilisation = self.__utilisation__()

        self.engine_price = None
        self.gear_price = None
        self.frame_price = None
        self.fuel_price = 2./unit.liter_usgal(1)   # 2 $/USgal

        self.frame_cost = None
        self.engine_cost = None
        self.cockpit_crew_cost = None
        self.cabin_crew_cost = None
        self.landing_fees = None
        self.navigation_fees = None
        self.catering_cost = None
        self.pax_handling_cost = None
        self.ramp_handling_cost = None

        self.std_op_cost = None
        self.cash_op_cost = None
        self.direct_op_cost = None

    def __utilisation__(self):
        """Number of flights per year
        """
        design_range = self.aircraft.requirement.design_range
        if(design_range <= unit.m_NM(3500.)): utilisation = 1600.
        else:                                 utilisation = 600.
        return utilisation


    def landing_gear_price(self):
        """Typical value
        """
        landing_gear_mass = self.aircraft.airframe.landing_gear.mass
        gear_price = 720. * landing_gear_mass
        return gear_price


    def one_engine_price(self):
        """Regression on catalog prices
        """
        reference_thrust = self.aircraft.airframe.nacelle.reference_thrust
        engine_price = ((2.115e-4*reference_thrust + 78.85)*reference_thrust)
        return engine_price


    def one_airframe_price(self):
        """Regression on catalog prices corrected with engine prices
        """
        mwe = self.aircraft.weight_cg.mwe
        airframe_price = 0.7e3*(9e4 + 1.15*mwe - 1.8e9/(2e4 + mwe**0.94))
        return airframe_price


    def operating_cost_analysis(self):
        """Computes Cash and Direct Operating Costs per flight (based on AAE 451 Spring 2004)
        """
        n_pax_ref = self.aircraft.requirement.n_pax_ref

        energy_source = self.aircraft.arrangement.energy_source
        power_architecture = self.aircraft.arrangement.power_architecture
        nacelle_mass = self.aircraft.airframe.nacelle.mass

        reference_thrust = self.aircraft.airframe.nacelle.reference_thrust
        n_engine = self.aircraft.airframe.nacelle.n_engine

        mtow = self.aircraft.weight_cg.mtow
        mwe = self.aircraft.weight_cg.mwe

        cost_range = self.aircraft.performance.mission.cost.range
        fuel_block = self.aircraft.performance.mission.cost.fuel_block
        time_block = self.aircraft.performance.mission.cost.time_block

        fuel_density = earth.fuel_density(energy_source)

        # Cash Operating Cost
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        self.fuel_cost =   (fuel_block*(self.fuel_price*1.e3)/fuel_density)
#        eco.elec_cost =  cost_mission.req_battery_mass * propulsion.battery_energy_density * eco.elec_price

        b_h = time_block/3600.
        t_t = b_h + 0.25
        w_f = (10000. + mwe - nacelle_mass)*1.e-5

        labor_frame = ((1.26+1.774*w_f-0.1071*w_f**2)*t_t + (1.614+0.7227*w_f+0.1204*w_f**2))*self.labor_cost
        matrl_frame = (12.39+29.8*w_f+0.1806*w_f**2)*t_t + (15.20+97.330*w_f-2.8620*w_f**2)
        self.frame_cost = labor_frame + matrl_frame

        t_h = 0.05*((reference_thrust)/4.4482198)*1e-4

        labor_engine = n_engine*(0.645*t_t+t_h*(0.566*t_t+0.434))*self.labor_cost
        matrl_engine = n_engine*(25.*t_t+t_h*(0.62*t_t+0.38))

        if (power_architecture=="pte1"):
            pass
#            rear_engine_cost = aircraft.rear_electric_nacelle.mass*eco.rear_nacelle_cost
        else:
            rear_engine_cost = 0.

        self.engine_cost = labor_engine + matrl_engine #+ rear_engine_cost

        w_g = mtow*1e-3

        self.cockpit_crew_cost = b_h*2*(440-0.532*w_g)
        self.cabin_crew_cost = b_h*np.ceil(n_pax_ref/50.)*self.labor_cost
        self.landing_fees = 8.66*(mtow*1e-3)
        self.navigation_fees = 57.*(cost_range/185200.)*np.sqrt((mtow/1000.)/50.)
        self.catering_cost = 3.07 * n_pax_ref
        self.pax_handling_cost = 2. * n_pax_ref
        self.ramp_handling_cost = 8.70 * n_pax_ref
        self.std_op_cost = self.fuel_cost + self.frame_cost + self.engine_cost + self.cockpit_crew_cost + self.landing_fees + self.navigation_fees #+ self.elec_cost
        self.cash_op_cost = self.std_op_cost + self.cabin_crew_cost + self.catering_cost + self.pax_handling_cost + self.ramp_handling_cost

        # DirectOperating Cost
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        self.engine_price = self.one_engine_price()
        self.gear_price = self.landing_gear_price()
        self.frame_price = self.one_airframe_price()

#        battery_price = eco.battery_mass_price*cost_mission.req_battery_mass

        self.aircraft_price = self.frame_price + self.engine_price * n_engine + self.gear_price #+ battery_price
        self.total_investment = self.frame_price * 1.06 + n_engine * self.engine_price * 1.025
        self.interest = (self.total_investment/(self.utilisation*self.period)) * (self.irp * 0.04 * (((1. + self.interest_rate)**self.irp)/((1. + self.interest_rate)**self.irp - 1.)) - 1.)
        self.insurance = 0.0035 * self.aircraft_price/self.utilisation
        self.depreciation = 0.99 * (self.total_investment / (self.utilisation * self.period))     # Depreciation
        self.direct_op_cost = self.cash_op_cost + self.interest + self.depreciation + self.insurance

        return


class Environment():

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.CO2_metric = None
        self.CO2_index = earth.emission_index("CO2"),
        self.H2O_index = earth.emission_index("H2O"),
        self.SO2_index = earth.emission_index("SO2"),
        self.NOx_index = earth.emission_index("NOx"),
        self.CO_index = earth.emission_index("CO"),
        self.HC_index = earth.emission_index("HC"),
        self.sulfuric_acid_index = earth.emission_index("sulfuric_acid"),
        self.nitrous_acid_index = earth.emission_index("nitrous_acid"),
        self.nitric_acid_index = earth.emission_index("nitric_acid"),
        self.soot_index = earth.emission_index("soot")

    def fuel_efficiency_metric(self):
        """
        Fuel efficiency metric (CO2 metric)
        """
        mtow = self.aircraft.weight_cg.mtow
        rgf = self.aircraft.airframe.cabin.projected_area      # Reference Geometric Factor (Pressurized floor area)
        disa = self.aircraft.requirement.cruise_disa
        mach = self.aircraft.requirement.cruise_mach    # take cruise mach instead of Maxi Range because SFC is constant

        high_weight = 0.92*mtow
        low_weight = 0.45*mtow + 0.63*mtow**0.924
        medium_weight = 0.5*(high_weight+low_weight)

        # WARNING : Maximum SAR altitude or speed may be lowered by propulsion ceilings
        #-----------------------------------------------------------------------------------------------------------
        altp_sar_max_hw,sar_max_hw,_,_,_,_,_,_ = self.aircraft.performance.max_sar(high_weight,mach,disa)
        altp_sar_max_hw,hw_ceiling = self.check_ceiling(high_weight,altp_sar_max_hw,mach,disa)
        if(hw_ceiling<0.):
            lower_mach = mach - 0.03
            altp_sar_max_hw,sar_max_hw,_,_,_,_,_,_ = self.aircraft.performance.max_sar(high_weight,lower_mach,disa)
            altp_sar_max_hw,hw_ceiling = self.check_ceiling(high_weight,altp_sar_max_hw,lower_mach,disa)
            sar_max_hw,_,_,_,_,_,_ = self.aircraft.performance.mission.eval_sar(altp_sar_max_hw,high_weight,lower_mach,disa)
        else:
            sar_max_hw,_,_,_,_,_,_ = self.aircraft.performance.mission.eval_sar(altp_sar_max_hw,high_weight,mach,disa)

        altp_sar_max_mw,sar_max_mw,_,_,_,_,_,_ = self.aircraft.performance.max_sar(medium_weight,mach,disa)
        altp_sar_max_mw,mw_ceiling = self.check_ceiling(medium_weight,altp_sar_max_mw,mach,disa)
        if(mw_ceiling<0.):
            lower_mach = mach - 0.03
            altp_sar_max_mw,sar_max_mw,_,_,_,_,_,_ = self.aircraft.performance.max_sar(medium_weight,lower_mach,disa)
            altp_sar_max_mw,mw_ceiling = self.check_ceiling(medium_weight,altp_sar_max_mw,lower_mach,disa)
            sar_max_mw,_,_,_,_,_,_ = self.aircraft.performance.mission.eval_sar(altp_sar_max_mw,medium_weight,lower_mach,disa)
        else:
            sar_max_mw,_,_,_,_,_,_ = self.aircraft.performance.mission.eval_sar(altp_sar_max_mw,medium_weight,mach,disa)

        altp_sar_max_lw,sar_max_lw,_,_,_,_,_,_ = self.aircraft.performance.max_sar(low_weight,mach,disa)
        altp_sar_max_lw,lw_ceiling = self.check_ceiling(low_weight,altp_sar_max_lw,mach,disa)
        if(lw_ceiling<0.):
            lower_mach = mach - 0.03
            altp_sar_max_lw,sar_max_lw,_,_,_,_,_,_ = self.aircraft.performance.max_sar(low_weight,lower_mach,disa)
            altp_sar_max_lw,lw_ceiling = self.check_ceiling(low_weight,altp_sar_max_lw,lower_mach,disa)
            sar_max_lw,_,_,_,_,_,_ = self.aircraft.performance.mission.eval_sar(altp_sar_max_lw,low_weight,lower_mach,disa)
        else:
            sar_max_lw,_,_,_,_,_,_ = self.aircraft.performance.mission.eval_sar(altp_sar_max_lw,low_weight,mach,disa)

        self.CO2_metric = (1./rgf**0.24)*(1./sar_max_hw + 1./sar_max_mw + 1./sar_max_lw)/3.        # kg/m/m2
        return


    def check_ceiling(self,mass,altp_ini,mach,disa):
        """
        Check reachable altitude
        """
        vz_req_mcl = self.aircraft.performance.mcl_ceiling.vz_req
        vz_req_mcr = self.aircraft.performance.mcr_ceiling.vz_req

        isomach = "mach"
        nei = 0

        altp = altp_ini
        ceiling = 0

        slope,vz_clb = self.aircraft.performance.air_path(nei,altp_ini,disa,isomach,mach,mass,"MCL")

        if(vz_clb<vz_req_mcl):
            altp, rei = self.aircraft.performance.propulsion_ceiling(altp_ini,nei,vz_req_mcl,disa,isomach,mach,mass,"MCL")
            if(rei==1):
                ceiling = 1
            else:
                ceiling = -1

        [slope,vz_crz] = self.aircraft.performance.air_path(nei,altp_ini,disa,isomach,mach,mass,"MCR")

        if(vz_crz<vz_req_mcr):
            altp, rei = self.aircraft.performance.propulsion_ceiling(altp_ini,nei,vz_req_mcr,disa,isomach,mach,mass,"MCR")

            if(rei==1):
                ceiling = 2
            else:
                ceiling = -1

        return altp,ceiling


