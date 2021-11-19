#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Department, ENAC
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import unit, util



class Flight(object):
    """Usefull methods for all simulations
    """
    def __init__(self, airplane):
        self.airplane = airplane

    def level_flight(self,pamb,tamb,mach,mass):
        """Level flight equilibrium
        """
        wing = self.airplane.wing
        aerodynamics = self.airplane.aerodynamics
        propulsion = self.airplane.propulsion

        g = 9.80665
        gam = 1.4
        cz = (2.*mass*g)/(gam*pamb*mach**2*wing.area)
        fn,cx,lod = aerodynamics.drag_force(pamb,tamb,mach,cz)
        sfc = propulsion.unitary_sc(pamb,tamb,mach,fn)
        return sfc, lod, cz, fn

    def mach_from_vcas(self,pamb,Vcas):
        """Mach number from calibrated air speed, subsonic only
        """
        gam = 1.4
        P0 = 101325.
        vc0 = 340.29    # m/s
        fac = gam/(gam-1.)
        mach = np.sqrt(((((((gam-1.)/2.)*(Vcas/vc0)**2+1)**fac-1.)*P0/pamb+1.)**(1./fac)-1.)*(2./(gam-1.)))
        return mach

    def vcas_from_mach(self,pamb,mach):
        """Calibrated air speed from Mach number, subsonic only
        """
        gam = 1.4
        P0 = 101325.
        vc0 = 340.29    # m/s
        fac = gam/(gam-1.)
        vcas = vc0*np.sqrt((2./(gam-1.))*((((pamb/P0)*((1.+((gam-1.)/2.)*mach**2)**fac-1.))+1.)**(1./fac)-1.))
        return vcas

    def get_speed(self,pamb,speed_mode,mach):
        """retrieve CAS or Mach from mach depending on speed_mode
        """
        speed = {"cas" : self.vcas_from_mach(pamb, mach),  # CAS required
                 "mach" : mach  # mach required
                 }.get(speed_mode, "Erreur: select speed_mode equal to cas or mach")
        return speed

    def get_mach(self,pamb,speed_mode,speed):
        """Retrieve Mach from CAS or mach depending on speed_mode
        """
        mach = {"cas" : self.mach_from_vcas(pamb, speed),  # Input is CAS
                "mach" : speed  # Input is mach
                }.get(speed_mode, "Erreur: select speed_mode equal to cas or mach")
        return mach

    def climb_mode(self,speed_mode,mach,dtodz,tstd,disa):
        """Acceleration factor depending on speed driver ('cas': constant CAS, 'mach': constant Mach)
        WARNING : input is mach number whatever speed_mode
        """
        g = 9.80665
        r = 287.053
        gam = 1.4
        if (speed_mode=="cas"):
            fac = (gam-1.)/2.
            acc_factor = 1. + (((1.+fac*mach**2)**(gam/(gam-1.))-1.)/(1.+fac*mach**2)**(1./(gam-1.))) \
                            + ((gam*r)/(2.*g))*(mach**2)*(tstd/(tstd+disa))*dtodz
        elif (speed_mode=="mach"):
            acc_factor = 1. + ((gam*r)/(2.*g))*(mach**2)*(tstd/(tstd+disa))*dtodz
        else:
            raise Exception("climb_mode key is unknown")
        return acc_factor

    def lift_from_speed(self,pamb,tamb,mach,mass):
        """Retrieve cz from mach using simplified lift equation
        """
        wing = self.airplane.wing

        g = 9.80665
        gam = 1.4
        cz = (2.*mass*g)/(gam*pamb*mach**2*wing.area)
        return cz

    def speed_from_lift(self,pamb,tamb,cz,mass):
        """Retrieve mach from cz using simplified lift equation
        """
        wing = self.airplane.wing

        g = 9.80665
        gam = 1.4
        mach = np.sqrt((mass*g)/(0.5*gam*pamb*wing.area*cz))
        return mach

    def air_path(self,nei,altp,disa,speed_mode,speed,mass,rating,kfn, full_output=False):
        """Retrieve air path in various conditions
        """
        aerodynamics = self.airplane.aerodynamics
        propulsion = self.airplane.propulsion

        g = 9.80665
        pamb,tamb,tstd,dtodz = util.atmosphere(altp, disa, full_output=True)
        mach = self.get_mach(pamb,speed_mode,speed)

        thrust = propulsion.unitary_thrust(pamb,tamb,mach,rating)
        sfc = propulsion.unitary_sc(pamb,tamb,mach,thrust)
        fn = kfn * thrust * (propulsion.n_engine - nei)
        ff = sfc * fn
        if kfn!=1. and full_output:
            print("WARNING, air_path method, kfn is different from 1, fuel flow may not be accurate")
        cz = self.lift_from_speed(pamb,tamb,mach,mass)
        fd,cx,lod = aerodynamics.drag_force(pamb,tamb,mach,cz)

        if(nei>0):
            dcx = propulsion.oei_drag(pamb,mach)
            cx = cx + dcx*nei
            lod = cz/cx

        acc_factor = self.climb_mode(speed_mode, mach, dtodz, tstd, disa)
        slope = ( fn/(mass*g) - 1./lod ) / acc_factor
        vz = slope * mach * util.sound_speed(tamb)
        acc = (acc_factor-1.)*g*slope
        if full_output:
            return slope,vz,fn,ff,acc,cz,cx,pamb,tamb
        else:
            return slope,vz

    def max_air_path(self,nei,altp,disa,speed_mode,mass,rating,kfn):
        """Optimize the speed of the aircraft to maximize the air path
        """
        def fct(cz):
            pamb,tamb = util.atmosphere(altp, disa)
            mach = self.speed_from_lift(pamb,tamb,cz,mass)
            speed = self.get_speed(pamb,speed_mode,mach)
            slope,vz = self.air_path(nei,altp,disa,speed_mode,speed,mass,rating,kfn)
            if isformax: return slope
            else: return slope,vz,mach

        cz_ini = 0.5
        dcz = 0.05

        isformax = True
        cz,slope,rc = util.maximize_1d(cz_ini,dcz,[fct])
        isformax = False

        slope,vz,mach = fct(cz)
        return slope,vz,mach,cz


class Missions(Flight):
    def __init__(self, airplane, holding_time, reserve_fuel_ratio, diversion_range):
        super(Missions, self).__init__(airplane)

        self.crz_disa = None
        self.crz_altp = None
        self.crz_mach = None
        self.crz_mass = None

        self.crz_tas = None
        self.crz_cz = None
        self.crz_lod = None
        self.crz_thrust = None
        self.crz_propulsive_power = None
        self.crz_sfc = None

        self.nominal = Breguet(airplane, holding_time, reserve_fuel_ratio, diversion_range)
        self.max_payload = Breguet(airplane, holding_time, reserve_fuel_ratio, diversion_range)
        self.max_fuel = Breguet(airplane, holding_time, reserve_fuel_ratio, diversion_range)
        self.zero_payload = Breguet(airplane, holding_time, reserve_fuel_ratio, diversion_range)
        self.cost = Breguet(airplane, holding_time, reserve_fuel_ratio, diversion_range)

    def eval_cruise_point(self):
        """Evaluate cruise point characteristics
        """
        self.crz_disa = 0
        self.crz_altp = self.airplane.cruise_altp
        self.crz_mach = self.airplane.cruise_mach
        self.crz_mass = 0.97 * self.airplane.mass.mtow

        pamb,tamb = util.atmosphere(self.crz_altp, self.crz_disa)

        sfc, lod, cz, fn = self.level_flight(pamb,tamb,self.crz_mach,self.crz_mass)

        self.crz_tas = self.crz_mach*util.sound_speed(tamb)
        self.crz_cz = cz
        self.crz_lod = lod
        self.crz_thrust = fn
        self.crz_propulsive_power = fn*self.crz_tas
        self.crz_sfc = sfc

    def eval_nominal_mission(self):
        """Compute missions
        """
        self.nominal.altp = self.airplane.cruise_altp
        self.nominal.mach = self.airplane.cruise_mach
        self.nominal.range = self.airplane.design_range
        self.nominal.tow = self.airplane.mass.mtow

        self.nominal.eval()

    def eval_max_payload_mission(self):
        """Compute missions
        """
        self.max_payload.altp = self.airplane.cruise_altp
        self.max_payload.mach = self.airplane.cruise_mach
        self.max_payload.tow = self.airplane.mass.mtow

        self.max_payload.eval()
        self.max_payload.residual = self.airplane.mass.max_payload - self.max_payload.payload       # INFO: range must drive residual to zero

    def eval_max_fuel_mission(self):
        """Compute missions
        """
        self.max_fuel.altp = self.airplane.cruise_altp
        self.max_fuel.mach = self.airplane.cruise_mach
        self.max_fuel.tow = self.airplane.mass.mtow

        self.max_fuel.eval()
        self.max_fuel.residual = self.airplane.mass.mfw - self.max_fuel.fuel_total       # INFO: range must drive residual to zero

    def eval_zero_payload_mission(self):
        """Compute missions
        """
        self.zero_payload.altp = self.airplane.cruise_altp
        self.zero_payload.mach = self.airplane.cruise_mach
        self.zero_payload.tow = self.airplane.mass.owe + self.airplane.mass.mfw

        self.zero_payload.eval()
        self.zero_payload.residual = self.airplane.mass.mfw - self.zero_payload.fuel_total       # INFO: range must drive residual to zero

    def eval_cost_mission(self):
        """Compute missions
        """
        mass = self.airplane.mass

        self.cost.altp = self.airplane.cruise_altp
        self.cost.mach = self.airplane.cruise_mach
        self.cost.range = self.airplane.cost_range

        self.cost.eval()
        self.cost.residual = mass.nominal_payload - self.cost.payload       # INFO: tow must drive residual to zero

    def eval_mission_solver(self, mission, var):

        if mission not in ["max_payload","max_fuel","zero_payload","cost"]:
            raise Exception("mission type not allowed")

        def fct(x, self):
            exec("self."+mission+"."+var+" = x")
            eval("self.eval_"+mission+"_mission()")
            return eval("self."+mission+".residual")

        xini = eval("self."+mission+"."+var)
        output_dict = fsolve(fct, x0=xini, args=(self), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        exec("self."+mission+"."+var+" = output_dict[0][0]")
        eval("self.eval_"+mission+"_mission()")

    def eval_payload_range_solver(self):
        """Compute missions and solve them
        """
        self.eval_cruise_point()
        self.eval_mission_solver("max_payload", "range")
        self.eval_mission_solver("max_fuel", "range")
        self.eval_mission_solver("zero_payload", "range")
        self.eval_mission_solver("cost", "tow")

    def eval_cost_mission_solver(self):
        """Compute missions and solve them
        """
        self.eval_mission_solver("cost", "tow")

    def payload_range_diagram(self):
        """
        Print the payload - range diagram
        """
        window_title = "MARILib extract"
        plot_title = "Payload - Range"

        payload = [self.max_payload.payload,
                   self.max_payload.payload,
                   self.max_fuel.payload,
                   0.]

        range = [0.,
                 unit.NM_m(self.max_payload.range),
                 unit.NM_m(self.max_fuel.range),
                 unit.NM_m(self.zero_payload.range)]

        nominal = [self.nominal.payload,
                   unit.NM_m(self.nominal.range)]

        fig,axes = plt.subplots(1,1)
        fig.canvas.set_window_title(window_title)
        fig.suptitle(plot_title, fontsize=14)

        plt.plot(range,payload,linewidth=2,color="blue")
        plt.scatter(range[1:],payload[1:],marker="+",c="orange",s=100)
        plt.scatter(nominal[1],nominal[0],marker="o",c="green",s=50)

        plt.grid(True)

        plt.ylabel('Payload (kg)')
        plt.xlabel('Range (NM)')

        plt.show()


class Breguet(Flight):
    def __init__(self, airplane, holding_time, reserve_fuel_ratio, diversion_range):
        super(Breguet, self).__init__(airplane)

        self.disa = 0.      # Mean cruise temperature shift
        self.altp = None    # Mean cruise altitude
        self.mach = None    # Cruise mach number

        range_init = airplane.design_range
        tow_init = 5. * 110. * airplane.cabin.n_pax
        total_fuel_init = 0.2 * tow_init

        self.range = range_init             # Mission distance
        self.tow = tow_init                 # Take Off Weight
        self.payload = None                 # Payload Weight
        self.time_block = None              # Mission block duration
        self.fuel_block = None              # Mission block fuel consumption
        self.fuel_reserve = None            # Mission reserve fuel
        self.fuel_total = total_fuel_init   # Mission total fuel

        self.holding_time = holding_time
        self.reserve_fuel_ratio = reserve_fuel_ratio
        self.diversion_range = diversion_range

    def holding(self,time,mass,altp,mach,disa):
        """Holding fuel
        """
        g = 9.80665
        pamb,tamb = util.atmosphere(altp, disa)
        sfc, lod, cz, fn = self.level_flight(pamb,tamb,mach,mass)
        fuel = sfc*(mass*g/lod)*time
        return fuel

    def breguet_range(self,range,tow,altp,mach,disa):
        """Breguet range equation
        """
        g = 9.80665
        pamb,tamb = util.atmosphere(altp, disa)
        tas = mach * util.sound_speed(tamb)
        time = 1.09*(range/tas)
        sfc, lod, cz, fn = self.level_flight(pamb,tamb,mach,tow)
        val = 1.05 * tow*(1-np.exp(-(sfc*g*range)/(tas*lod)))
        return val,time

    def eval(self):
        """
        Mission computation using bregueçt equation, fixed L/D and fixed sfc
        """
        mass = self.airplane.mass
        nacelles = self.airplane.nacelles
        propulsion = self.airplane.propulsion

        g = 9.80665

        disa = self.disa
        altp = self.altp
        mach = self.mach

        range = self.range
        tow = self.tow

        # Departure ground legs
        #-----------------------------------------------------------------------------------------------------------
        time_taxi_out = 540.
        fuel_taxi_out = (34. + 2.3e-4*nacelles.engine_slst)*propulsion.n_engine

        time_take_off = 220.*tow/(nacelles.engine_slst*propulsion.n_engine)
        fuel_take_off = 1e-4*(2.8+2.3/nacelles.engine_bpr)*tow

        # Mission leg
        #-----------------------------------------------------------------------------------------------------------
        fuel_mission,time_mission = self.breguet_range(range,tow,altp,mach,disa)

        weight = tow - (fuel_taxi_out + fuel_take_off + fuel_mission)     # mass is not landing weight

        # Arrival ground legs
        #-----------------------------------------------------------------------------------------------------------
        time_landing = 180.
        fuel_landing = 1e-4*(0.5+2.3/nacelles.engine_bpr)*weight

        time_taxi_in = 420.
        fuel_taxi_in = (26. + 1.8e-4*nacelles.engine_slst)*propulsion.n_engine

        # Block fuel and time
        #-----------------------------------------------------------------------------------------------------------
        self.fuel_block = fuel_taxi_out + fuel_take_off + fuel_mission + fuel_landing + fuel_taxi_in
        self.time_block = time_taxi_out + time_take_off + time_mission + time_landing + time_taxi_in

        # Diversion fuel
        #-----------------------------------------------------------------------------------------------------------
        fuel_diversion,t = self.breguet_range(self.diversion_range,weight,altp,mach,disa)

        # Holding fuel
        #-----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.65 * mach
        fuel_holding = self.holding(self.holding_time,weight,altp_holding,mach_holding,disa)

        # Total
        #-----------------------------------------------------------------------------------------------------------
        self.fuel_reserve = fuel_mission*self.reserve_fuel_ratio + fuel_diversion + fuel_holding
        self.fuel_total = float(self.fuel_block + self.fuel_reserve)
        self.payload = self.tow - mass.owe - self.fuel_total

        #-----------------------------------------------------------------------------------------------------------
        return


class Operations(Flight):
    def __init__(self, airplane, hld_conf_to, kvs1g_to, s2_min_path_to, hld_conf_ld, kvs1g_ld,
                                 tofl_req, app_speed_req, vz_mcl_req, vz_mcr_req, oei_path_req, oei_altp_req):
        super(Operations, self).__init__(airplane)

        self.take_off = TakeOff(airplane, hld_conf_to, kvs1g_to, s2_min_path_to, tofl_req)
        self.approach = Approach(airplane, hld_conf_ld, kvs1g_ld, app_speed_req)
        self.mcl_ceiling = ClimbCeiling(airplane, rating="MCL", speed_mode="cas", vz_req=vz_mcl_req)
        self.mcr_ceiling = ClimbCeiling(airplane, rating="MCR", speed_mode="mach", vz_req=vz_mcr_req)
        self.oei_ceiling = OeiCeiling(airplane, rating="MCN", speed_mode="mach", path_req=oei_path_req, altp_req=oei_altp_req)

    def eval_take_off(self):
        """Compute performances
        """
        mass = self.airplane.mass

        self.take_off.disa = 15.
        self.take_off.altp = unit.m_ft(0.)
        self.take_off.tow = mass.mtow
        self.take_off.eval()

    def eval_approach(self):
        """Compute performances
        """
        mass = self.airplane.mass

        self.approach.disa = 0.
        self.approach.altp = unit.m_ft(0.)
        self.approach.lw = mass.mlw
        self.approach.eval()

    def eval_climb_ceiling(self):
        """Compute performances
        """
        mass = self.airplane.mass

        self.mcl_ceiling.disa = 15.
        self.mcl_ceiling.altp = self.airplane.cruise_altp
        self.mcl_ceiling.mach = self.airplane.cruise_mach
        self.mcl_ceiling.mass = 0.97 * mass.mtow
        self.mcl_ceiling.eval()

        self.mcr_ceiling.disa = 15.
        self.mcr_ceiling.altp = self.airplane.cruise_altp
        self.mcr_ceiling.mach = self.airplane.cruise_mach
        self.mcr_ceiling.mass = 0.97 * mass.mtow
        self.mcr_ceiling.eval()

    def eval_oei_ceiling(self):
        """Compute performances
        """
        mass = self.airplane.mass

        self.oei_ceiling.disa = 15.
        self.oei_ceiling.mass = 0.97 * mass.mtow
        self.oei_ceiling.eval()

    def solve_oei_ceiling(self):
        """Compute performances
        """
        mass = self.airplane.mass

        self.oei_ceiling.disa = 15.
        self.oei_ceiling.mass = 0.97 * mass.mtow
        self.oei_ceiling.solve()

    def eval(self):
        """Compute performances
        """
        self.eval_take_off()
        self.eval_approach()
        self.eval_climb_ceiling()
        self.solve_oei_ceiling()


class TakeOff(Flight):
    """Take Off Field Length
    """
    def __init__(self, airplane, hld_conf, kvs1g_req, s2_path_req, tofl_req):
        super(TakeOff, self).__init__(airplane)

        self.disa = None
        self.altp = None
        self.tow = None
        self.tofl_eff = None      # INFO: tofl_eff must be lower or equal to tofl_req
        self.kvs1g_eff = None
        self.s2_path_eff = None
        self.limit = None

        self.v2 = None
        self.mach2 = None

        self.tofl_req = tofl_req
        self.kvs1g_req = kvs1g_req
        self.s2_path_req = s2_path_req
        self.hld_conf = hld_conf

    def eval(self):
        """Take off field length and climb path with eventual kVs1g increase to recover min regulatory slope
        """
        disa = self.disa
        altp = self.altp
        mass = self.tow

        s2_min_path = self.s2_path_req
        kvs1g = self.kvs1g_req
        hld_conf = self.hld_conf
        rating = "MTO"
        kfn = 1.

        tofl,s2_path,cas,mach = self.take_off(kvs1g,altp,disa,mass,hld_conf,rating,kfn)

        if(s2_min_path<s2_path):
            limitation = "fl"   # field length
        else:
            dkvs1g = 0.005
            kvs1g_ = np.array([0.,0.])
            kvs1g_[0] = kvs1g
            kvs1g_[1] = kvs1g_[0] + dkvs1g

            s2_path_ = np.array([0.,0.])
            s2_path_[0] = s2_path
            tofl,s2_path_[1],cas,mach = self.take_off(kvs1g_[1],altp,disa,mass,hld_conf,rating,kfn)

            while(s2_path_[0]<s2_path_[1] and s2_path_[1]<s2_min_path):
                kvs1g_[0] = kvs1g_[1]
                kvs1g_[1] = kvs1g_[1] + dkvs1g
                tofl,s2_path_[1],cas,mach = self.take_off(kvs1g_[1],altp,disa,mass,hld_conf,rating,kfn)

            if(s2_min_path<s2_path_[1]):
                kvs1g = kvs1g_[0] + ((kvs1g_[1]-kvs1g_[0])/(s2_path_[1]-s2_path_[0]))*(s2_min_path-s2_path_[0])
                tofl,s2_path,cas,mach = self.take_off(kvs1g,altp,disa,mass,hld_conf,rating,kfn)
                s2_path = s2_min_path
                limitation = "s2"   # second segment
            else:
                tofl = np.nan
                kvs1g = np.nan
                s2_path = 0.
                limitation = None

        self.tofl_eff = tofl
        self.kvs1g_eff = kvs1g
        self.s2_path_eff = s2_path
        self.limit = limitation
        self.v2 = cas
        self.mach2 = mach
        return

    def take_off(self,kvs1g,altp,disa,mass,hld_conf,rating,kfn):
        """Take off field length and climb path at 35 ft depending on stall margin (kVs1g)
        """
        wing = self.airplane.wing
        aerodynamics = self.airplane.aerodynamics
        propulsion = self.airplane.propulsion

        czmax,cz0 = aerodynamics.wing_high_lift(hld_conf)

        pamb,tamb = util.atmosphere(altp, disa)
        rho,sig = util.air_density(pamb, tamb)

        cz_to = czmax / kvs1g**2
        mach = self.speed_from_lift(pamb,tamb,cz_to,mass)
        speed_factor = 0.7

        nei = 0             # For tofl computation
        thrust = propulsion.unitary_thrust(pamb,tamb,speed_factor*mach,rating)
        fn = kfn*thrust*(propulsion.n_engine - nei)

        ml_factor = mass**2 / (cz_to*fn*wing.area*sig**0.8 )  # Magic Line factor
        tofl = 11.8*ml_factor + 100.    # Magic line

        nei = 1             # For 2nd segment computation
        speed_mode = "cas"  # Constant CAS
        speed = self.get_speed(pamb,speed_mode,mach)

        s2_path,vz = self.air_path(nei,altp,disa,speed_mode,speed,mass,"MTO",kfn)

        return tofl,s2_path,speed,mach


class Approach(Flight):
    """Approach speed
    """
    def __init__(self, airplane, hld_conf, kvs1g, app_speed_req):
        super(Approach, self).__init__(airplane)

        self.disa = None
        self.altp = None
        self.lw = None
        self.app_speed_eff = None      # INFO: app_speed_eff must be lower or equal to app_speed_req

        self.app_speed_req = app_speed_req
        self.hld_conf = hld_conf
        self.kvs1g = kvs1g

    def eval(self):
        """Minimum approach speed (VLS)
        """
        aerodynamics = self.airplane.aerodynamics

        disa = self.disa
        altp = self.altp
        mass = self.lw

        hld_conf = self.hld_conf
        kvs1g = self.kvs1g

        pamb,tamb = util.atmosphere(altp, disa)
        czmax,cz0 = aerodynamics.wing_high_lift(hld_conf)
        cz = czmax / kvs1g**2
        mach = self.speed_from_lift(pamb,tamb,cz,mass)
        vapp = self.get_speed(pamb,"cas",mach)
        self.app_speed_eff = vapp
        return


class ClimbCeiling(Flight):
    """Propulsion ceiling in MCL rating
    """
    def __init__(self, airplane, rating, speed_mode, vz_req):
        super(ClimbCeiling, self).__init__(airplane)

        self.disa = None
        self.altp = None
        self.mach = None
        self.mass = None
        self.vz_eff = None      # INFO: vz_eff must be higher or equal to vz_req

        self.vz_req = vz_req
        self.rating = rating
        self.speed_mode = speed_mode

    def eval(self):
        """Residual climb speed in MCL rating
        """
        disa = self.disa
        altp = self.altp
        mach = self.mach
        mass = self.mass

        speed_mode = self.speed_mode
        rating = self.rating
        kfn = 1.
        nei = 0
        pamb,tamb = util.atmosphere(altp, disa)
        speed = self.get_speed(pamb,self.speed_mode,mach)
        path,vz = self.air_path(nei,altp,disa,speed_mode,speed,mass,rating,kfn)
        self.vz_eff = vz
        return


class OeiCeiling(Flight):
    """One engine ceiling in MCN rating
    """
    def __init__(self, airplane, rating, speed_mode, path_req, altp_req):
        super(OeiCeiling, self).__init__(airplane)

        self.disa = None
        self.altp_req = altp_req
        self.mach = 0.55*self.airplane.cruise_mach      # INFO: mach must maximize path_eff
        self.path_eff = None

        self.path_req = path_req
        self.rating = rating
        self.speed_mode = speed_mode

    def eval(self):
        """One engine ceiling in MCN rating WITHOUT speed optimization
        """
        disa = self.disa
        altp = self.altp_req
        mach = self.mach
        mass = self.mass

        speed_mode = self.speed_mode
        rating = self.rating
        kfn = 1.
        nei = 1
        pamb,tamb = util.atmosphere(altp, disa)
        speed = self.get_speed(pamb,speed_mode,mach)
        path,vz = self.air_path(nei,altp,disa,speed_mode,speed,mass,rating,kfn)
        self.path_eff = path
        return

    def solve(self):
        """One engine ceiling in MCN rating WITH speed optimization
        """
        disa = self.disa
        altp = self.altp_req
        mass = self.mass

        speed_mode = self.speed_mode
        rating = self.rating
        kfn = 1.
        nei = 1
        path,vz,mach,cz = self.max_air_path(nei,altp,disa,speed_mode,mass,rating,kfn)
        self.mach = mach
        self.path_eff = path
        return



class Economics():

    def __init__(self, airplane, d_cost):
        self.airplane = airplane

        cost_range = self.airplane.cost_range

        self.period = unit.s_year(15)
        self.irp = unit.s_year(10)
        self.interest_rate = 0.04
        self.labor_cost = 120.
        self.utilization = self.yearly_utilization(cost_range)

        self.fuel_price = 2./unit.m3_usgal(1)
        self.energy_price = 0.10/unit.W_kW(1)
        self.battery_price = 20.

        self.engine_price = None
        self.gear_price = None
        self.frame_price = None

        self.frame_cost = None
        self.engine_cost = None
        self.cockpit_crew_cost = None
        self.cabin_crew_cost = None
        self.landing_fees = None
        self.navigation_fees = None
        self.catering_cost = None
        self.pax_handling_cost = None
        self.ramp_handling_cost = None

        self.d_cost = d_cost
        self.std_op_cost = None
        self.cash_op_cost = None
        self.direct_op_cost = None

    def yearly_utilization(self, mean_range):
        """Compute the yearly utilization from the average range

        :param mean_range: Average range
        :return:
        """
        range = unit.convert_from("NM",
                      [ 100.,  500., 1000., 1500., 2000., 2500., 3000., 3500., 4000.])
        utilization = [2300., 2300., 1500., 1200.,  900.,  800.,  700.,  600.,  600.]
        return util.lin_interp_1d(mean_range, range, utilization)

    def landing_gear_price(self):
        """Typical value
        """
        landing_gears = self.airplane.landing_gears

        gear_price = 720. * landing_gears.mass
        return gear_price

    def one_engine_price(self):
        """Regression on catalog prices
        """
        nacelles = self.airplane.nacelles

        engine_price = ((2.115e-4*nacelles.engine_slst + 78.85)*nacelles.engine_slst)
        return engine_price

    def one_airframe_price(self):
        """Regression on catalog prices corrected with engine prices
        """
        mass = self.airplane.mass

        airframe_price = 0.7e3*(9e4 + 1.15*mass.mwe - 1.8e9/(2e4 + mass.mwe**0.94))
        return airframe_price

    def eval(self):
        """Computes Cash and Direct Operating Costs per flight (based on AAE 451 Spring 2004)
        """
        cabin = self.airplane.cabin
        nacelles = self.airplane.nacelles
        propulsion = self.airplane.propulsion
        mass = self.airplane.mass
        missions = self.airplane.missions

        # Cash Operating Cost
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        fuel_density = 803.
        fuel_block = missions.cost.fuel_block
        self.fuel_cost = fuel_block*self.fuel_price/fuel_density

        b_h = missions.cost.time_block/3600.
        t_t = b_h + 0.25

        w_f = (10000. + mass.mwe - nacelles.mass)*1.e-5

        labor_frame = ((1.26+1.774*w_f-0.1071*w_f**2)*t_t + (1.614+0.7227*w_f+0.1204*w_f**2))*self.labor_cost
        matrl_frame = (12.39+29.8*w_f+0.1806*w_f**2)*t_t + (15.20+97.330*w_f-2.8620*w_f**2)
        self.frame_cost = labor_frame + matrl_frame

        t_h = 0.05*((nacelles.engine_slst)/4.4482198)*1e-4

        labor_engine = propulsion.n_engine*(0.645*t_t+t_h*(0.566*t_t+0.434))*self.labor_cost
        matrl_engine = propulsion.n_engine*(25.*t_t+t_h*(0.62*t_t+0.38))

        self.engine_cost = labor_engine + matrl_engine

        w_g = mass.mtow*1e-3

        self.cockpit_crew_cost = b_h*2*(440-0.532*w_g)
        self.cabin_crew_cost = b_h*np.ceil(cabin.n_pax/50.)*self.labor_cost
        self.landing_fees = 8.66*(mass.mtow*1e-3)
        self.navigation_fees = 57.*(self.airplane.cost_range/185200.)*np.sqrt((mass.mtow/1000.)/50.)
        self.catering_cost = 3.07 * cabin.n_pax
        self.pax_handling_cost = 2. * cabin.n_pax
        self.ramp_handling_cost = 8.70 * cabin.n_pax
        self.std_op_cost = self.fuel_cost + self.frame_cost + self.engine_cost + self.cockpit_crew_cost + self.landing_fees + self.navigation_fees #+ self.elec_cost
        self.cash_op_cost = self.std_op_cost + self.cabin_crew_cost + self.catering_cost + self.pax_handling_cost + self.ramp_handling_cost + self.d_cost

        # DirectOperating Cost
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        self.engine_price = self.one_engine_price()
        self.gear_price = self.landing_gear_price()
        self.frame_price = self.one_airframe_price()

#        battery_price = eco.battery_mass_price*cost_mission.req_battery_mass

        self.utilization = self.yearly_utilization(self.airplane.cost_range)
        self.aircraft_price = self.frame_price + self.engine_price * propulsion.n_engine + self.gear_price #+ battery_price
        self.total_investment = self.frame_price * 1.06 + propulsion.n_engine * self.engine_price * 1.025
        irp_year = unit.year_s(self.irp)
        period_year = unit.year_s(self.period)
        self.interest = (self.total_investment/(self.utilization*period_year)) * (irp_year * 0.04 * (((1. + self.interest_rate)**irp_year)/((1. + self.interest_rate)**irp_year - 1.)) - 1.)
        self.insurance = 0.0035 * self.aircraft_price/self.utilization
        self.depreciation = 0.99 * (self.total_investment / (self.utilization * period_year))     # Depreciation
        self.direct_op_cost = self.cash_op_cost + self.interest + self.depreciation + self.insurance

        return

