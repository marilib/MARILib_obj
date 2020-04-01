#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

from aircraft.tool import unit
import earth

import numpy as np
from scipy.optimize import fsolve

from aircraft.tool.math import vander3, trinome, maximize_1d
import aircraft.requirement as requirement


class Performance(object):
    """
    Master class for all aircraft performances
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.mission = Mission(aircraft)
        self.take_off = Take_off(aircraft)
        self.approach = Approach(aircraft)
        self.mcr_ceiling = Power_ceiling(aircraft)
        self.mcl_ceiling = Power_ceiling(aircraft)
        self.oei_ceiling = OEI_ceiling(aircraft)
        self.time_to_climb = Time_to_Climb(aircraft)

    def analysis(self):
        disa = self.aircraft.requirement.take_off.disa
        altp = self.aircraft.requirement.take_off.altp
        kmtow = self.aircraft.requirement.take_off.kmtow
        kvs1g = self.aircraft.requirement.take_off.kvs1g
        s2_min_path = self.aircraft.requirement.take_off.s2_min_path
        hld_conf = self.aircraft.performance.take_off.hld_conf
        mass = kmtow*self.aircraft.weight_cg.mtow

        self.tofl_req = self.aircraft.requirement.take_off.tofl_req
        self.take_off.eval(disa,altp,mass,hld_conf,kvs1g,s2_min_path)

        disa = self.aircraft.requirement.approach.disa
        altp = self.aircraft.requirement.approach.altp
        kmlw = self.aircraft.requirement.approach.kmlw
        kvs1g = self.aircraft.requirement.approach.kvs1g
        hld_conf = self.aircraft.performance.approach.hld_conf
        mass = kmlw*self.aircraft.weight_cg.mlw

        self.app_speed_req = self.aircraft.requirement.approach.app_speed_req
        self.approach.eval(disa,altp,mass,hld_conf,kvs1g)

        disa = self.aircraft.requirement.vz_mcl.disa
        altp = self.aircraft.requirement.vz_mcl.altp
        kmtow = self.aircraft.requirement.vz_mcl.kmtow
        mach = self.aircraft.requirement.vz_mcl.mach
        rating = self.aircraft.requirement.vz_mcl.rating
        speed_mode = self.aircraft.requirement.vz_mcl.speed_mode
        mass = kmtow*self.aircraft.weight_cg.mtow

        self.vz_req = self.aircraft.requirement.vz_mcl.vz_req
        self.mcl_ceiling.eval(disa,altp,mach,mass,rating,speed_mode)

        disa = self.aircraft.requirement.vz_mcr.disa
        altp = self.aircraft.requirement.vz_mcr.altp
        kmtow = self.aircraft.requirement.vz_mcr.kmtow
        mach = self.aircraft.requirement.vz_mcr.mach
        rating = self.aircraft.requirement.vz_mcr.rating
        speed_mode = self.aircraft.requirement.vz_mcr.speed_mode
        mass = kmtow*self.aircraft.weight_cg.mtow

        self.vz_req = self.aircraft.requirement.vz_mcr.vz_req
        self.mcr_ceiling.eval(disa,altp,mach,mass,rating,speed_mode)

        disa = self.aircraft.requirement.oei.disa
        altp = self.aircraft.requirement.oei.altp
        kmtow = self.aircraft.requirement.oei.kmtow
        rating = self.aircraft.requirement.oei.rating
        speed_mode = self.aircraft.requirement.oei.speed_mode
        mass = kmtow*self.aircraft.weight_cg.mtow

        self.path_req = self.aircraft.requirement.oei.path_req
        self.oei_ceiling.eval(disa,altp,mass,rating,speed_mode)

        disa = self.aircraft.requirement.ttc.disa
        cas1 = self.aircraft.requirement.ttc.cas1
        altp1 = self.aircraft.requirement.ttc.altp1
        cas2 = self.aircraft.requirement.ttc.cas2
        altp2 = self.aircraft.requirement.ttc.altp2
        mach = self.aircraft.requirement.ttc.mach
        toc = self.aircraft.requirement.ttc.altp
        mass = self.aircraft.weight_cg.mtow

        self.ttc_req = self.aircraft.requirement.ttc.ttc_req
        self.time_to_climb.eval(disa,toc,mach,mass,altp1,cas1,altp2,cas2)

    def get_speed(self,pamb,speed_mode,mach):
        """retrieves CAS or Mach from mach depending on speed_mode
        """
        speed = {"cas" : earth.vcas_from_mach(pamb,mach),   # CAS required
                 "mach" : mach                               # mach required
                 }.get(speed_mode, "Erreur: select speed_mode equal to cas or mach")
        return speed

    def get_mach(self,pamb,speed_mode,speed):
        """
        Retrieves Mach from CAS or mach depending on speed_mode
        """
        mach = {"cas" : earth.mach_from_vcas(pamb,speed),   # Input is CAS
                "mach" : speed                               # Input is mach
                }.get(speed_mode, "Erreur: select speed_mode equal to cas or mach")
        return mach

    def speed_from_lift(self,pamb,tamb,cz,mass):
        """Retrieves mach from cz using simplified lift equation
        """
        g = earth.gravity()
        r,gam,Cp,Cv = earth.gas_data()
        mach = np.sqrt((mass*g)/(0.5*gam*pamb*self.aircraft.airframe.wing.area*cz))
        return mach

    def lift_from_speed(self,pamb,tamb,mach,mass):
        """Retrieves cz from mach using simplified lift equation
        """
        g = earth.gravity()
        r,gam,Cp,Cv = earth.gas_data()
        cz = (2.*mass*g)/(gam*pamb*mach**2*self.aircraft.airframe.wing.area)
        return cz

    def level_flight(self,pamb,tamb,mach,mass):
        """Level flight equilibrium
        """
        g = earth.gravity()
        r,gam,Cp,Cv = earth.gas_data()

        cz = (2.*mass*g)/(gam*pamb*mach**2*self.aircraft.airframe.wing.area)
        cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

        fn = (gam/2.)*pamb*mach**2*self.aircraft.airframe.wing.area*cx
        sfc,throttle = self.aircraft.power_system.sc(pamb,tamb,mach,"MCR",fn)
        if (throttle>1.): print("level_flight, throttle is higher than 1, throttle = ",throttle)

        return cz,cx,lod,fn,sfc,throttle

    def air_path(self,nei,altp,disa,speed_mode,speed,mass,rating):
        """Retrieves air path in various conditions
        """
        g = earth.gravity()
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        mach = self.get_mach(pamb,speed_mode,speed)

        fn,ff,sfc = self.aircraft.power_system.thrust(pamb,tamb,mach,rating)
        cz = self.lift_from_speed(pamb,tamb,mach,mass)
        cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

        if(nei>0):
            dcx = self.aircraft.power_system.oei_drag(pamb,mach)
            cx = cx + dcx*nei
            lod = cz/cx

        acc_factor = earth.climb_mode(speed_mode,dtodz,tstd,disa,mach)
        slope = ( fn/(mass*g) - 1/lod ) / acc_factor
        vz = mach*slope*earth.sound_speed(tamb)

        return slope,vz

    def max_air_path(self,nei,altp,disa,speed_mode,mass,rating):
        """Optimizes the speed of the aircraft to maximize the air path
        """
        def fct(cz):
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
            mach = self.speed_from_lift(pamb,tamb,cz,mass)
            speed = self.get_speed(pamb,speed_mode,mach)
            [slope,vz] = self.air_path(nei,altp,disa,speed_mode,speed,mass,rating)
            if isformax: return slope
            else: return slope,vz,mach

        cz_ini = 0.5
        dcz = 0.05
        isformax = True

        cz,slope,rc = maximize_1d(cz_ini,dcz,[fct])

        isformax = False

        slope,vz,mach = fct(cz)

        return slope,vz,mach,cz


    def acceleration(self,nei,altp,disa,speed_mode,speed,mass,rating):
        """Aircraft acceleration on level flight
        """
        r,gam,Cp,Cv = earth.gas_data()

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        mach = self.get_mach(pamb,speed_mode,speed)

        fn,ff,sfc = self.aircraft.power_system.thrust(pamb,tamb,mach,rating,nei=nei)

        cz = self.lift_from_speed(pamb,tamb,mach,mass)
        cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

        if(nei>0):
            dcx = self.aircraft.power_system.oei_drag(pamb,mach)
            cx = cx + dcx*nei

        acc = (fn - 0.5*gam*pamb*mach**2*self.aircraft.airframe.wing.area*cx) / mass

        return acc


class Take_off(requirement.Take_off_req):
    """Definition of all mission types
    """
    def __init__(self, aircraft):
        super(Take_off, self).__init__(aircraft.arrangement, aircraft.requirement)
        self.aircraft = aircraft

        self.hld_conf = self.aircraft.aerodynamics.hld_conf_to
        self.tofl_eff = None
        self.kvs1g_eff = None
        self.v2 = None
        self.s2_path = None
        self.limit = None

    def eval(self,disa,altp,mass,hld_conf,kvs1g,s2_min_path):
        """Take off field length and climb path with eventual kVs1g increase to recover min regulatory slope
        """
        self.disa = disa
        self.altp = altp
        self.kmtow = mass/self.aircraft.weight_cg.mtow
        self.kvs1g = kvs1g
        self.s2_min_path = s2_min_path

        tofl,s2_path,cas,mach = self.take_off(kvs1g,altp,disa,mass,hld_conf)

        if(s2_min_path<s2_path):
            limitation = "fl"   # field length
        else:
            dkvs1g = 0.005
            kvs1g_ = np.array([0.,0.])
            kvs1g_[0] = self.kvs1g
            kvs1g_[1] = kvs1g_[0] + dkvs1g

            s2_path_ = np.array([0.,0.])
            s2_path_[0] = s2_path
            tofl,s2_path_[1],cas,mach = self.take_off(kvs1g_[1],altp,disa,mass,hld_conf)

            while(s2_path_[0]<s2_path_[1] and s2_path_[1]<s2_min_path):
                kvs1g_[0] = kvs1g_[1]
                kvs1g_[1] = kvs1g_[1] + dkvs1g
                tofl,s2_path_[1],cas,mach = self.take_off(kvs1g_[1],altp,disa,mass,hld_conf)

            if(s2_min_path<s2_path_[1]):
                kvs1g = kvs1g_[0] + ((kvs1g_[1]-kvs1g_[0])/(s2_path_[1]-s2_path_[0]))*(s2_min_path-s2_path_[0])
                tofl,s2_path,cas,mach = self.take_off(kvs1g,altp,disa,mass,hld_conf)
                s2_path = s2_min_path
                limitation = "s2"   # second segment
            else:
                tofl = np.nan
                kvs1g = np.nan
                s2_path = 0.
                limitation = None

            self.hld_conf = hld_conf
            self.tofl_eff = tofl
            self.kvs1g_eff = kvs1g
            self.s2_path = s2_path
            self.v2 = cas
            self.limit = limitation

        return

    def take_off(self,kvs1g,altp,disa,mass,hld_conf):
        """Take off field length and climb path at 35 ft depending on stall margin (kVs1g)
        """
        czmax,cz0 = self.aircraft.airframe.wing.high_lift(hld_conf)

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        rho,sig = earth.air_density(pamb,tamb)

        cz_to = czmax / kvs1g**2
        mach = self.aircraft.performance.speed_from_lift(pamb,tamb,cz_to,mass)

        nei = 0    # For Magic Line factor computation
        throttle = 1.
        fn,ff,sfc = self.aircraft.power_system.thrust(pamb,tamb,mach,"MTO",throttle,nei)

        ml_factor = mass**2 / (cz_to*fn*self.aircraft.airframe.wing.area*sig**0.8 )  # Magic Line factor
        tofl = 15.5*ml_factor + 100.    # Magic line

        nei = 1             # For 2nd segment computation
        speed_mode = "cas"  # Constant CAS
        speed = self.aircraft.performance.get_speed(pamb,speed_mode,mach)

        s2_path,vz = self.aircraft.performance.air_path(nei,altp,disa,speed_mode,speed,mass,"MTO")

        return tofl,s2_path,speed,mach


class Approach(requirement.Approach_req):
    """Definition of all mission types
    """
    def __init__(self, aircraft):
        super(Approach, self).__init__(aircraft.arrangement, aircraft.requirement)
        self.aircraft = aircraft

        self.hld_conf = self.aircraft.aerodynamics.hld_conf_ld
        self.app_speed_eff = None

    def eval(self,disa,altp,mass,hld_conf,kvs1g):
        """
        Minimum approach speed (VLS)
        """
        self.disa = disa
        self.altp = altp
        self.kmlw = mass/self.aircraft.weight_cg.mlw
        self.kvs1g = kvs1g

        g = earth.gravity()

        czmax,cz0 = self.aircraft.airframe.wing.high_lift(hld_conf)

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        rho,sig = earth.air_density(pamb,tamb)

        vapp = np.sqrt((mass*g) / (0.5*rho*self.aircraft.airframe.wing.area*(czmax / kvs1g**2)))

        self.hld_conf = hld_conf
        self.app_speed_eff = vapp

        return


class Power_ceiling(requirement.Climb_req):
    """Definition of all mission types
    """
    def __init__(self, aircraft):
        super(Power_ceiling, self).__init__(aircraft.arrangement, aircraft.requirement)
        self.aircraft = aircraft

        self.rating = None
        self.vz_eff = None

    def eval(self,disa,altp,mach,mass,rating,speed_mode):
        """
        Minimum approach speed (VLS)
        """
        self.disa = disa
        self.altp = altp
        self.mach = mach
        self.rating = rating
        self.speed_mode = speed_mode
        self.kmtow = mass/self.aircraft.weight_cg.mtow

        nei = 0

        slope,vz = self.aircraft.performance.air_path(nei,altp,disa,speed_mode,mach,mass,rating)

        self.vz_eff = vz

        return


class OEI_ceiling(requirement.OEI_ceiling_req):
    """Definition of all mission types
    """
    def __init__(self, aircraft):
        super(OEI_ceiling, self).__init__(aircraft.arrangement, aircraft.requirement)
        self.aircraft = aircraft

        self.path_eff = None
        self.mach_opt = None

    def eval(self,disa,altp,mass,rating,speed_mode):
        """
        Compute one engine inoperative maximum path
        """
        self.disa = disa
        self.altp = altp
        self.kmtow = mass/self.aircraft.weight_cg.mtow
        self.rating = rating
        self.speed_mode = speed_mode

        nei = 1.

        path,vz,mach,cz = self.aircraft.performance.max_air_path(nei,altp,disa,speed_mode,mass,rating)

        self.path_eff = path
        self.mach_opt = mach

        return


class Time_to_Climb(requirement.TTC_req):
    """
    Definition of all mission types
    """
    def __init__(self, aircraft):
        super(Time_to_Climb, self).__init__(aircraft.arrangement, aircraft.requirement)
        self.aircraft = aircraft

        self.mass = None
        self.ttc_eff = None

    def eval(self,disa,toc,mach,mass,altp1,vcas1,altp2,vcas2):
        """
        Time to climb to initial cruise altitude
        For simplicity reasons, airplane mass is supposed constant
        """
        self.disa = disa
        self.cas1 = vcas1
        self.altp1 = altp1
        self.cas2 = vcas2
        self.altp2 = altp2
        self.mach = mach
        self.altp = toc

        if(vcas1>unit.mps_kt(250.)):
            print("vcas1 = ",unit.kt_mps(vcas1))
            print("vcas1 must be lower than or equal to 250kt")
        if(vcas1>vcas2):
            print("vcas1 = ",unit.kt_mps(vcas1))
            print("vcas2 = ",unit.kt_mps(vcas2))
            print("vcas1 must be lower than or equal to vcas2")

        cross_over_altp = earth.cross_over_altp(vcas2,mach)

        if(cross_over_altp<altp1):
            print("Cross over altitude is too low")

        if(toc<cross_over_altp):
            cross_over_altp = toc

        # Duration of initial climb
        #-----------------------------------------------------------------------------------------------------------
        altp_0 = altp1
        altp_2 = altp2
        altp_1 = (altp_0+altp_2)/2.
        altp = np.array([altp_0, altp_1, altp_2])

        nei = 0
        speed_mode = "cas"    # Constant CAS
        rating = "MCL"

        [slope,v_z0] = self.aircraft.performance.air_path(nei,altp[0],disa,speed_mode,vcas1,mass,rating)
        [slope,v_z1] = self.aircraft.performance.air_path(nei,altp[1],disa,speed_mode,vcas1,mass,rating)
        [slope,v_z2] = self.aircraft.performance.air_path(nei,altp[2],disa,speed_mode,vcas1,mass,rating)
        v_z = np.array([v_z0, v_z1, v_z2])

        if (v_z[0]<0. or v_z[1]<0. or v_z[2]<0.):
            print("Climb to acceleration altitude is not possible")

        A = vander3(altp)
        B = 1./v_z
        C = trinome(A,B)

        time1 = ((C[0]*altp[2]/3. + C[1]/2.)*altp[2] + C[2])*altp[2]
        time1 = time1 - ((C[0]*altp[0]/3. + C[1]/2.)*altp[0] + C[2])*altp[0]

        # Acceleration
        #-----------------------------------------------------------------------------------------------------------
        vc0 = vcas1
        vc2 = vcas2
        vc1 = (vc0+vc2)/2.
        vcas = np.array([vc0, vc1, vc2])

        acc0 = self.aircraft.performance.acceleration(nei,altp[2],disa,speed_mode,vcas[0],mass,rating)
        acc1 = self.aircraft.performance.acceleration(nei,altp[2],disa,speed_mode,vcas[1],mass,rating)
        acc2 = self.aircraft.performance.acceleration(nei,altp[2],disa,speed_mode,vcas[2],mass,rating)
        acc = np.array([acc0, acc1, acc2])

        if(acc[0]<0. or acc[1]<0. or acc[2]<0.):
            print("Acceleration is not possible")

        A = vander3(vcas)
        B = 1./acc
        C = trinome(A,B)

        time2 = ((C[0]*vcas[2]/3. + C[1]/2.)*vcas[2] + C[2])*vcas[2]
        time2 = time2 - ((C[0]*vcas[0]/3. + C[1]/2.)*vcas[0] + C[2])*vcas[0]

        # Duration of climb to cross over
        #-----------------------------------------------------------------------------------------------------------
        altp_0 = altp2
        altp_2 = cross_over_altp
        altp_1 = (altp_0+altp_2)/2.
        altp = np.array([altp_0, altp_1, altp_2])

        [slope,v_z0] = self.aircraft.performance.air_path(nei,altp[0],disa,speed_mode,vcas2,mass,rating)
        [slope,v_z1] = self.aircraft.performance.air_path(nei,altp[1],disa,speed_mode,vcas2,mass,rating)
        [slope,v_z2] = self.aircraft.performance.air_path(nei,altp[2],disa,speed_mode,vcas2,mass,rating)
        v_z = np.array([v_z0, v_z1, v_z2])

        if(v_z[0]<0. or v_z[1]<0. or v_z[2]<0.):
            print("Climb to cross over altitude is not possible")

        A = vander3(altp)
        B = 1./v_z
        C = trinome(A,B)

        time3 = ((C[0]*altp[2]/3. + C[1]/2.)*altp[2] + C[2])*altp[2]
        time3 = time3 - ((C[0]*altp[0]/3. + C[1]/2.)*altp[0] + C[2])*altp[0]

        # Duration of climb to altp
        #-----------------------------------------------------------------------------------------------------------
        if(cross_over_altp<toc):
            altp_0 = cross_over_altp
            altp_2 = toc
            altp_1 = (altp_0+altp_2)/2.
            altp = np.array([altp_0, altp_1, altp_2])

            speed_mode = "mach"    # mach

            [slope,v_z0] = self.aircraft.performance.air_path(nei,altp[0],disa,speed_mode,mach,mass,rating)
            [slope,v_z1] = self.aircraft.performance.air_path(nei,altp[1],disa,speed_mode,mach,mass,rating)
            [slope,v_z2] = self.aircraft.performance.air_path(nei,altp[2],disa,speed_mode,mach,mass,rating)
            v_z = np.array([v_z0, v_z1, v_z2])

            if(v_z[0]<0. or v_z[1]<0. or v_z[2]<0.):
                print("Climb to top of climb is not possible")

            A = vander3(altp)
            B = 1./v_z
            C = trinome(A,B)

            time4 =  ((C[0]*altp[2]/3. + C[1]/2.)*altp[2] + C[2])*altp[2] \
                   - ((C[0]*altp[0]/3. + C[1]/2.)*altp[0] + C[2])*altp[0]
        else:
            time4 = 0.

        #    Total time
        #-----------------------------------------------------------------------------------------------------------
        self.mass = mass
        self.ttc_eff = time1 + time2 + time3 + time4

        return


class Mission(object):
    """Definition of all mission types
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.max_payload = Mission_range_from_payload_and_tow(aircraft)
        #self.max_payload2 = Mission_generic(aircraft)
        self.nominal = Mission_fuel_from_range_and_tow(aircraft)
        self.max_fuel = Mission_range_from_fuel_and_tow(aircraft)
        self.zero_payload = Mission_range_from_fuel_and_payload(aircraft)
        self.cost = Mission_fuel_from_range_and_payload(aircraft)

    def payload_range(self):
        payload_max = self.aircraft.airframe.cabin.maximum_payload
        mtow = self.aircraft.weight_cg.mtow
        owe = self.aircraft.weight_cg.owe

        disa = self.aircraft.requirement.cruise_disa
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach

        self.max_payload.eval(payload_max,mtow,owe,altp,mach,disa)
        #self.max_payload2.eval(owe,altp,mach,disa, payload=payload_max,tow=mtow)

        range = self.aircraft.requirement.design_range
        self.nominal.eval(range,mtow,owe,altp,mach,disa)

        fuel_max = self.aircraft.weight_cg.mfw
        self.max_fuel.eval(fuel_max,mtow,owe,altp,mach,disa)

        payload = 0.
        self.zero_payload.eval(fuel_max,payload,owe,altp,mach,disa)

        range = self.aircraft.requirement.cost_range
        payload = self.aircraft.airframe.cabin.nominal_payload
        self.cost.eval(range,payload,owe,altp,mach,disa)

    def mass_mission_adaptation(self):
        """
        Build an aircraft
        """
        range = self.aircraft.requirement.design_range
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach
        disa = self.aircraft.requirement.cruise_disa

        payload = self.aircraft.airframe.cabin.nominal_payload

        def fct(mtow):
            self.aircraft.weight_cg.mtow = mtow[0]
            self.aircraft.weight_cg.mass_pre_design()
            owe = self.aircraft.weight_cg.owe
            self.nominal.eval(range,mtow,owe,altp,mach,disa)
            fuel_total = self.nominal.fuel_total
            return mtow - (owe + payload + fuel_total)

        mtow_ini = [self.aircraft.weight_cg.mtow]
        output_dict = fsolve(fct, x0=mtow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.aircraft.weight_cg.mtow = output_dict[0][0]
        self.aircraft.performance.mission.payload_range()


class Mission_def(object):
    """Defines a mission evaluation for a fuel based propulsion system (kerozen, H2 ...etc)"""
    def __init__(self,aircraft):
        # Inputs
        self.aircraft = aircraft
        self.disa = None  # Mean cruise temperature shift
        self.altp = None  # Mean cruise altitude
        self.mach = None  # Cruise mach number
        self.range = None  # Mission distance
        self.owe = None # default Operating Weight Empty
        # Outputs
        self.tow = None  # Take Off Weight
        self.payload = None  # Mission payload
        self.time_block = None  # Mission block duration
        self.fuel_block = None  # Mission block fuel consumption
        self.fuel_reserve = None  # Mission reserve fuel
        self.fuel_total = None  # Mission total fuel

        self.holding_time = unit.s_min(30)  # Holding duration
        self.reserve_fuel_ratio = self.__reserve_fuel_ratio__()  # Ratio of mission fuel to account into reserve
        self.diversion_range = self.__diversion_range__()  # Diversion leg

    def set_mission_parameters(self,mach=None, altp=None, disa=None, owe=None):
        """Set the flight condition of the mission:
            1) reset to default aircraft requirements value if no value is specified
            2) Change only one attribut if a value is specified

        :param mach: cruise Mach number
        :param altp: cruise altitude
        :param disa: mean temperature shift
        :param owe: Operating Weight empty
        """
        if mach==None and altp==None and disa==None and owe==None: # 1: reset to default
            self.mach = self.aircraft.requirement.cruise_mach
            self.altp = self.aircraft.requirement.cruise_altp
            self.disa = self.aircraft.requirement.cruise_disa
            self.owe = self.aircraft.weight_cg.owe

        else:
            if mach != None:
                self.mach = mach
            if disa != None:
                self.disa = disa
            if owe != None:
                self.owe = owe
            if altp != None:
                self.altp = altp

    def eval(self, inputs={'range':None,'tow':None}, **kwargs):
        """Solve mission equations for given inputs.
        During a mission at given cruise mach, altitude, temperature shift (disa) and Operating Weight Empty (owe)
        the four following parameters are linked
            * tow : Take-Off Weight
            * payload : weight of Payload
            * range : mission range
            * fuel_total : weight of fuel taking into account safety margins
        by two equations :
            1) fSolve mission constraint for given inputs.
        During a mission at given cruise mach, altitude, temperature shift (disa) and Operating Weight Empty (owe)
        the four following variables are linked
            * tow : Take-Off Weight
            * payload : weight of Payload
            * range : mission range
            * fuel_total : weight of fuel taking into account safety margins
        by two equations :
            1) fuel_total = eval_Breguet(range,tow, altp, mach, disa)
            2) tow - payload - fuel_total - owe = 0
        By fixing two of the previous variables, we deduce the two remaining unknowns.

        :param inputs: a dictionary of two fixed parameters. Default is {'range','tow'}
        :param kwargs: optional named parameters for set_mission_parameters
        :return: a dictionary of the two remaining unknown parameter. By default {'range':value, 'fuel_total':value}
        """
        # range,tow,altp,mach,disa
        # payload,tow,owe,altp,mach,disa
        # fuel_total,tow,owe,altp,mach,disa
        # fuel_total,payload,owe,altp,mach,disa
        # range,payload,owe,altp,mach,disa
        # range, tow, altp, mach, disa, payload, owe, fuel_total

        if len(kwargs)>0:
            self.set_mission_parameters(**kwargs)

        # Read the 2 inputs and store values in attributs
        for key,val in inputs:
            self.__dict__[key] = val

        # Build the unknown dict
        all_variables = ['range','tow','payload','fuel_total']
        unknowns = []
        for name in all_variables:
            if name not in inputs.keys():
                unknowns.append(name)
        unknowns = dict.fromkeys(unknowns) # Build an empty dict

        # TODO: implement the solve function

        self.bob
        self.range = range  # Mission distance
        self.tow = tow  # Take Off Weight
        self.eval_breguet(range, tow, altp, mach, disa)
        self.eval_payload(owe)

    def __reserve_fuel_ratio__(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range > unit.m_NM(6500.)):
            reserve_fuel_ratio = 0.03
        else:
            reserve_fuel_ratio = 0.05
        return reserve_fuel_ratio

    def __diversion_range__(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range > unit.m_NM(200.)):
            diversion_range = unit.m_NM(200.)
        else:
            diversion_range = design_range
        return diversion_range

    def eval_payload(self, owe):
        """
        Computing resulting payload
        """
        self.payload = self.tow - self.fuel_total - owe

    def __mass_equation_to_solve__(self,unknowns,**kargs): # TODO
        all_variables = ['range','tow','payload','fuel_total']

        return kwargs['tow'] - kwargs['fuel_total'] - kwargs['owe'] - kwargs['payload']

    def eval_breguet(self, range, tow, altp, mach, disa):
        """
        Mission computation using breguet equation, fixed L/D and fixed sfc
        """

        g = earth.gravity()
        fhv = self.aircraft.power_system.fuel_heat
        n_engine = self.aircraft.airframe.nacelle.n_engine
        reference_thrust = self.aircraft.airframe.nacelle.reference_thrust
        engine_bpr = self.aircraft.airframe.nacelle.engine_bpr

        # Departure ground phases
        # -----------------------------------------------------------------------------------------------------------
        fuel_taxi_out = (34. + 2.3e-4 * reference_thrust) * n_engine
        time_taxi_out = 540.

        fuel_take_off = 1e-4 * (2.8 + 2.3 / engine_bpr) * tow
        time_take_off = 220. * tow / (reference_thrust * n_engine)

        # Mission leg
        # -----------------------------------------------------------------------------------------------------------
        fuel_mission, time_mission = self.aircraft.power_system.breguet_range(range, tow, altp, mach, disa)

        mass = tow - (fuel_taxi_out + fuel_take_off + fuel_mission)

        # Arrival ground phases
        # -----------------------------------------------------------------------------------------------------------
        fuel_landing = 1e-4 * (0.5 + 2.3 / engine_bpr) * mass
        time_landing = 180.

        fuel_taxi_in = (26. + 1.8e-4 * reference_thrust) * n_engine
        time_taxi_in = 420.

        # Block fuel and time
        # -----------------------------------------------------------------------------------------------------------
        self.block_fuel = fuel_taxi_out + fuel_take_off + fuel_mission + fuel_landing + fuel_taxi_in
        self.time_block = time_taxi_out + time_take_off + time_mission + time_landing + time_taxi_in

        # Diversion fuel
        # -----------------------------------------------------------------------------------------------------------
        fuel_diversion, t = self.aircraft.power_system.breguet_range(self.diversion_range, tow, altp, mach, disa)

        # Holding fuel
        # -----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.50 * mach
        pamb, tamb, tstd, dtodz = earth.atmosphere(altp_holding, disa)
        cz, cx, lod, fn, sfc, throttle = self.aircraft.performance.level_flight(pamb, tamb, mach_holding, mass)
        fuel_holding = sfc * (mass * g / lod) * self.holding_time

        # Total
        # -----------------------------------------------------------------------------------------------------------
        self.fuel_reserve = fuel_mission * self.reserve_fuel_ratio + fuel_diversion + fuel_holding
        self.fuel_total = self.block_fuel + self.fuel_reserve

        # -----------------------------------------------------------------------------------------------------------
        return


class Mission_fuel_from_range_and_tow(object):
    """Define common features for all mission types.
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.disa = None    # Mean cruise temperature shift
        self.altp = None    # Mean cruise altitude
        self.mach = None    # Cruise mach number
        self.range = None   # Mission distance
        self.tow = None     # Take Off Weight
        self.payload = None         # Mission payload
        self.time_block = None      # Mission block duration
        self.fuel_block = None      # Mission block fuel consumption
        self.fuel_reserve = None    # Mission reserve fuel
        self.fuel_total = None      # Mission total fuel

        self.holding_time = unit.s_min(30)  # Holding duration
        self.reserve_fuel_ratio = self.__reserve_fuel_ratio__() # Ratio of mission fuel to account into reserve
        self.diversion_range = self.__diversion_range__()       # Diversion leg

    def eval(self,range,tow,owe,altp,mach,disa):
        self.range = range  # Mission distance
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = tow      # Take Off Weight
        self.eval_breguet(range,tow,altp,mach,disa)
        self.eval_payload(owe)

    def __reserve_fuel_ratio__(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range> unit.m_NM(6500.)):
            reserve_fuel_ratio = 0.03
        else:
            reserve_fuel_ratio = 0.05
        return reserve_fuel_ratio

    def __diversion_range__(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range> unit.m_NM(200.)):
            diversion_range = unit.m_NM(200.)
        else:
            diversion_range = design_range
        return diversion_range

    def eval_payload(self,owe):
        """
        Computing resulting payload
        """
        self.payload = self.tow - self.fuel_total - owe

    def eval_breguet(self,range,tow,altp,mach,disa):
        """
        Mission computation using bregueçt equation, fixed L/D and fixed sfc
        """
        g = earth.gravity()
        fhv = self.aircraft.power_system.fuel_heat
        n_engine = self.aircraft.airframe.nacelle.n_engine
        reference_thrust = self.aircraft.airframe.nacelle.reference_thrust
        engine_bpr = self.aircraft.airframe.nacelle.engine_bpr

        # Departure ground phases
        #-----------------------------------------------------------------------------------------------------------
        fuel_taxi_out = (34. + 2.3e-4*reference_thrust)*n_engine
        time_taxi_out = 540.

        fuel_take_off = 1e-4*(2.8+2.3/engine_bpr)*tow
        time_take_off = 220.*tow/(reference_thrust*n_engine)

        # Mission leg
        #-----------------------------------------------------------------------------------------------------------
        fuel_mission,time_mission = self.aircraft.power_system.breguet_range(range,tow,altp,mach,disa)

        mass = tow - (fuel_taxi_out + fuel_take_off + fuel_mission)

        # Arrival ground phases
        #-----------------------------------------------------------------------------------------------------------
        fuel_landing = 1e-4*(0.5+2.3/engine_bpr)*mass
        time_landing = 180.

        fuel_taxi_in = (26. + 1.8e-4*reference_thrust)*n_engine
        time_taxi_in = 420.

        # Block fuel and time
        #-----------------------------------------------------------------------------------------------------------
        self.block_fuel = fuel_taxi_out + fuel_take_off + fuel_mission + fuel_landing + fuel_taxi_in
        self.time_block = time_taxi_out + time_take_off + time_mission + time_landing + time_taxi_in

        # Diversion fuel
        #-----------------------------------------------------------------------------------------------------------
        fuel_diversion,t = self.aircraft.power_system.breguet_range(self.diversion_range,tow,altp,mach,disa)

        # Holding fuel
        #-----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.50 * mach
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp_holding,disa)
        cz,cx,lod,fn,sfc,throttle = self.aircraft.performance.level_flight(pamb,tamb,mach_holding,mass)
        fuel_holding = sfc*(mass*g/lod)*self.holding_time

        # Total
        #-----------------------------------------------------------------------------------------------------------
        self.fuel_reserve = fuel_mission*self.reserve_fuel_ratio + fuel_diversion + fuel_holding
        self.fuel_total = self.block_fuel + self.fuel_reserve

        #-----------------------------------------------------------------------------------------------------------
        return


class Mission_range_from_payload_and_tow(Mission_fuel_from_range_and_tow):
    """Specific mission evaluation from payload and take off weight
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and total_fuel from payload and tow
    """
    def __init__(self, aircraft):
        super(Mission_range_from_payload_and_tow, self).__init__(aircraft)

    def eval(self,payload,tow,owe,altp,mach,disa):
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = tow      # Take Off Weight
        self.payload = payload  # Mission payload

        def fct(range):
            self.eval_breguet(range,tow,altp,mach,disa)
            return  self.tow - (owe+self.payload+self.fuel_total)

        range_ini = [self.aircraft.requirement.design_range]
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self.range,tow,altp,mach,disa)


class Mission_range_from_fuel_and_tow(Mission_fuel_from_range_and_tow):
    """Specific mission evaluation from total fuel and take off weight
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and payload from total_fuel and tow
    """
    def __init__(self, aircraft):
        super(Mission_range_from_fuel_and_tow, self).__init__(aircraft)

    def eval(self,fuel_total,tow,owe,altp,mach,disa):
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = tow      # Take Off Weight
        self.payload = tow-owe-fuel_total  # Mission payload

        def fct(range):
            self.eval_breguet(range,tow,altp,mach,disa)
            return  self.tow - (owe+self.payload+self.fuel_total)

        range_ini = [self.aircraft.requirement.design_range]
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self.range,tow,altp,mach,disa)


class Mission_range_from_fuel_and_payload(Mission_fuel_from_range_and_tow):
    """Specific mission evaluation from payload and total fuel
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and tow from total_fuel and payload
    """
    def __init__(self, aircraft):
        super(Mission_range_from_fuel_and_payload, self).__init__(aircraft)

    def eval(self,fuel_total,payload,owe,altp,mach,disa):
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = owe+payload+fuel_total     # Take Off Weight
        self.payload = payload  # Mission payload

        def fct(range):
            self.eval_breguet(range,self.tow,altp,mach,disa)
            return  self.tow - (owe+self.payload+self.fuel_total)

        range_ini = [self.aircraft.requirement.design_range]
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self.range,self.tow,altp,mach,disa)


class Mission_fuel_from_range_and_payload(Mission_fuel_from_range_and_tow):
    """Specific mission evaluation from range and payload
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes total_fuel and tow from range and payload
    """
    def __init__(self, aircraft):
        super(Mission_fuel_from_range_and_payload, self).__init__(aircraft)

    def eval(self,range,payload,owe,altp,mach,disa):
        self.range = range     # Take Off Weight
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.payload = payload  # Mission payload

        def fct(tow):
            self.eval_breguet(range,tow,altp,mach,disa)
            return  tow - (owe+self.payload+self.fuel_total)

        tow_ini = [self.aircraft.weight_cg.mtow]
        output_dict = fsolve(fct, x0=tow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.tow = output_dict[0][0]
        self.eval_breguet(self.range,self.tow,altp,mach,disa)




class Mission_generic(Mission_fuel_from_range_and_tow):
    """Specific mission evaluation from payload and take off weight
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and total_fuel from payload and tow
    """
    def __init__(self, aircraft):
        super(Mission_generic, self).__init__(aircraft)

    def eval(self,owe,altp,mach,disa,**kwargs):
        """Generic mission solver
        kwargs must contain affectations to the parameters that are fixed
        among the following list : range, tow, payload, fuel_total
        """
        vars = list(set(["range","tow","payload","fuel_total"])-set(kwargs.keys())) # extract variable names
#        (range,tow,payload,fuel_total) = [0.,0.,0.,0.]                              # initialize all variables
        for key,val in kwargs.items():      # load parameter values, this quantities will not be modified
            if (key=="fuel_total"): fuel_total = val
            elif (key=="payload"): payload = val
            elif (key=="range"): range = val
            elif (key=="tow"): tow = val

        def fct(x_in):
            for k,key in enumerate(vars):      # load variable values
                if (key=="fuel_total"): fuel_total = x_in[k]
                elif (key=="payload"): payload = x_in[k]
                elif (key=="range"): range = x_in[k]
                elif (key=="tow"): tow = x_in[k]
            self.eval_breguet(range,tow,altp,mach,disa)         # eval Breguet equation, fuel_total is updated in the object
            return  [self.fuel_total - fuel_total,
                     self.tow - (owe+payload+self.fuel_total)]  # constraints residuals are sent back

        x_ini = np.zeros(2)
        for k,key in enumerate(vars):              # load init values from object
            if (key=="fuel_total"): x_ini[k] = self.fuel_total
            elif (key=="payload"): x_ini[k] = self.payload
            elif (key=="range"): x_ini[k] = self.range
            elif (key=="tow"): x_ini[k] = self.tow
        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        for k,key in enumerate(vars):              # get solution
            if (key=="fuel_total"): fuel_total = output_dict[0][k]
            elif (key=="payload"): payload = output_dict[0][k]
            elif (key=="range"): range = output_dict[0][k]
            elif (key=="tow"): tow = output_dict[0][k]
        self.eval_breguet(range,tow,altp,mach,disa)

