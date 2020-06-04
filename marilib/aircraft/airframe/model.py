#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.optimize import fsolve

from marilib.utils import earth

from marilib.aircraft.performance import Flight

from marilib.utils.math import lin_interp_1d, maximize_1d


# -----------------------------------------------------------------------------------
#                            AERODYNAMIC & WEIGHTS
# -----------------------------------------------------------------------------------

class Aerodynamics(object):

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.cx_correction = 0.     # drag correction on cx coefficient
        self.cruise_lodmax = 16.    # Assumption on L/D max for some initializations
        self.cz_cruise_lodmax = None

        self.hld_conf_clean = 0.
        self.czmax_conf_clean = None

        self.hld_conf_to = 0.30
        self.czmax_conf_to = None

        self.hld_conf_ld = 1.00
        self.czmax_conf_ld = None

    def aerodynamic_analysis(self):
        mach = self.aircraft.requirement.cruise_mach
        altp = self.aircraft.requirement.cruise_altp
        disa = 0.
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        self.cruise_lodmax, self.cz_cruise_lodmax = self.lod_max(pamb,tamb,mach)
        self.czmax_conf_clean,Cz0 = self.aircraft.airframe.wing.high_lift(self.hld_conf_clean)
        self.czmax_conf_to,Cz0 = self.aircraft.airframe.wing.high_lift(self.hld_conf_to)
        self.czmax_conf_ld,Cz0 = self.aircraft.airframe.wing.high_lift(self.hld_conf_ld)

    def drag(self,pamb,tamb,mach,cz):
        """Retrieves airplane drag and L/D in current flying conditions
        """
        # Form & friction drag
        #-----------------------------------------------------------------------------------------------------------
        re = earth.reynolds_number(pamb, tamb, mach)

        fac = ( 1. + 0.126*mach**2 )

        ac_nwa = 0.
        cxf = 0.
        for comp in self.aircraft.airframe:
            nwa = comp.get_net_wet_area()
            ael = comp.get_aero_length()
            frm = comp.get_form_factor()
            cxf += frm*((0.455/fac)*(np.log(10)/np.log(re*ael))**2.58 ) * (nwa/self.aircraft.airframe.wing.area)
            ac_nwa += nwa

        # Parasitic drag (seals, antennas, sensors, ...)
        #-----------------------------------------------------------------------------------------------------------
        knwa = ac_nwa/1000.

        kp = (0.0247*knwa - 0.11)*knwa + 0.166       # Parasitic drag factor

        cx_par = cxf*kp

        # Additional drag
        #-----------------------------------------------------------------------------------------------------------
        X = np.array([1.0, 1.5, 2.4, 3.3, 4.0, 5.0])
        Y = np.array([0.036, 0.020, 0.0075, 0.0025, 0., 0.])

        param = self.aircraft.airframe.body.tail_cone_length/self.aircraft.airframe.body.width

        cx_tap_base = lin_interp_1d(param,X,Y)     # Tapered fuselage drag (tail cone)

        cx_tap = cx_tap_base*self.aircraft.power_system.tail_cone_drag_factor()     # Effect of tail cone fan

        # Total zero lift drag
        #-----------------------------------------------------------------------------------------------------------
        cx0 = cxf + cx_par + cx_tap + self.cx_correction

        # Induced drag
        #-----------------------------------------------------------------------------------------------------------
        cxi = self.aircraft.airframe.wing.induced_drag_factor*cz**2  # Induced drag

        # Compressibility drag
        #-----------------------------------------------------------------------------------------------------------
        # Freely inspired from Korn equation
        cz_design = 0.5
        mach_div = self.aircraft.requirement.cruise_mach + (0.03 + 0.1*(cz_design-cz))

        cxc = 0.0025 * np.exp(40.*(mach - mach_div) )

        # Sum up
        #-----------------------------------------------------------------------------------------------------------
        cx = cx0 + cxi + cxc
        lod = cz/cx

        return cx,lod

    def lod_max(self,pamb,tamb,mach):
        """Maximum lift to drag ratio
        """
        def fct(cz):
            cx,lod = self.drag(pamb,tamb,mach,cz)
            return lod

        cz_ini = 0.5
        dcz = 0.05
        cz_lodmax,lodmax,rc = maximize_1d(cz_ini,dcz,[fct])

        return lodmax,cz_lodmax


class OweBreakdown(object):

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.owe = None
        self.op_item_mass = None
        self.container_pallet_mass = None
        self.mwe = None
        self.furnishing_mass = None
        self.wing_mass = None
        self.body_mass = None
        self.htp_mass = None
        self.vtp_mass = None
        self.tank_mass = None
        self.ldg_mass = None
        self.system_mass = None
        self.propeller_mass = None
        self.engine_mass = None
        self.pylon_mass = None


class WeightCg(object):

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.mtow = self.__mtow_init__()
        self.mzfw = self.__mzfw_init__()
        self.mlw = self.__mlw_init__()
        self.owe = None
        self.mwe = None
        self.mfw = None

        self.breakdown = OweBreakdown(aircraft)

    def __mtow_init__(self):
        return 20500. + 67.e-6*self.aircraft.requirement.n_pax_ref*self.aircraft.requirement.design_range

    def __mzfw_init__(self):
        return 25000. + 41.e-6*self.aircraft.requirement.n_pax_ref*self.aircraft.requirement.design_range

    def __mlw_init__(self):
        return 1.07*(25000. + 41.e-6*self.aircraft.requirement.n_pax_ref*self.aircraft.requirement.design_range)

    def mass_analysis(self):
        """Update all component mass
        """
        for comp in self.aircraft.airframe.mass_iter():
            comp.eval_mass()

        # sum all MWE & OWE contributions
        mwe = 0.
        owe = 0.
        for comp in self.aircraft.airframe.mass_iter():
            mwe += comp.get_mass_mwe()
            owe += comp.get_mass_owe()
        self.mwe = mwe
        self.owe = owe

        self.breakdown.owe = self.owe
        self.breakdown.op_item_mass = self.aircraft.airframe.cabin.m_op_item
        self.breakdown.container_pallet_mass = self.aircraft.airframe.cargo.mass
        self.breakdown.mwe = self.mwe
        self.breakdown.furnishing_mass = self.aircraft.airframe.cabin.m_furnishing
        self.breakdown.wing_mass = self.aircraft.airframe.wing.mass
        self.breakdown.body_mass = self.aircraft.airframe.body.mass
        self.breakdown.htp_mass = self.aircraft.airframe.horizontal_stab.mass
        self.breakdown.vtp_mass = self.aircraft.airframe.vertical_stab.mass
        self.breakdown.tank_mass = self.aircraft.airframe.tank.mass
        self.breakdown.ldg_mass = self.aircraft.airframe.landing_gear.mass
        self.breakdown.system_mass = self.aircraft.airframe.system.mass
        self.breakdown.propeller_mass = self.aircraft.airframe.nacelle.propeller_mass
        self.breakdown.engine_mass = self.aircraft.airframe.nacelle.engine_mass
        self.breakdown.pylon_mass = self.aircraft.airframe.nacelle.pylon_mass

        if (self.aircraft.arrangement.power_source=="battery"):
            self.mzfw = self.mtow
        else:
            self.mzfw = self.owe + self.aircraft.airframe.cabin.maximum_payload

        if (self.aircraft.arrangement.power_source=="battery"):
            self.mlw = self.mtow
        else:
            if (self.aircraft.requirement.n_pax_ref>100):
                self.mlw = min(self.mtow , (1.07*self.mzfw))
            else:
                self.mlw = self.mtow

        # WARNING : for battery powered architecture, MFW corresponds to max battery weight
        self.mfw = min(self.aircraft.airframe.tank.mfw_volume_limited, self.mtow - self.owe)

        # TODO
        # calculer les cg

    def mass_pre_design(self):
        """Solve the coupling through MZFW & MLW for a given mtow
        """
        def fct(x_in):
            self.aircraft.weight_cg.mzfw = x_in[0]
            self.aircraft.weight_cg.mlw = x_in[1]

            self.mass_analysis()

            y_out = np.array([x_in[0] - self.aircraft.weight_cg.mzfw,
                              x_in[1] - self.aircraft.weight_cg.mlw])
            return y_out

        x_ini = np.array([self.aircraft.weight_cg.mzfw,
                          self.aircraft.weight_cg.mlw])

        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.aircraft.weight_cg.mzfw = output_dict[0][0]        # Coupling variable
        self.aircraft.weight_cg.mlw = output_dict[0][1]         # Coupling variable

        self.mass_analysis()


# -----------------------------------------------------------------------------------
#                            POWER SYSTEM
# -----------------------------------------------------------------------------------

class ThrustData(object):
    def __init__(self, nei=0):
        self.disa = None
        self.altp = None
        self.mach = None
        self.nei = nei
        self.kfn_opt = None

class PowerSystem(object):
    """A generic class that describes a power system."""

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.data = {"MTO":ThrustData(nei=1),
                     "MCN":ThrustData(nei=1),
                     "MCL":ThrustData(nei=0),
                     "MCR":ThrustData(nei=0)}

    def thrust_requirement(self):
        self.data["MTO"].disa = self.aircraft.performance.take_off.disa
        self.data["MTO"].altp = self.aircraft.performance.take_off.altp
        self.data["MTO"].mach = self.aircraft.performance.take_off.mach2

        fct = self.aircraft.performance.take_off.thrust_opt
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.data["MTO"].kfn_opt = output_dict[0][0]

        self.data["MCN"].disa = self.aircraft.performance.oei_ceiling.disa
        self.data["MCN"].altp = self.aircraft.performance.oei_ceiling.altp
        self.data["MCN"].mach = self.aircraft.performance.oei_ceiling.mach_opt

        fct = self.aircraft.performance.oei_ceiling.thrust_opt
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.data["MCN"].kfn_opt = output_dict[0][0]

        self.data["MCL"].disa = self.aircraft.performance.mcl_ceiling.disa
        self.data["MCL"].altp = self.aircraft.performance.mcl_ceiling.altp
        self.data["MCL"].mach = self.aircraft.performance.mcl_ceiling.mach

        fct = self.aircraft.performance.mcl_ceiling.thrust_opt
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.data["MCL"].kfn_opt = output_dict[0][0]

        self.data["MCR"].disa = self.aircraft.performance.mcr_ceiling.disa
        self.data["MCR"].altp = self.aircraft.performance.mcr_ceiling.altp
        self.data["MCR"].mach = self.aircraft.performance.mcr_ceiling.mach

        fct = self.aircraft.performance.mcr_ceiling.thrust_opt
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.data["MCR"].kfn_opt = output_dict[0][0]

    def thrust_analysis(self):
        raise NotImplementedError

    def thrust(self,pamb,tamb,mach,rating,throttle,nei):
        raise NotImplementedError

    def sc(self,pamb,tamb,mach,rating,thrust,nei):
        raise NotImplementedError

    def oei_drag(self,pamb,tamb):
        raise NotImplementedError

    def tail_cone_drag_factor(self):
        raise NotImplementedError

    def specific_air_range(self,mass,tas,dict):
        raise NotImplementedError

    def specific_breguet_range(self,tow,range,tas,dict):
        raise NotImplementedError

    def specific_holding(self,mass,time,tas,dict):
        raise NotImplementedError


class ThrustDataTf(ThrustData):
    def __init__(self, nei):
        super(ThrustDataTf, self).__init__(nei)
        self.thrust_opt = None
        self.thrust = None
        self.fuel_flow = None
        self.tsfc = None
        self.T41 = None

class Turbofan(PowerSystem, Flight):

    def __init__(self, aircraft):
        super(Turbofan, self).__init__(aircraft)

        self.sfc_type = "thrust"
        self.data = {"MTO":ThrustDataTf(nei=1),
                     "MCN":ThrustDataTf(nei=1),
                     "MCL":ThrustDataTf(nei=0),
                     "MCR":ThrustDataTf(nei=0)}

    def update_power_transfert(self):
        pass

    def thrust_analysis(self):
        self.thrust_requirement()
        for rating in self.data.keys():
            disa = self.data[rating].disa
            altp = self.data[rating].altp
            mach = self.data[rating].mach
            nei = self.data[rating].nei
            kfn = self.data[rating].kfn_opt
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
            dict = self.thrust(pamb,tamb,mach,rating, nei=nei)
            self.data[rating].thrust_opt = kfn*dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].thrust = dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].fuel_flow = dict["ff"]
            self.data[rating].tsfc = dict["sfc"]
            self.data[rating].T41 = dict["t4"]

    def thrust(self,pamb,tamb,mach,rating, throttle=1., nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fuel_type = self.aircraft.arrangement.fuel_type
        fuel_heat = earth.fuel_heat(fuel_type)

        dict = self.aircraft.airframe.nacelle.unitary_thrust(pamb,tamb,mach,rating,throttle=throttle)

        fn = dict["fn"]*(n_engine-nei)
        ff = dict["ff"]*(n_engine-nei) * earth.fuel_heat("kerosene") / fuel_heat
        sfc = ff / fn
        t41 = dict["t4"]

        return {"fn":fn, "ff":ff, "sfc":sfc, "t4":t41}

    def sc(self,pamb,tamb,mach,rating, thrust, nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fuel_type = self.aircraft.arrangement.fuel_type
        fuel_heat = earth.fuel_heat(fuel_type)

        fn = thrust/(n_engine - nei)

        dict = self.aircraft.airframe.nacelle.unitary_sc(pamb,tamb,mach,rating,fn)
        dict["sfc"] = dict["sfc"] * earth.fuel_heat("kerosene") / fuel_heat

        return dict

    def oei_drag(self,pamb,tamb):
        """Inoperative engine drag coefficient
        """
        wing_area = self.aircraft.airframe.wing.area
        nacelle_width = self.aircraft.airframe.nacelle.width

        dCx = 0.12*nacelle_width**2 / wing_area

        return dCx

    def tail_cone_drag_factor(self):
        return 1.

    def specific_air_range(self,mass,tas,dict):
        g = earth.gravity()
        return (tas*dict["lod"])/(mass*g*dict["sfc"])

    def specific_breguet_range(self,tow,range,tas,dict):
        g = earth.gravity()
        return tow*(1-np.exp(-(dict["sfc"]*g*range)/(tas*dict["lod"])))

    def specific_holding(self,mass,time,tas,dict):
        g = earth.gravity()
        return dict["sfc"]*(mass*g/dict["lod"])*time

class ThrustDataTp(ThrustData):
    def __init__(self, nei):
        super(ThrustDataTp, self).__init__(nei)
        self.thrust_opt = None
        self.thrust = None
        self.power = None
        self.fuel_flow = None
        self.psfc = None
        self.T41 = None

class Turboprop(PowerSystem, Flight):

    def __init__(self, aircraft):
        super(Turboprop, self).__init__(aircraft)

        self.sfc_type = "power"
        self.data = {"MTO":ThrustDataTp(nei=1),
                     "MCN":ThrustDataTp(nei=1),
                     "MCL":ThrustDataTp(nei=0),
                     "MCR":ThrustDataTp(nei=0)}

    def reference_power_offtake(self):
        return 0.

    def thrust_analysis(self):
        self.thrust_requirement()
        for rating in self.data.keys():
            disa = self.data[rating].disa
            altp = self.data[rating].altp
            mach = self.data[rating].mach
            nei = self.data[rating].nei
            kfn = self.data[rating].kfn_opt
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
            dict = self.thrust(pamb,tamb,mach,rating, nei=nei)
            self.data[rating].thrust_opt = kfn*dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].thrust = dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].power = dict["pw"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].fuel_flow = dict["ff"]
            self.data[rating].psfc = dict["sfc"]
            self.data[rating].T41 = dict["t4"]

    def thrust(self,pamb,tamb,mach,rating, throttle=1., nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fuel_type = self.aircraft.arrangement.fuel_type
        fuel_heat = earth.fuel_heat(fuel_type)

        dict = self.aircraft.airframe.nacelle.unitary_thrust(pamb,tamb,mach,rating,throttle=throttle)

        fn = dict["fn"]*(n_engine-nei)
        ff = dict["ff"]*(n_engine-nei) * earth.fuel_heat("kerosene") / fuel_heat
        pw = dict["pw"]*(n_engine-nei)
        sfc = ff / pw
        t41 = dict["t4"]

        return {"fn":fn, "ff":ff, "pw":pw, "sfc":sfc, "t4":t41}

    def sc(self,pamb,tamb,mach,rating, thrust, nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fuel_type = self.aircraft.arrangement.fuel_type
        fuel_heat = earth.fuel_heat(fuel_type)

        fn = thrust/(n_engine - nei)

        dict = self.aircraft.airframe.nacelle.unitary_sc(pamb,tamb,mach,rating,fn)
        dict["sfc"] = dict["sfc"] * earth.fuel_heat("kerosene") / fuel_heat

        return dict

    def oei_drag(self,pamb,tamb):
        """Inoperative engine drag coefficient
        """
        wing_area = self.aircraft.airframe.wing.area
        nacelle_width = self.aircraft.airframe.nacelle.width

        dCx = 1.15*nacelle_width**2 / wing_area

        return dCx

    def tail_cone_drag_factor(self):
        return 1.

    def specific_air_range(self,mass,tas,dict):
        g = earth.gravity()
        eta_prop = self.aircraft.airframe.nacelle.propeller_efficiency
        return (eta_prop*dict["lod"])/(mass*g*dict["sfc"])

    def specific_breguet_range(self,tow,range,tas,dict):
        g = earth.gravity()
        eta_prop = self.aircraft.airframe.nacelle.propeller_efficiency
        return tow*(1.-np.exp(-(dict["sfc"]*g*range)/(eta_prop*dict["lod"])))

    def specific_holding(self,mass,time,tas,dict):
        g = earth.gravity()
        eta_prop = self.aircraft.airframe.nacelle.propeller_efficiency
        return ((mass*g*tas*dict["sfc"])/(eta_prop*dict["lod"]))*time


class ThrustDataEp(ThrustData):
    def __init__(self, aircraft, nei):
        super(ThrustDataEp, self).__init__(nei)
        self.thrust_opt = None
        self.thrust = None
        self.power = None
        self.sec = None
        if (aircraft.arrangement.power_source == "fuel_cell"):
            self.psfc = None

class Electroprop(PowerSystem, Flight):

    def __init__(self, aircraft):
        super(Electroprop, self).__init__(aircraft)

        if (self.aircraft.arrangement.power_source == "fuel_cell"):
            self.sfc_type = "power"

        self.data = {"MTO":ThrustDataEp(aircraft, nei=1),
                     "MCN":ThrustDataEp(aircraft, nei=1),
                     "MCL":ThrustDataEp(aircraft, nei=0),
                     "MCR":ThrustDataEp(aircraft, nei=0)}

    def thrust_analysis(self):
        self.thrust_requirement()
        for rating in self.data.keys():
            disa = self.data[rating].disa
            altp = self.data[rating].altp
            mach = self.data[rating].mach
            nei = self.data[rating].nei
            kfn = self.data[rating].kfn_opt
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
            dict = self.thrust(pamb,tamb,mach,rating, nei=nei)
            self.data[rating].thrust_opt = kfn*dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].thrust = dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].power = dict["pw"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].sec = dict["sec"]
            if (self.aircraft.arrangement.power_source == "fuel_cell"):
                self.data[rating].psfc = dict["sfc"]

    def thrust(self,pamb,tamb,mach,rating, throttle=1., nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fuel_type = self.aircraft.arrangement.fuel_type
        fuel_heat = earth.fuel_heat(fuel_type)

        dict = self.aircraft.airframe.nacelle.unitary_thrust(pamb,tamb,mach,rating,throttle=throttle)

        fn = dict["fn"]*(n_engine-nei)
        pw = dict["pw"]*(n_engine-nei)
        pw_net = pw / (self.aircraft.airframe.system.wiring_efficiency * self.aircraft.airframe.system.cooling_efficiency)
        sec = pw_net / fn

        dict = {"fn":fn, "pw":pw_net, "sec":sec}

        if (self.aircraft.arrangement.power_source == "fuel_cell"):
            dict["sfc"] = 1. / (self.aircraft.airframe.system.power_chain_efficiency * self.aircraft.airframe.system.fuel_cell_efficiency * fuel_heat)

        return dict

    def sc(self,pamb,tamb,mach,rating, thrust, nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fuel_type = self.aircraft.arrangement.fuel_type
        fuel_heat = earth.fuel_heat(fuel_type)
        fn = thrust/(n_engine - nei)

        dict = self.aircraft.airframe.nacelle.unitary_sc(pamb,tamb,mach,rating,fn)

        dict["sec"] = dict["sec"] / (self.aircraft.airframe.system.wiring_efficiency * self.aircraft.airframe.system.cooling_efficiency)

        if (self.aircraft.arrangement.power_source == "fuel_cell"):
            dict["sfc"] = 1. / (self.aircraft.airframe.system.power_chain_efficiency * self.aircraft.airframe.system.fuel_cell_efficiency * fuel_heat)

        return dict

    def oei_drag(self,pamb,tamb):
        """Inoperative engine drag coefficient
        """
        wing_area = self.aircraft.airframe.wing.area
        nacelle_width = self.aircraft.airframe.nacelle.width

        dCx = 1.15*nacelle_width**2 / wing_area

        return dCx

    def tail_cone_drag_factor(self):
        return 1.

    def specific_air_range(self,mass,tas,dict):
        g = earth.gravity()
        if (self.aircraft.arrangement.power_source == "battery"):
            return (tas*dict["lod"]) / (mass*g*dict["sec"])
        elif (self.aircraft.arrangement.power_source == "fuel_cell"):
            eta_prop = self.aircraft.airframe.nacelle.propeller_efficiency
            return (eta_prop*dict["lod"])/(mass*g*dict["sfc"])
        else:
            raise Exception("Power source is unknown")

    def specific_breguet_range(self,tow,range,tas,dict):
        if (self.aircraft.arrangement.power_source == "battery"):
            return range / self.specific_air_range(tow,tas,dict)
        elif (self.aircraft.arrangement.power_source == "fuel_cell"):
            g = earth.gravity()
            eta_prop = self.aircraft.airframe.nacelle.propeller_efficiency
            return tow*(1.-np.exp(-(dict["sfc"]*g*range)/(eta_prop*dict["lod"])))
        else:
            raise Exception("Power source is unknown")

    def specific_holding(self,mass,time,tas,dict):
        g = earth.gravity()
        if (self.aircraft.arrangement.power_source == "battery"):
            return dict["sec"]*(mass*g/dict["lod"])*time
        elif (self.aircraft.arrangement.power_source == "fuel_cell"):
            eta_prop = self.aircraft.airframe.nacelle.propeller_efficiency
            return ((mass*g*tas*dict["sfc"])/(eta_prop*dict["lod"]))*time
        else:
            raise Exception("Power source is unknown")


class ThrustDataEf(ThrustData):
    def __init__(self, aircraft, nei):
        super(ThrustDataEf, self).__init__(nei)
        self.thrust_opt = None
        self.thrust = None
        self.power = None
        self.sec = None
        if (aircraft.arrangement.power_source == "fuel_cell"):
            self.tsfc = None

class Electrofan(PowerSystem, Flight):

    def __init__(self, aircraft):
        super(Electrofan, self).__init__(aircraft)

        if (self.aircraft.arrangement.power_source == "fuel_cell"):
            self.sfc_type = "thrust"

        self.data = {"MTO":ThrustDataEf(aircraft, nei=1),
                     "MCN":ThrustDataEf(aircraft, nei=1),
                     "MCL":ThrustDataEf(aircraft, nei=0),
                     "MCR":ThrustDataEf(aircraft, nei=0)}

    def thrust_analysis(self):
        self.thrust_requirement()
        for rating in self.data.keys():
            disa = self.data[rating].disa
            altp = self.data[rating].altp
            mach = self.data[rating].mach
            nei = self.data[rating].nei
            kfn = self.data[rating].kfn_opt
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
            dict = self.thrust(pamb,tamb,mach,rating, nei=nei)
            self.data[rating].thrust_opt = kfn*dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].thrust = dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].power = dict["pw"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].sec = dict["sec"]
            if (self.aircraft.arrangement.power_source == "fuel_cell"):
                self.data[rating].tsfc = dict["sfc"]

    def thrust(self,pamb,tamb,mach,rating, throttle=1., nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fuel_type = self.aircraft.arrangement.fuel_type
        fuel_heat = earth.fuel_heat(fuel_type)

        dict = self.aircraft.airframe.nacelle.unitary_thrust(pamb,tamb,mach,rating,throttle=throttle)

        fn = dict["fn"]*(n_engine-nei)
        pw = dict["pw"]*(n_engine-nei)
        pw_net = pw / (self.aircraft.airframe.system.wiring_efficiency * self.aircraft.airframe.system.cooling_efficiency)
        sec = pw_net / fn

        dict = {"fn":fn, "pw":pw_net, "sec":sec}

        if (self.aircraft.arrangement.power_source == "fuel_cell"):
            dict["sfc"] = sec / (self.aircraft.airframe.system.fuel_cell_efficiency * fuel_heat)

        return dict

    def sc(self,pamb,tamb,mach,rating, thrust, nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fuel_type = self.aircraft.arrangement.fuel_type
        fuel_heat = earth.fuel_heat(fuel_type)
        fn = thrust/(n_engine - nei)

        dict = self.aircraft.airframe.nacelle.unitary_sc(pamb,tamb,mach,rating,fn)

        dict["sec"] = dict["sec"] / (self.aircraft.airframe.system.wiring_efficiency * self.aircraft.airframe.system.cooling_efficiency)

        if (self.aircraft.arrangement.power_source == "fuel_cell"):
            dict["sfc"] = dict["sec"] / (self.aircraft.airframe.system.fuel_cell_efficiency * fuel_heat)

        return dict

    def oei_drag(self,pamb,tamb):
        """Inoperative engine drag coefficient
        """
        wing_area = self.aircraft.airframe.wing.area
        nacelle_width = self.aircraft.airframe.nacelle.width

        dCx = 0.12*nacelle_width**2 / wing_area

        return dCx

    def tail_cone_drag_factor(self):
        return 1.

    def specific_air_range(self,mass,tas,dict):
        g = earth.gravity()
        if (self.aircraft.arrangement.power_source == "battery"):
            return (tas*dict["lod"])/(mass*g*dict["sec"])
        elif (self.aircraft.arrangement.power_source == "fuel_cell"):
            return (tas*dict["lod"])/(mass*g*dict["sfc"])
        else:
            raise Exception("Power source is unknown")

    def specific_breguet_range(self,tow,range,tas,dict):
        if (self.aircraft.arrangement.power_source == "battery"):
            return range / self.specific_air_range(tow,tas,dict)
        elif (self.aircraft.arrangement.power_source == "fuel_cell"):
            g = earth.gravity()
            return tow*(1-np.exp(-(dict["sfc"]*g*range)/(tas*dict["lod"])))
        else:
            raise Exception("Power source is unknown")

    def specific_holding(self,mass,time,tas,dict):
        g = earth.gravity()
        if (self.aircraft.arrangement.power_source == "battery"):
            return dict["sec"]*(mass*g/dict["lod"])*time
        elif (self.aircraft.arrangement.power_source == "fuel_cell"):
            return dict["sfc"]*(mass*g/dict["lod"])*time
        else:
            raise Exception("Power source is unknown")


class ThrustDataPte(ThrustData):
    def __init__(self, nei):
        super(ThrustDataTf, self).__init__(nei)
        self.thrust_opt = None
        self.thrust = None
        self.fuel_flow = None
        self.tsfc = None
        self.T41 = None

class PartialTurboElectric(PowerSystem, Flight):

    def __init__(self, aircraft):
        super(PartialTurboElectric, self).__init__(aircraft)

        self.sfc_type = "thrust"
        self.data = {"MTO":ThrustDataTf(nei=1),
                     "MCN":ThrustDataTf(nei=1),
                     "MCL":ThrustDataTf(nei=0),
                     "MCR":ThrustDataTf(nei=0)}

    def update_power_transfert(self):
        ref_power = self.aircraft.airframe.system.chain_power
        power_chain_efficiency =   self.aircraft.airframe.system.generator_efficiency \
                                 * self.aircraft.airframe.system.rectifier_efficiency \
                                 * self.aircraft.airframe.system.wiring_efficiency \
                                 * self.aircraft.airframe.tail_nacelle.controller_efficiency \
                                 * self.aircraft.airframe.tail_nacelle.motor_efficiency
        n_engine = self.aircraft.airframe.nacelle.n_engine
        self.aircraft.airframe.tail_nacelle.reference_power = ref_power
        self.aircraft.airframe.nacelle.reference_offtake = ref_power/power_chain_efficiency/n_engine

    def thrust_analysis(self):
        self.thrust_requirement()
        for rating in self.data.keys():
            disa = self.data[rating].disa
            altp = self.data[rating].altp
            mach = self.data[rating].mach
            nei = self.data[rating].nei
            kfn = self.data[rating].kfn_opt
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
            dict = self.thrust(pamb,tamb,mach,rating, nei=nei)
            self.data[rating].thrust_opt = kfn*dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].thrust = dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].fuel_flow = dict["ff"]
            self.data[rating].tsfc = dict["sfc"]
            self.data[rating].T41 = dict["t4"]

    def thrust(self,pamb,tamb,mach,rating, throttle=1., nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fuel_type = self.aircraft.arrangement.fuel_type
        fuel_heat = earth.fuel_heat(fuel_type)

        dict_ef = self.aircraft.airframe.tail_nacelle.unitary_thrust(pamb,tamb,mach,rating)

        pw_elec = dict_ef["pw"]
        pw_offtake =    pw_elec \
                     / self.aircraft.airframe.system.wiring_efficiency \
                     / self.aircraft.airframe.system.rectifier_efficiency \
                     / self.aircraft.airframe.system.generator_efficiency \
                     / (n_engine - nei)

        dict_tf = self.aircraft.airframe.nacelle.unitary_thrust(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

        fn = dict_tf["fn"]*(n_engine-nei) + dict_ef["fn"]*self.aircraft.airframe.tail_nacelle.n_engine
        ff = dict_tf["ff"]*(n_engine-nei) * earth.fuel_heat("kerosene") / fuel_heat
        sfc = ff / fn
        t41 = dict_tf["t4"]
        efn = dict_ef["fn"]
        epw = dict_ef["pw"]

        return {"fn":fn, "ff":ff, "sfc":sfc, "t4":t41, "efn":efn, "epw":epw}

    def sc(self,pamb,tamb,mach,rating, thrust, nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fn = thrust/(n_engine - nei)

        def fct(thtl):
            dict = self.thrust(pamb,tamb,mach,rating, throttle=thtl, nei=nei)
            return thrust-dict["fn"]

        thtl_ini = 0.9
        output_dict = fsolve(fct, x0=thtl_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        throttle = output_dict[0][0]

        dict = self.thrust(pamb,tamb,mach,rating, throttle=throttle, nei=nei)
        sfc = dict["sfc"]
        t41 = dict["t4"]
        efn = dict["efn"]
        epw = dict["epw"]

        return {"sfc":sfc, "thtl":throttle, "t4":t41, "efn":efn, "epw":epw}

    def oei_drag(self,pamb,tamb):
        """Inoperative engine drag coefficient
        """
        wing_area = self.aircraft.airframe.wing.area
        nacelle_width = self.aircraft.airframe.nacelle.width

        dCx = 0.12*nacelle_width**2 / wing_area

        return dCx

    def tail_cone_drag_factor(self):
        return 1.

    def specific_air_range(self,mass,tas,dict):
        g = earth.gravity()
        return (tas*dict["lod"])/(mass*g*dict["sfc"])

    def specific_breguet_range(self,tow,range,tas,dict):
        g = earth.gravity()
        return tow*(1-np.exp(-(dict["sfc"]*g*range)/(tas*dict["lod"])))

    def specific_holding(self,mass,time,tas,dict):
        g = earth.gravity()
        return dict["sfc"]*(mass*g/dict["lod"])*time


