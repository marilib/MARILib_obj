#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin

.. note:: All physical parameters are given in SI units.
"""

import numpy as np
from scipy.optimize import fsolve

from marilib.utils import earth, unit

from marilib.aircraft.airframe.component import Component


class InboradWingMountedNacelle(Component):

    def __init__(self, aircraft):
        super(InboradWingMountedNacelle, self).__init__(aircraft)

    def locate_nacelle(self):
        body_width = self.aircraft.airframe.body.width
        wing_root_loc = self.aircraft.airframe.wing.root_loc
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_dihedral = self.aircraft.airframe.wing.dihedral
        wing_kink_c = self.aircraft.airframe.wing.kink_c
        wing_kink_loc = self.aircraft.airframe.wing.kink_loc
        wing_tip_c = self.aircraft.airframe.wing.tip_c
        wing_tip_loc = self.aircraft.airframe.wing.tip_loc

        tan_phi0 = 0.25*(wing_kink_c-wing_tip_c)/(wing_tip_loc[1]-wing_kink_loc[1]) + np.tan(wing_sweep25)

        y_int = 0.6 * body_width + self.lateral_margin()
        x_int = wing_root_loc[0] + (y_int-wing_root_loc[1])*tan_phi0 - 0.7*self.length
        z_int = wing_root_loc[2] + (y_int-wing_root_loc[2])*np.tan(wing_dihedral) - self.vertical_margin()

        return np.array([x_int, y_int, z_int])


class OutboradWingMountedNacelle(Component):

    def __init__(self, aircraft):
        super(OutboradWingMountedNacelle, self).__init__(aircraft)

    def locate_nacelle(self):
        body_width = self.aircraft.airframe.body.width
        wing_root_loc = self.aircraft.airframe.wing.root_loc
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_dihedral = self.aircraft.airframe.wing.dihedral
        wing_kink_c = self.aircraft.airframe.wing.kink_c
        wing_kink_loc = self.aircraft.airframe.wing.kink_loc
        wing_tip_c = self.aircraft.airframe.wing.tip_c
        wing_tip_loc = self.aircraft.airframe.wing.tip_loc
        nac_y_int = self.aircraft.airframe.internal_nacelle.frame_origin[1]

        tan_phi0 = 0.25*(wing_kink_c-wing_tip_c)/(wing_tip_loc[1]-wing_kink_loc[1]) + np.tan(wing_sweep25)

        y_ext = 1.6 * body_width + self.lateral_margin()
        x_ext = wing_root_loc[0] + (y_ext-wing_root_loc[1])*tan_phi0 - 0.7*self.length
        z_ext = wing_root_loc[2] + (y_ext-wing_root_loc[2])*np.tan(wing_dihedral) - self.vertical_margin()

        return np.array([x_ext, y_ext, z_ext])


class RearFuselageMountedNacelle(Component):

    def __init__(self, aircraft):
        super(RearFuselageMountedNacelle, self).__init__(aircraft)

    def locate_nacelle(self):
        body_width = self.aircraft.airframe.body.width
        body_height = self.aircraft.airframe.body.height
        body_length = self.aircraft.airframe.body.length

        y_int = 0.5 * body_width + 1.2 * self.width      # statistical regression
        x_int = 0.80 * body_length - self.length
        z_int = body_height

        return np.array([x_int, y_int, z_int])


class System(Component):

    def __init__(self, aircraft):
        super(System, self).__init__(aircraft)

    def eval_geometry(self):
        self.frame_origin = [0., 0., 0.]

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        body_cg = self.aircraft.airframe.body.cg
        wing_cg = self.aircraft.airframe.wing.cg
        horizontal_stab_cg = self.aircraft.airframe.horizontal_stab.cg
        vertical_stab_cg = self.aircraft.airframe.vertical_stab.cg
        nacelle_cg = self.aircraft.airframe.nacelle.cg
        landing_gear_cg = self.aircraft.airframe.landing_gear.cg

        self.mass = 0.545*mtow**0.8    # global mass of all systems

        self.cg =   0.50*body_cg \
                  + 0.20*wing_cg \
                  + 0.10*landing_gear_cg \
                  + 0.05*horizontal_stab_cg \
                  + 0.05*vertical_stab_cg \
                  + 0.10*nacelle_cg


class RatingFactor(object):
    def __init__(self, MTO=None, MCN=None, MCL=None, MCR=None, FID=None):
        self.MTO = MTO
        self.MCN = MCN
        self.MCL = MCL
        self.MCR = MCR
        self.FID = FID


class SemiEmpiricTfNacelle(Component):

    def __init__(self, aircraft):
        super(SemiEmpiricTfNacelle, self).__init__(aircraft)

        ne = self.aircraft.arrangement.number_of_engine
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range

        self.n_engine = {"twin":2, "quadri":4}.get(ne, "number of engine is unknown")
        self.reference_thrust = (1.e5 + 177.*n_pax_ref*design_range*1.e-6)/self.n_engine
        self.reference_offtake = 0.
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.86, MCL=0.78, MCR=0.70, FID=0.10)
        self.fuel_heat = self.__fuel_heat()
        self.sfc_type = "thrust"
        self.tune_factor = 1.
        self.engine_bpr = self.__turbofan_bpr()
        self.core_thrust_ratio = 0.13
        self.propeller_efficiency = 0.82

        self.width = None
        self.length = None

        self.frame_origin = np.full(3,None)

    def __fuel_heat(self):
        energy_source = self.aircraft.arrangement.energy_source
        return earth.fuel_heat(energy_source)

    def __turbofan_bpr(self):
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        if (80<n_pax_ref):
            bpr = 9.
        else:
            bpr = 5.
        return bpr

    def lateral_margin(self):
        return 1.5*self.width

    def vertical_margin(self):
        return 0.55*self.width

    def eval_geometry(self):
        self.fuel_heat = self.__fuel_heat()

        # info : reference_thrust is defined by thrust(mach=0.25, altp=0, disa=15) / 0.80
        mach = 0.25
        disa = 15.
        altp = 0.

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        vair = mach * earth.sound_speed(tamb)

        # tune_factor allows that output of unitary_thrust matches the definition of the reference thrust
        self.tune_factor = 1.
        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO")
        self.tune_factor = self.reference_thrust / (dict["fn"]/0.80)

        # Following computation as aim to model the decrease in nacelle dimension due to
        # the amount of power offtaken to drive an eventual electric chain
        total_thrust0 = self.reference_thrust*0.80
        core_thrust0 = total_thrust0*self.core_thrust_ratio
        fan_thrust0 = total_thrust0*(1.-self.core_thrust_ratio)
        fan_power0 = fan_thrust0*vair/self.propeller_efficiency

        # total offtake is split over all engines
        fan_power = fan_power0 - self.reference_offtake*self.n_engine
        fan_thrust = (fan_power/vair)*self.propeller_efficiency
        total_thrust = fan_thrust + core_thrust0

        thrust_factor = total_thrust / total_thrust0

        self.width = 0.5*self.engine_bpr**0.7 + 5.E-6*self.reference_thrust*thrust_factor
        self.length = 0.86*self.width + self.engine_bpr**0.37      # statistical regression

        knac = np.pi * self.width * self.length
        self.gross_wet_area = knac*(1.48 - 0.0076*knac)*self.n_engine       # statistical regression, all engines
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        engine_mass = (1250. + 0.021*self.reference_thrust)*self.n_engine       # statistical regression, all engines
        pylon_mass = 0.0031*self.reference_thrust*self.n_engine
        self.mass = engine_mass + pylon_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of a pure turbofan engine (semi-empirical model)
        """
        kth =  0.475*mach**2 + 0.091*(self.engine_bpr/10.)**2 \
             - 0.283*mach*self.engine_bpr/10. \
             - 0.633*mach - 0.081*self.engine_bpr/10. + 1.192

        rho,sig = earth.air_density(pamb, tamb)
        vair = mach * earth.sound_speed(tamb)

        total_thrust0 =   self.reference_thrust \
                        * self.tune_factor \
                        * kth \
                        * getattr(self.rating_factor,rating) \
                        * throttle \
                        * sig**0.75
        core_thrust0 = total_thrust0 * self.core_thrust_ratio        # Core thrust
        fan_thrust0 = total_thrust0 * (1.-self.core_thrust_ratio)    # Fan thrust
        fan_power0 = fan_thrust0*vair/self.propeller_efficiency   # Available total shaft power for one engine

        fan_power = fan_power0 - pw_offtake
        fan_thrust = (fan_power/vair)*self.propeller_efficiency
        total_thrust = fan_thrust + core_thrust0

        sfc_ref = ( 0.4 + 1./self.engine_bpr**0.895 )/36000.
        sfc = sfc_ref * earth.fuel_heat("kerosene") / self.fuel_heat
        fuel_flow = sfc * total_thrust0

        return {"fn":total_thrust, "ff":fuel_flow, "t4":None}

    def unitary_sc(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary thrust of a pure turbofan engine (semi-empirical model)
        """
        dict = self.unitary_thrust(pamb,tamb,mach,rating,pw_offtake=pw_offtake)
        throttle = thrust/dict["fn"]
        sfc = dict["ff"]/dict["fn"]
        t41 = dict["t4"]
        return {"sfc":sfc, "thtl":throttle, "t4":t41}


class OutboardWingMountedTfNacelle(SemiEmpiricTfNacelle,OutboradWingMountedNacelle):
    def __init__(self, aircraft):
        super(OutboardWingMountedTfNacelle, self).__init__(aircraft)

class InboardWingMountedTfNacelle(SemiEmpiricTfNacelle,InboradWingMountedNacelle):
    def __init__(self, aircraft):
        super(InboardWingMountedTfNacelle, self).__init__(aircraft)

class RearFuselageMountedTfNacelle(SemiEmpiricTfNacelle,RearFuselageMountedNacelle):
    def __init__(self, aircraft):
        super(RearFuselageMountedTfNacelle, self).__init__(aircraft)


class SemiEmpiricTpNacelle(Component):

    def __init__(self, aircraft):
        super(SemiEmpiricTpNacelle, self).__init__(aircraft)

        ne = self.aircraft.arrangement.number_of_engine
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range

        self.n_engine = {"twin":2, "quadri":4}.get(ne, "number of engine is unknown")
        self.propeller_efficiency = 0.82
        self.propeller_disk_load = 3000.    # N/m2
        self.sfc_type = "power"
        self.reference_power = 0.25*(1./0.8)*(87.26/self.propeller_efficiency)*(1.e5 + 177.*n_pax_ref*design_range*1.e-6)/self.n_engine
        self.reference_thrust = self.reference_power*(self.propeller_efficiency/87.26)
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.95, MCL=0.90, MCR=0.70, FID=0.10)
        self.fuel_heat = self.__fuel_heat()
        self.engine_bpr = 100.

        self.hub_width = None
        self.propeller_width = None
        self.width = None
        self.length = None

        self.frame_origin = np.full(3,None)

    def __fuel_heat(self):
        energy_source = self.aircraft.arrangement.energy_source
        return earth.fuel_heat(energy_source)

    def lateral_margin(self):
        return 0.8*self.propeller_width

    def vertical_margin(self):
        return 0.

    def eval_geometry(self):
        self.fuel_heat = self.__fuel_heat()

        # info : reference_thrust is defined by thrust(mach=0.25, altp=0, disa=15) / 0.80
        mach = 0.25
        disa = 15.
        altp = 0.

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO")
        self.reference_thrust = dict["fn"] / 0.80

        self.hub_width = 0.2
        self.propeller_width = np.sqrt((4./np.pi)*(self.reference_thrust/self.propeller_disk_load))      # Assuming 3000 N/m2

        self.width = 0.25*(self.reference_power/1.e3)**0.2        # statistical regression
        self.length = 0.84*(self.reference_power/1.e3)**0.2       # statistical regression

        self.gross_wet_area = 2.8*(self.width*self.length)*self.n_engine     # statistical regression
        self.net_wet_area = 0.80*self.gross_wet_area
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        engine_mass = (0.633*(self.reference_power/1.e3)**0.9)*self.n_engine       # statistical regression
        propeller_mass = (165./1.5e6)*self.reference_power * self.n_engine
        self.mass = engine_mass + propeller_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of a pure turboprop engine (semi-empirical model)
        """
        factor = self.rating_factor
        eta_prop = self.propeller_efficiency

        psfc_ref = unit.kgpWps_lbpshpph(0.4)   # 0.4 lb/shp/h
        psfc = psfc_ref * earth.fuel_heat("kerosene") / self.fuel_heat

        rho,sig = earth.air_density(pamb,tamb)
        Vsnd = earth.sound_speed(tamb)
        Vair = Vsnd*mach

        shaft_power = throttle*getattr(factor,rating)*self.reference_power*sig**0.5 - pw_offtake

        fn = eta_prop*shaft_power/Vair
        ff = psfc*shaft_power

        return {"fn":fn, "ff":ff, "pw":shaft_power, "t4":None}

    def unitary_sc(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary thrust of a pure turbofan engine (semi-empirical model)
        """
        dict = self.unitary_thrust(pamb,tamb,mach,rating,pw_offtake=pw_offtake)
        throttle = thrust/dict["fn"]
        sfc = dict["ff"]/dict["pw"]     # Power SFC
        t41 = dict["t4"]
        return {"sfc":sfc, "thtl":throttle, "t4":t41}


class OutboardWingMountedTpNacelle(SemiEmpiricTpNacelle,OutboradWingMountedNacelle):
    def __init__(self, aircraft):
        super(OutboardWingMountedTpNacelle, self).__init__(aircraft)

class InboardWingMountedTpNacelle(SemiEmpiricTpNacelle,InboradWingMountedNacelle):
    def __init__(self, aircraft):
        super(InboardWingMountedTpNacelle, self).__init__(aircraft)



class SystemWithBattery(Component):

    def __init__(self, aircraft):
        super(SystemWithBattery, self).__init__(aircraft)

        self.wiring_efficiency = 0.995
        self.wiring_pw_density = 20.e3      # W/kg, Wiring

        self.cooling_pw_density = 15.e3     # W/kg, Cooling

        self.battery_density = 2800.                    # kg/m3
        self.battery_energy_density = unit.J_kWh(0.4)   # J/kg

        self.power_chain_efficiency = None

    def eval_geometry(self):
        self.frame_origin = [0., 0., 0.]

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        body_cg = self.aircraft.airframe.body.cg
        wing_cg = self.aircraft.airframe.wing.cg
        horizontal_stab_cg = self.aircraft.airframe.horizontal_stab.cg
        vertical_stab_cg = self.aircraft.airframe.vertical_stab.cg
        nacelle_cg = self.aircraft.airframe.nacelle.cg
        landing_gear_cg = self.aircraft.airframe.landing_gear.cg

        self.power_chain_efficiency =   self.wiring_efficiency \
                                      * self.aircraft.airframe.nacelle.controller_efficiency \
                                      * self.aircraft.airframe.nacelle.motor_efficiency

        shaft_power_max = self.aircraft.airframe.nacelle.reference_power
        n_engine = self.aircraft.airframe.nacelle.n_engine

        power_elec_mass = (  1./self.wiring_pw_density + 1./self.cooling_pw_density
                          ) * (shaft_power_max * n_engine)

        power_elec_cg = 0.70*nacelle_cg + 0.30*body_cg

        self.mass = 0.545*mtow**0.8  + power_elec_mass  # global mass of all systems

        self.cg =   0.40*body_cg \
                  + 0.20*wing_cg \
                  + 0.10*landing_gear_cg \
                  + 0.05*horizontal_stab_cg \
                  + 0.05*vertical_stab_cg \
                  + 0.10*nacelle_cg \
                  + 0.10*power_elec_cg


class SemiEmpiricEpNacelle(Component):

    def __init__(self, aircraft):
        super(SemiEmpiricEpNacelle, self).__init__(aircraft)

        ne = self.aircraft.arrangement.number_of_engine
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range

        self.n_engine = {"twin":2, "quadri":4}.get(ne, "number of engine is unknown")
        self.propeller_efficiency = 0.82
        self.propeller_disk_load = 3000.    # N/m2
        self.reference_power = 0.25*(1./0.8)*(87.26/self.propeller_efficiency)*(1.e5 + 177.*n_pax_ref*design_range*1.e-6)/self.n_engine
        self.reference_thrust = self.reference_power*(self.propeller_efficiency/87.26)
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.90, MCL=0.90, MCR=0.90, FID=0.10)
        self.motor_efficiency = 0.95
        self.controller_efficiency = 0.99
        self.controller_pw_density = 20.e3    # W/kg
        self.nacelle_pw_density = 5.e3    # W/kg
        self.motor_pw_density = 10.e3    # W/kg
        self.engine_bpr = 100.

        self.hub_width = None
        self.propeller_width = None
        self.width = None
        self.length = None

        self.frame_origin = np.full(3,None)

    def __fuel_heat(self):
        energy_source = self.aircraft.arrangement.energy_source
        return earth.fuel_heat(energy_source)

    def lateral_margin(self):
        return 0.8*self.propeller_width

    def vertical_margin(self):
        return 0.

    def eval_geometry(self):
        # info : reference_thrust is defined by thrust(mach=0.25, altp=0, disa=15) / 0.80
        mach = 0.25
        disa = 15.
        altp = 0.

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO")
        self.reference_thrust = dict["fn"] / 0.80

        self.hub_width = 0.2
        self.propeller_width = np.sqrt((4./np.pi)*(self.reference_thrust/self.propeller_disk_load))      # Assuming 3000 N/m2

        self.width = 0.15*(self.reference_power/1.e3)**0.2        # statistical regression
        self.length = 0.55*(self.reference_power/1.e3)**0.2       # statistical regression

        self.gross_wet_area = 2.8*(self.width*self.length)*self.n_engine     # statistical regression
        self.net_wet_area = 0.80*self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        engine_mass = (  1./self.controller_pw_density + 1./self.motor_pw_density
                       + 1./self.nacelle_pw_density
                      ) * self.reference_power * self.n_engine
        propeller_mass = (165./1.5e6)*self.reference_power * self.n_engine
        self.mass = engine_mass + propeller_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression


    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of a pure turboprop engine (semi-empirical model)
        """
        Vsnd = earth.sound_speed(tamb)
        Vair = Vsnd*mach
        pw_input = self.reference_power*getattr(self.rating_factor,rating)*throttle
        pw_shaft = pw_input*self.motor_efficiency*self.controller_efficiency - pw_offtake
        fn = self.propeller_efficiency*pw_shaft/Vair
        return {"fn":fn, "pw":pw_input}

    def unitary_sc(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary thrust of a pure turbofan engine (semi-empirical model)
        """
        dict = self.unitary_thrust(pamb,tamb,mach,rating,pw_offtake=pw_offtake)
        fn = dict["fn"]
        pw = dict["pw"]
        throttle = thrust/fn
        sec = pw/fn     # Specific Energy Consumption
        return {"sec":sec, "thtl":throttle}


class OutboardWingMountedEpNacelle(SemiEmpiricEpNacelle,OutboradWingMountedNacelle):
    def __init__(self, aircraft):
        super(OutboardWingMountedEpNacelle, self).__init__(aircraft)

class InboardWingMountedEpNacelle(SemiEmpiricEpNacelle,InboradWingMountedNacelle):
    def __init__(self, aircraft):
        super(InboardWingMountedEpNacelle, self).__init__(aircraft)


class SemiEmpiricEfNacelle(Component):

    def __init__(self, aircraft):
        super(SemiEmpiricEfNacelle, self).__init__(aircraft)

        ne = self.aircraft.arrangement.number_of_engine
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range

        self.n_engine = {"twin":2, "quadri":4}.get(ne, "number of engine is unknown")
        self.propeller_efficiency = 0.82
        self.fan_efficiency = 0.95
        self.reference_power = 0.5*(1./0.8)*(87.26/self.propeller_efficiency)*(1.e5 + 177.*n_pax_ref*design_range*1.e-6)/self.n_engine
        self.reference_thrust = self.reference_power*(self.propeller_efficiency/87.26)
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.90, MCL=0.90, MCR=0.90, FID=0.10)
        self.motor_efficiency = 0.95
        self.controller_efficiency = 0.99
        self.controller_pw_density = 20.e3    # W/kg
        self.nacelle_pw_density = 5.e3    # W/kg
        self.motor_pw_density = 10.e3    # W/kg

        self.hub_width = 0.20
        self.fan_width = None
        self.nozzle_width = None
        self.nozzle_area = None
        self.width = None
        self.length = None

        self.frame_origin = np.full(3,None)

    def lateral_margin(self):
        return 1.5*self.width

    def vertical_margin(self):
        return 0.55*self.width

    def eval_geometry(self):
        # Electric fan geometry is design for MCR in cruise condition
        disa = self.aircraft.requirement.cruise_disa
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        shaft_power = self.reference_power*self.rating_factor.MCR

        self.efan_nacelle_design(pamb,tamb,mach,shaft_power)

        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

        # info : reference_thrust is defined by thrust(mach=0.25, altp=0, disa=15) / 0.80
        mach = 0.25
        disa = 15.
        altp = 0.

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO")
        self.reference_thrust = dict["fn"] / 0.80

    def eval_mass(self):
        engine_mass = (  1./self.controller_pw_density + 1./self.motor_pw_density
                       + 1./self.nacelle_pw_density
                      ) * self.reference_power * self.n_engine
        pylon_mass = 0.0031*self.reference_thrust*self.n_engine
        self.mass = engine_mass + pylon_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression

    def efan_nacelle_design(self,Pamb,Tamb,Mach,shaft_power):
        """Electrofan nacelle design
        """
        r,gam,Cp,Cv = earth.gas_data()
        Vair = Mach * earth.sound_speed(Tamb)

        # Electrical nacelle geometry : e-nacelle diameter is size by cruise conditions
        deltaV = 2.*Vair*(self.fan_efficiency/self.propeller_efficiency - 1.)      # speed variation produced by the fan

        PwInput = self.fan_efficiency*shaft_power     # kinetic energy produced by the fan

        Vinlet = Vair
        Vjet = Vinlet + deltaV
        q1 = 2.*PwInput / (Vjet**2 - Vinlet**2)

        MachInlet = Mach     # The inlet is in free stream
        Ptot = earth.total_pressure(Pamb, MachInlet)        # Stagnation pressure at inlet position
        Ttot = earth.total_temperature(Tamb, MachInlet)     # Stagnation temperature at inlet position

        MachFan = 0.5       # required Mach number at fan position
        CQoA1 = self.corrected_air_flow(Ptot,Ttot,MachFan)        # Corrected air flow per area at fan position

        eFanArea = q1/CQoA1     # Fan area around the hub
        fan_width = np.sqrt(self.hub_width**2 + 4*eFanArea/np.pi)        # Fan diameter

        TtotJet = Ttot + shaft_power/(q1*Cp)        # Stagnation pressure increases due to introduced work
        Tstat = TtotJet - 0.5*Vjet**2/Cp        # static temperature

        VsndJet = np.sqrt(gam*r*Tstat) # Sound velocity at nozzle exhaust
        MachJet = Vjet/VsndJet # Mach number at nozzle output
        PtotJet = earth.total_pressure(Pamb, MachJet)       # total pressure at nozzle exhaust (P = Pamb)

        CQoA2 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)     # Corrected air flow per area at nozzle output
        nozzle_area = q1/CQoA2        # Fan area around the hub
        nozzle_width = np.sqrt(4*nozzle_area/np.pi)       # Nozzle diameter

        self.fan_width = fan_width
        self.nozzle_width = nozzle_width
        self.nozzle_area = nozzle_area

        self.width = 1.20*fan_width      # Surrounding structure
        self.length = 1.50*self.width

        self.gross_wet_area = np.pi*self.width*self.length*self.n_engine
        self.net_wet_area = self.gross_wet_area

    def corrected_air_flow(self,Ptot,Ttot,Mach):
        """Computes the corrected air flow per square meter
        """
        r,gam,Cp,Cv = earth.gas_data()
        f_m = Mach*(1. + 0.5*(gam-1)*Mach**2)**(-(gam+1.)/(2.*(gam-1.)))
        cqoa = (np.sqrt(gam/r)*Ptot/np.sqrt(Ttot))*f_m
        return cqoa

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of an electrofan engine (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(q,PwShaft,pamb,Ttot,Vair):
            Vinlet = Vair
            PwInput = self.fan_efficiency*PwShaft
            Vjet = np.sqrt(2.*PwInput/q + Vinlet**2)    # Supposing isentropic compression
            TtotJet = Ttot + PwShaft/(q*Cp)             # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
            VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                      # Mach number at nozzle output
            PtotJet = earth.total_pressure(pamb, MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            q0 = CQoA1*self.nozzle_area
            y = q0 - q
            return y

        PwInput = self.reference_power*getattr(self.rating_factor,rating)*throttle
        PwShaft = PwInput*self.motor_efficiency*self.controller_efficiency - pw_offtake

        Ptot = earth.total_pressure(pamb, mach)        # Total pressure at inlet position
        Ttot = earth.total_temperature(tamb, mach)     # Total temperature at inlet position

        Vair = mach * earth.sound_speed(tamb)

        fct_arg = (PwShaft,pamb,Ttot,Vair)

        CQoA0 = self.corrected_air_flow(Ptot,Ttot,mach)       # Corrected air flow per area at fan position
        q0init = CQoA0*(0.25*np.pi*self.fan_width**2)

        # Computation of the air flow swallowed by the inlet
        output_dict = fsolve(fct, x0=q0init, args=fct_arg, full_output=True)

        q0 = output_dict[0][0]
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        Vinlet = Vair
        PwInput = self.fan_efficiency*PwShaft
        Vjet = np.sqrt(2.*PwInput/q0 + Vinlet**2)
        eFn = q0*(Vjet - Vinlet)

        return {"fn":eFn, "pw":PwInput}

    def unitary_sc_2(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary thrust of a pure turbofan engine (semi-empirical model)
        """
        def fct_sc(thtl):
            dict = self.unitary_thrust(pamb,tamb,mach,rating,throttle=thtl,pw_offtake=pw_offtake)
            y = thrust - dict["fn"]
            return y

        # Computation of both air flow and shaft power
        output_dict = fsolve(fct_sc, x0=0.95, args=(), full_output=True)

        thtl = output_dict[0][0]
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        dict = self.unitary_thrust(pamb,tamb,mach,rating,throttle=thtl,pw_offtake=pw_offtake)
        pw = dict["pw"]
        sec = pw/dict["fn"]
        return {"sec":sec, "thtl":thtl}

    def unitary_sc(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary power required of an electrofan engine delivering a given thrust (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(x_in,thrust,pamb,Ttot,Vair):
            q = x_in[0]
            PwShaft = x_in[1]
            Vinlet = Vair
            PwInput = self.fan_efficiency*PwShaft
            Vjet = np.sqrt(2.*PwInput/q + Vinlet**2)    # Supposing isentropic compression
            TtotJet = Ttot + PwShaft/(q*Cp)             # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
            VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                      # Mach number at nozzle output
            PtotJet = earth.total_pressure(pamb, MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            q0 = CQoA1*self.nozzle_area
            eFn = q*(Vjet - Vinlet)
            return [q0-q, thrust-eFn]

        Ptot = earth.total_pressure(pamb, mach)        # Total pressure at inlet position
        Ttot = earth.total_temperature(tamb, mach)     # Total temperature at inlet position

        Vair = mach * earth.sound_speed(tamb)

        fct_arg = (thrust,pamb,Ttot,Vair)

        CQoA0 = self.corrected_air_flow(Ptot,Ttot,mach)       # Corrected air flow per area at fan position
        q0init = CQoA0*(0.25*np.pi*self.fan_width**2)

        PwInput = self.reference_power*getattr(self.rating_factor,rating) - pw_offtake
        PWinit = PwInput*self.motor_efficiency*self.controller_efficiency

        x_init = [q0init,PWinit]

        # Computation of both air flow and shaft power
        output_dict = fsolve(fct, x0=x_init, args=fct_arg, full_output=True)

        q0 = output_dict[0][0]
        Pw = output_dict[0][1]
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        Vinlet = Vair
        PwInput = self.fan_efficiency*Pw
        Vjet = np.sqrt(2.*PwInput/q0 + Vinlet**2)
        eFn = q0*(Vjet - Vinlet)

        throttle = (Pw+pw_offtake)/(self.reference_power*getattr(self.rating_factor,rating))
        sec = Pw/eFn

        return {"sec":sec, "thtl":throttle}


class OutboardWingMountedEfNacelle(SemiEmpiricEfNacelle,OutboradWingMountedNacelle):
    def __init__(self, aircraft):
        super(OutboardWingMountedEfNacelle, self).__init__(aircraft)

class InboardWingMountedEfNacelle(SemiEmpiricEfNacelle,InboradWingMountedNacelle):
    def __init__(self, aircraft):
        super(InboardWingMountedEfNacelle, self).__init__(aircraft)

class RearFuselageMountedEfNacelle(SemiEmpiricEfNacelle,RearFuselageMountedNacelle):
    def __init__(self, aircraft):
        super(RearFuselageMountedEfNacelle, self).__init__(aircraft)


class SystemPartialTurboelectric(Component):

    def __init__(self, aircraft):
        super(SystemPartialTurboelectric, self).__init__(aircraft)

        self.generator_efficiency = 0.95
        self.generator_pw_density = 10.e3   # W/kg, Electric generator

        self.rectifier_efficiency = 0.98
        self.rectifier_pw_density = 20.e3   # W/kg, Rectifier

        self.wiring_efficiency = 0.995
        self.wiring_pw_density = 20.e3      # W/kg, Wiring

        self.cooling_pw_density = 15.e3     # W/kg, Cooling

        self.battery_density = 2800.                    # kg/m3
        self.battery_energy_density = unit.J_kWh(1.2)   # J/kg

        self.power_chain_efficiency = None

