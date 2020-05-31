#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC

.. note:: All physical parameters are given in SI units.
"""

import numpy as np
from scipy.optimize import fsolve

from marilib.utils import earth, unit, math

from marilib.aircraft.airframe.component import Component

from marilib.aircraft.model_config import get_init


def number_of_engine(aircraft):
    ne = aircraft.arrangement.number_of_engine
    return {"twin":2, "quadri":4}.get(ne, "number of engine is unknown")

def init_thrust(aircraft):
    n_pax_ref = aircraft.requirement.n_pax_ref
    design_range = aircraft.requirement.design_range
    return (1.e5 + 177.*n_pax_ref*design_range*1.e-6)

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

        y_int = 0.5 * body_width + 0.9 * self.width      # statistical regression
        x_int = 0.80 * body_length - self.length
        z_int = body_height

        return np.array([x_int, y_int, z_int])

class FuselageTailConeMountedNacelle(Component):

    def __init__(self, aircraft):
        super(FuselageTailConeMountedNacelle, self).__init__(aircraft)

        self.tail_cone_height_ratio = get_init("FuselageTailConeMountedNacelle","tail_cone_height_ratio")
        self.specific_nacelle_cost = get_init("FuselageTailConeMountedNacelle","specific_nacelle_cost")

    def locate_nacelle(self):
        body_origin = self.aircraft.airframe.body.frame_origin
        body_height = self.aircraft.airframe.body.height
        body_length = self.aircraft.airframe.body.length

        y_axe = body_origin[1]
        x_axe = body_origin[0] + body_length
        z_axe = body_origin[2] + self.tail_cone_height_ratio * body_height

        return np.array([x_axe, y_axe, z_axe])


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

        class_name = "SemiEmpiricTfNacelle"

        ne = self.aircraft.arrangement.number_of_engine
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range

        self.n_engine = number_of_engine(aircraft)
        self.reference_thrust = init_thrust(aircraft)/self.n_engine
        self.reference_offtake = 0.
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.86, MCL=0.78, MCR=0.70, FID=0.10)
        self.tune_factor = 1.
        self.engine_bpr = get_init(class_name,"engine_bpr", val=self.__turbofan_bpr())
        self.core_thrust_ratio = get_init(class_name,"core_thrust_ratio")
        self.propeller_efficiency = get_init(class_name,"propeller_efficiency")

        self.thrust_factor = None
        self.width = None
        self.length = None

        self.propeller_mass = 0.
        self.engine_mass = 0.
        self.pylon_mass = 0.

        self.frame_origin = np.full(3,None)

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
        # Update power transfert in case of hybridization
        self.aircraft.power_system.update_power_transfert()

        # info : reference_thrust is defined by thrust(mach=0.25, altp=0, disa=15) / 0.80
        mach = 0.25
        disa = 15.
        altp = 0.

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        vair = mach * earth.sound_speed(tamb)

        # tune_factor allows that output of unitary_thrust matches the definition of the reference thrust
        self.tune_factor = 1.
        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO",pw_offtake=self.reference_offtake)
        self.tune_factor = self.reference_thrust / (dict["fn"]/0.80)

        # Following computation as aim to model the decrease in nacelle dimension due to
        # the amount of power offtaken to drive an eventual electric chain
        total_thrust0 = self.reference_thrust*0.80
        core_thrust0 = total_thrust0*self.core_thrust_ratio
        fan_thrust0 = total_thrust0*(1.-self.core_thrust_ratio)
        fan_power0 = fan_thrust0*vair/self.propeller_efficiency

        fan_power = fan_power0 - self.reference_offtake
        fan_thrust = (fan_power/vair)*self.propeller_efficiency
        total_thrust = fan_thrust + core_thrust0

        self.thrust_factor = total_thrust / total_thrust0

        self.width = 0.5*self.engine_bpr**0.7 + 5.E-6*self.reference_thrust*self.thrust_factor
        self.length = 0.86*self.width + self.engine_bpr**0.37      # statistical regression

        knac = np.pi * self.width * self.length
        self.gross_wet_area = knac*(1.48 - 0.0076*knac)*self.n_engine       # statistical regression, all engines
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        self.engine_mass = (1250. + 0.021*self.reference_thrust*self.thrust_factor)*self.n_engine       # statistical regression, all engines
        self.pylon_mass = 0.0031*self.reference_thrust*self.thrust_factor*self.n_engine
        self.mass = self.engine_mass + self.pylon_mass
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
        fuel_flow = sfc_ref * total_thrust0

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

        class_name = "SemiEmpiricTpNacelle"

        ne = self.aircraft.arrangement.number_of_engine
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range

        self.n_engine = number_of_engine(aircraft)
        self.propeller_efficiency = get_init(class_name,"propeller_efficiency")
        self.propeller_disk_load = get_init(class_name,"propeller_disk_load")
        self.sfc_type = "power"
        self.reference_power = self.__reference_power(aircraft)
        self.reference_thrust = self.reference_power*(self.propeller_efficiency/87.26)
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.95, MCL=0.90, MCR=0.70, FID=0.10)
        self.engine_bpr = 100.

        self.hub_width = None
        self.propeller_width = None
        self.width = None
        self.length = None
        self.propeller_mass = None

        self.propeller_mass = 0.
        self.engine_mass = 0.
        self.pylon_mass = 0.

        self.frame_origin = np.full(3,None)

    def __reference_power(self, aircraft):
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range
        ref_power = 0.25*(1./0.8)*(87.26/self.propeller_efficiency)*init_thrust(aircraft)/self.n_engine
        return ref_power

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

        self.width = 0.25*(self.reference_power/1.e3)**0.2        # statistical regression
        self.length = 0.84*(self.reference_power/1.e3)**0.2       # statistical regression

        self.gross_wet_area = 2.8*(self.width*self.length)*self.n_engine     # statistical regression
        self.net_wet_area = 0.80*self.gross_wet_area
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        self.engine_mass = (0.633*(self.reference_power/1.e3)**0.9)*self.n_engine       # statistical regression
        self.propeller_mass = (165./1.5e6)*self.reference_power * self.n_engine
        self.mass = self.engine_mass + self.propeller_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of a pure turboprop engine (semi-empirical model)
        """
        factor = self.rating_factor
        eta_prop = self.propeller_efficiency

        psfc_ref = unit.kgpWps_lbpshpph(0.4)   # 0.4 lb/shp/h

        rho,sig = earth.air_density(pamb,tamb)
        Vsnd = earth.sound_speed(tamb)
        Vair = Vsnd*mach

        shaft_power = throttle*getattr(factor,rating)*self.reference_power*sig**0.5 - pw_offtake

        fn = eta_prop*shaft_power/Vair
        ff = psfc_ref*(shaft_power + pw_offtake)

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


class SemiEmpiricEpNacelle(Component):

    def __init__(self, aircraft):
        super(SemiEmpiricEpNacelle, self).__init__(aircraft)

        class_name = "SemiEmpiricEpNacelle"

        ne = self.aircraft.arrangement.number_of_engine
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range

        self.n_engine = number_of_engine(aircraft)
        self.propeller_efficiency = get_init(class_name,"propeller_efficiency")
        self.propeller_disk_load = get_init(class_name,"propeller_disk_load")
        self.reference_power = self.__reference_power(aircraft)
        self.reference_thrust = self.reference_power*(self.propeller_efficiency/87.26)
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.90, MCL=0.90, MCR=0.90, FID=0.10)
        self.motor_efficiency = get_init(class_name,"motor_efficiency")
        self.controller_efficiency = get_init(class_name,"controller_efficiency")
        self.controller_pw_density = get_init(class_name,"controller_pw_density")
        self.nacelle_pw_density = get_init(class_name,"nacelle_pw_density")
        self.motor_pw_density = get_init(class_name,"motor_pw_density")
        self.engine_bpr = 100.

        self.hub_width = get_init(class_name,"hub_width")
        self.propeller_width = None
        self.width = None
        self.length = None

        self.propeller_mass = 0.
        self.engine_mass = 0.
        self.pylon_mass = 0.

        self.frame_origin = np.full(3,None)

    def __reference_power(self, aircraft):
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range
        ref_power = 0.25*(1./0.8)*(87.26/self.propeller_efficiency)*init_thrust(aircraft)/self.n_engine
        return ref_power

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

        self.propeller_width = np.sqrt((4./np.pi)*(self.reference_thrust/self.propeller_disk_load))      # Assuming 3000 N/m2

        self.width = 0.15*(self.reference_power/1.e3)**0.2        # statistical regression
        self.length = 0.55*(self.reference_power/1.e3)**0.2       # statistical regression

        self.gross_wet_area = 2.8*(self.width*self.length)*self.n_engine     # statistical regression
        self.net_wet_area = 0.80*self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        self.engine_mass = (  1./self.controller_pw_density + 1./self.motor_pw_density
                            + 1./self.nacelle_pw_density
                           ) * self.reference_power * self.n_engine
        self.propeller_mass = (165./1.5e6)*self.reference_power * self.n_engine
        self.mass = self.engine_mass + self.propeller_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression


    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of a pure turboprop engine (semi-empirical model)
        """
        Vsnd = earth.sound_speed(tamb)
        Vair = Vsnd*mach
        pw_shaft = self.reference_power*getattr(self.rating_factor,rating)*throttle - pw_offtake
        pw_elec = pw_shaft / (self.motor_efficiency*self.controller_efficiency)
        fn = self.propeller_efficiency*pw_shaft/Vair
        return {"fn":fn, "pw":pw_elec}

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

        class_name = "SemiEmpiricEfNacelle"

        ne = self.aircraft.arrangement.number_of_engine

        self.n_engine = number_of_engine(aircraft)
        self.propeller_efficiency = get_init(class_name,"propeller_efficiency")
        self.fan_efficiency = get_init(class_name,"fan_efficiency")
        self.reference_power = self.__reference_power(aircraft)
        self.reference_thrust = self.reference_power*(self.propeller_efficiency/87.26)
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.90, MCL=0.90, MCR=0.90, FID=0.10)
        self.motor_efficiency = get_init(class_name,"motor_efficiency")
        self.controller_efficiency = get_init(class_name,"controller_efficiency")
        self.controller_pw_density = get_init(class_name,"controller_pw_density")
        self.nacelle_pw_density = get_init(class_name,"nacelle_pw_density")
        self.motor_pw_density = get_init(class_name,"motor_pw_density")

        self.hub_width = get_init(class_name,"hub_width")
        self.fan_width = None
        self.nozzle_width = None
        self.nozzle_area = None
        self.width = None
        self.length = None

        self.propeller_mass = 0.
        self.engine_mass = 0.
        self.pylon_mass = 0.

        self.frame_origin = np.full(3,None)

    def __reference_power(self, aircraft):
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range
        ref_power = 0.5*(1./0.8)*(87.26/self.propeller_efficiency)*init_thrust(aircraft)/self.n_engine
        return ref_power

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
        self.engine_mass = (  1./self.controller_pw_density + 1./self.motor_pw_density
                            + 1./self.nacelle_pw_density
                          ) * self.reference_power * self.n_engine
        self.pylon_mass = 0.0031*self.reference_thrust*self.n_engine
        self.mass = self.engine_mass + self.pylon_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression

    def efan_nacelle_design(self,Pamb,Tamb,Mach,shaft_power):
        """Electrofan nacelle design
        """
        r,gam,Cp,Cv = earth.gas_data()
        Vair = Mach * earth.sound_speed(Tamb)

        # Electrical nacelle geometry : e-nacelle diameter is size by cruise conditions
        deltaV = 2.*Vair*(self.fan_efficiency/self.propeller_efficiency - 1.)      # speed variation produced by the fan

        pw_input = self.fan_efficiency*shaft_power     # kinetic energy produced by the fan

        Vinlet = Vair
        Vjet = Vinlet + deltaV
        q1 = 2.*pw_input / (Vjet**2 - Vinlet**2)

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

    def air_flow(self,rho,vair,r,d,y):
        """Air flows and averaged speed at rear end of a cylinder of radius r mouving at vair in the direction of its axes,
           y is the elevation upon the surface of the cylinder : 0 < y < inf
        """
        n = 1./7.   # exponent in the formula of the speed profile inside a turbulent BL of thickness bly : Vy/Vair = (y/d)**(1/7)
        q0 = (2.*np.pi)*(rho*vair)*(r*y + 0.5*y**2)     # Cumulated air flow at y_elev, without BL
        ym = min(y,d)
        q1 = (2.*np.pi)*(rho*vair)*d*( (r/(n+1))*(ym/d)**(n+1) + (d/(n+2))*(ym/d)**(n+2) )      # Cumulated air flow at ym, with BL
        if (y>d): q1 = q1 + q0 - (2.*np.pi)*(rho*vair)*( r*d + 0.5*d**2 )                       # Add to Q1 the air flow outside the BL
        q2 = q1 - q0        # Cumulated air flow at y_elev, inside the BL (going speed wise)
        v1 = vair*(q1/q0)   # Mean speed of q1 air flow at y_elev
        dv = vair - v1      # Mean air flow speed variation at y_elev
        return q0,q1,q2,v1,dv

    def specific_air_flow(self,r,d,y):
        """Specific air flows and speeds at rear end of a cylinder of radius r mouving at Vair in the direction of its axes,
           y is the elevation upon the surface of the cylinder : 0 < y < inf
        Qs = Q/(rho*Vair)
        Vs = V/Vair
        WARNING : even if all mass flows are positive,
        Q0 and Q1 are going backward in fuselage frame, Q2 is going forward in ground frame
        """
        n = 1/7     # exponent in the formula of the speed profile inside a turbulent BL of thickness d : Vy/Vair = (y/d)^(1/7)
        q0s = (2.*np.pi)*( r*y + 0.5*y**2 )     # Cumulated specific air flow at y, without BL, AIRPLANE FRAME
        ym = min(y,d)
        q1s = (2.*np.pi)*d*( (r/(n+1))*(ym/d)**(n+1) + (d/(n+2))*(ym/d)**(n+2) )    # Cumulated specific air flow at y inside of the BL, AIRPLANE FRAME
        if y>d: q1s = q1s + q0s - (2.*np.pi)*( r*d + 0.5*d**2 )                     # Add to Q1 the specific air flow outside of the BL, AIRPLANE FRAME
        q2s = q0s - q1s     # Cumulated specific air flow at y, inside the BL, GROUND FRAME (going speed wise)
        v1s = (q1s/q0s)     # Averaged specific speed of Q1 air flow at y
        dVs = (1. - v1s)    # Averaged specific air flow speed variation at y
        return q0s,q1s,q2s,v1s,dVs

    def boundary_layer(self,re,x_length):
        """Thickness of a turbulent boundary layer which developped turbulently from its starting point
        """
        return (0.385*x_length)/(re*x_length)**(1./5.)

    def tail_cone_boundary_layer(self,body_width,hub_width):
        """Compute the increase of BL thickness due to the fuselage tail cone tapering
        Compute the relation between d0 and d1
        d0 : boundary layer thickness around a tube of constant diameter
        d1 : boundary layer thickness around the tapered part of the tube, the nacelle hub in fact
        """
        r0 = 0.5 * body_width   # Radius of the fuselage, supposed constant
        r1 = 0.5 * hub_width    # Radius of the hub of the efan nacelle

        def fct(d1,r1,d0,r0):
            q0s0,q1s0,q2s0,v1s0,dvs0 = self.specific_air_flow(r0,d0,d0)
            q0s1,q1s1,q2s1,v1s1,dvs1 = self.specific_air_flow(r1,d1,d1)
            y = q2s0 - q2s1
            return y

        n = 25
        yVein = np.linspace(0.001,1.50,n)
        body_bnd_layer = np.zeros((n,2))

        for j in range (0, n-1):
            fct1s = (r1,yVein[j],r0)
            # computation of d1 theoretical thickness of the boundary layer that passes the same air flow around the hub
            body_bnd_layer[j,0] = yVein[j]
            body_bnd_layer[j,1] = fsolve(fct,yVein[j],fct1s)

        return body_bnd_layer

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of an electrofan engine (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(q,pw_shaft,pamb,Ttot,Vair):
            Vinlet = Vair
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q + Vinlet**2)    # Supposing isentropic compression
            TtotJet = Ttot + pw_shaft/(q*Cp)             # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
            VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                      # Mach number at nozzle output
            PtotJet = earth.total_pressure(pamb, MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            q0 = CQoA1*self.nozzle_area
            y = q0 - q
            return y

        pw_shaft = self.reference_power*getattr(self.rating_factor,rating)*throttle - pw_offtake
        pw_elec = pw_shaft / (self.controller_efficiency*self.motor_efficiency)

        Ptot = earth.total_pressure(pamb, mach)        # Total pressure at inlet position
        Ttot = earth.total_temperature(tamb, mach)     # Total temperature at inlet position

        Vair = mach * earth.sound_speed(tamb)

        fct_arg = (pw_shaft,pamb,Ttot,Vair)

        CQoA0 = self.corrected_air_flow(Ptot,Ttot,mach)       # Corrected air flow per area at fan position
        q0init = CQoA0*(0.25*np.pi*self.fan_width**2)

        # Computation of the air flow swallowed by the inlet
        output_dict = fsolve(fct, x0=q0init, args=fct_arg, full_output=True)

        q0 = output_dict[0][0]
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        Vinlet = Vair
        pw_input = self.fan_efficiency*pw_shaft
        Vjet = np.sqrt(2.*pw_input/q0 + Vinlet**2)
        eFn = q0*(Vjet - Vinlet)

        return {"fn":eFn, "pw":pw_elec}

    def unitary_sc(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary power required of an electrofan engine delivering a given thrust (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(x_in,thrust,pamb,Ttot,Vair):
            q = x_in[0]
            pw_shaft = x_in[1]
            Vinlet = Vair
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q + Vinlet**2)    # Supposing isentropic compression
            TtotJet = Ttot + pw_shaft/(q*Cp)             # Stagnation temperature increases due to introduced work
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
        PWinit = self.reference_power*getattr(self.rating_factor,rating) - pw_offtake
        x_init = [q0init,PWinit]

        # Computation of both air flow and shaft power
        output_dict = fsolve(fct, x0=x_init, args=fct_arg, full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        q0 = output_dict[0][0]
        Pw = output_dict[0][1]

        Vinlet = Vair
        pw_input = self.fan_efficiency*Pw
        Vjet = np.sqrt(2.*pw_input/q0 + Vinlet**2)
        eFn = q0*(Vjet - Vinlet)

        throttle = (Pw+pw_offtake)/(self.reference_power*getattr(self.rating_factor,rating))
        pw_elec = Pw / (self.controller_efficiency*self.motor_efficiency)
        sec = pw_elec/eFn

        return {"sec":sec, "thtl":throttle}

    def unitary_thrust_bli(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of an electrofan engine (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(y,pw_shaft,pamb,rho,Ttot,Vair,r1,d1):
            q0,q1,q2,Vinlet,dVbli = self.air_flow(rho,Vair,r1,d1,y)
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q1 + Vinlet**2)       # Supposing isentropic compression
            TtotJet = Ttot + pw_shaft/(q1*Cp)                # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp             # Static temperature
            VsndJet = earth.sound_speed(TstatJet)           # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                          # Mach number at nozzle output
            PtotJet = earth.total_pressure(pamb,MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            q = CQoA1*self.nozzle_area
            y = q1 - q
            return y

        pw_shaft = self.reference_power*getattr(self.rating_factor,rating)*throttle - pw_offtake
        pw_elec = pw_shaft / (self.controller_efficiency*self.motor_efficiency)

        Re = earth.reynolds_number(pamb,tamb,mach)
        rho,sig = earth.air_density(pamb,pamb)
        Vair = mach * earth.sound_speed(tamb)
        Ttot = earth.total_temperature(tamb, mach)     # Total temperature at inlet position

        d0 = self.boundary_layer(Re,self.body_length)      # theorical thickness of the boundary layer without taking account of fuselage tapering
        r1 = 0.5*self.hub_width      # Radius of the hub of the eFan nacelle
        d1 = math.lin_interp_1d(d0,self.bnd_layer[:,0],self.bnd_layer[:,1])     # Using the precomputed relation

        fct_arg = (pw_shaft,pamb,rho,Ttot,Vair,r1,d1)

        # Computation of y1 : thikness of the vein swallowed by the inlet
        output_dict = fsolve(fct, x0=0.5, args=fct_arg, full_output=True)

        y1 = output_dict[0][0]
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        (q0,q1,q2,Vinlet,dVbli) = self.air_flow(rho,Vair,r1,d1,y1)

        pw_input = self.fan_efficiency*pw_shaft
        Vjet = np.sqrt(2.*pw_input/q1 + Vinlet**2)
        eFn = q1*(Vjet - Vinlet)

        return {"fn":eFn, "pw":pw_elec}

    def unitary_sc_bli(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary power required of an electrofan engine delivering a given thrust (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(x_in,thrust,pamb,rho,Ttot,Vair,r1,d1):
            y = x_in[0]
            pw_shaft = x_in[1]
            q0,q1,q2,Vinlet,dVbli = self.air_flow(rho,Vair,r1,d1,y)
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q1 + Vinlet**2)    # Supposing isentropic compression
            TtotJet = Ttot + pw_shaft/(q1*Cp)             # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
            VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                      # Mach number at nozzle output
            PtotJet = earth.total_pressure(pamb, MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            q = CQoA1*self.nozzle_area
            eFn = q*(Vjet - Vinlet)
            return [q1-q, thrust-eFn]

        Re = earth.reynolds_number(pamb,tamb,mach)
        rho,sig = earth.air_density(pamb,pamb)
        Vair = mach * earth.sound_speed(tamb)
        Ptot = earth.total_pressure(pamb, mach)        # Total pressure at inlet position
        Ttot = earth.total_temperature(tamb, mach)     # Total temperature at inlet position

        d0 = self.boundary_layer(Re,self.body_length)   # theorical thickness of the boundary layer without taking account of fuselage tapering
        r1 = 0.5*self.hub_width                         # Radius of the hub of the eFan nacelle
        d1 = math.lin_interp_1d(d0,self.bnd_layer[:,0],self.bnd_layer[:,1])     # Using the precomputed relation

        fct_arg = (thrust,pamb,rho,Ttot,Vair,r1,d1)

        CQoA0 = self.corrected_air_flow(Ptot,Ttot,mach)       # Corrected air flow per area at fan position
        q0init = CQoA0*(0.25*np.pi*self.fan_width**2)
        PWinit = self.reference_power*getattr(self.rating_factor,rating) - pw_offtake
        x_init = [q0init,PWinit]

        # Computation of both air flow and shaft power
        output_dict = fsolve(fct, x0=x_init, args=fct_arg, full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        q0 = output_dict[0][0]
        Pw = output_dict[0][1]

        Vinlet = Vair
        pw_input = self.fan_efficiency*Pw
        Vjet = np.sqrt(2.*pw_input/q0 + Vinlet**2)
        eFn = q0*(Vjet - Vinlet)

        throttle = (Pw+pw_offtake)/(self.reference_power*getattr(self.rating_factor,rating))
        pw_elec = Pw / (self.controller_efficiency*self.motor_efficiency)
        sec = pw_elec/eFn

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

class FuselageTailConeMountedEfNacelle(SemiEmpiricEfNacelle,FuselageTailConeMountedNacelle):
    def __init__(self, aircraft):
        super(FuselageTailConeMountedEfNacelle, self).__init__(aircraft)
        self.n_engine = 1
        self.reference_power = self.aircraft.airframe.system.chain_power
        self.hub_width = get_init("FuselageTailConeMountedNacelle","hub_width")
        self.body_width = self.aircraft.airframe.body.width
        self.body_length = self.aircraft.airframe.body.length
        self.bnd_layer = self.tail_cone_boundary_layer(self.body_width,self.hub_width)





