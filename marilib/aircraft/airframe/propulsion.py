#!/usr/bin/env python3
"""
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

        y_int = 0.6 * body_width + self.y_wise_margin(1)
        x_int = wing_root_loc[0] + (y_int-wing_root_loc[1])*tan_phi0 - 0.7*self.length
        z_int = wing_root_loc[2] + (y_int-wing_root_loc[2])*np.tan(wing_dihedral) - self.z_wise_margin()

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

        tan_phi0 = 0.25*(wing_kink_c-wing_tip_c)/(wing_tip_loc[1]-wing_kink_loc[1]) + np.tan(wing_sweep25)

        y_ext = 0.6 * body_width + self.y_wise_margin(2)
        x_ext = wing_root_loc[0] + (y_ext-wing_root_loc[1])*tan_phi0 - 0.7*self.length
        z_ext = wing_root_loc[2] + (y_ext-wing_root_loc[2])*np.tan(wing_dihedral) - self.z_wise_margin()

        return np.array([x_ext, y_ext, z_ext])

class ExternalWingMountedNacelle(Component):

    def __init__(self, aircraft):
        super(ExternalWingMountedNacelle, self).__init__(aircraft)

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

        y_ext = 0.6 * body_width + self.y_wise_margin(3)
        x_ext = wing_root_loc[0] + (y_ext-wing_root_loc[1])*tan_phi0 - 0.7*self.length
        z_ext = wing_root_loc[2] + (y_ext-wing_root_loc[2])*np.tan(wing_dihedral) - self.z_wise_margin()

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

class PodTailConeMountedNacelle(Component):

    def __init__(self, aircraft):
        super(PodTailConeMountedNacelle, self).__init__(aircraft)

        self.lateral_margin = get_init("PodTailConeMountedNacelle","lateral_margin")
        self.x_loc_ratio = get_init("PodTailConeMountedNacelle","x_loc_ratio")
        self.specific_nacelle_cost = get_init("PodTailConeMountedNacelle","specific_nacelle_cost")

    def locate_nacelle(self):
        body_width = self.aircraft.airframe.body.width
        wing_root_loc = self.aircraft.airframe.wing.root_loc
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_dihedral = self.aircraft.airframe.wing.dihedral
        wing_kink_c = self.aircraft.airframe.wing.kink_c
        wing_kink_loc = self.aircraft.airframe.wing.kink_loc
        wing_tip_c = self.aircraft.airframe.wing.tip_c
        wing_tip_loc = self.aircraft.airframe.wing.tip_loc
        tank_length = self.aircraft.airframe.tank.length
        tank_width = self.aircraft.airframe.tank.width

        tan_phi0 = 0.25*(wing_kink_c-wing_tip_c)/(wing_tip_loc[1]-wing_kink_loc[1]) + np.tan(wing_sweep25)

        # Recompute pod position
        y_axe = 0.6 * body_width + (0.5 + self.lateral_margin)*tank_width
        x_axe = wing_root_loc[0] + (y_axe-wing_root_loc[1])*tan_phi0 - self.x_loc_ratio*tank_length
        z_axe = wing_root_loc[2] + (y_axe-wing_root_loc[2])*np.tan(wing_dihedral)

        self.aircraft.airframe.tank.frame_origin = [x_axe, y_axe, z_axe]

        # Locate nacelle
        y_int = y_axe
        x_int = x_axe + self.aircraft.airframe.tank.length
        z_int = z_axe

        return np.array([x_int, y_int, z_int])


class RatingFactor(object):
    def __init__(self, MTO=None, MCN=None, MCL=None, MCR=None, FID=None):
        self.MTO = MTO
        self.MCN = MCN
        self.MCL = MCL
        self.MCR = MCR
        self.FID = FID


class SemiEmpiricTf0Nacelle(Component):

    def __init__(self, aircraft):
        super(SemiEmpiricTf0Nacelle, self).__init__(aircraft)

        class_name = "SemiEmpiricTf0Nacelle"

        ne = self.aircraft.arrangement.number_of_engine
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range

        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.86, MCL=0.78, MCR=0.70, FID=0.10)
        self.reference_offtake = 0.
        self.tune_factor = 1.
        self.engine_bpr = get_init(class_name,"engine_bpr", val=self.__turbofan_bpr())
        self.core_thrust_ratio = get_init(class_name,"core_thrust_ratio")
        self.propeller_efficiency = get_init(class_name,"propeller_efficiency")
        self.lateral_margin = get_init(class_name,"lateral_margin")
        self.vertical_margin = get_init(class_name,"vertical_margin")

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

    def y_wise_margin(self, n):
        if n==1: return 1.5 * self.lateral_margin * self.width
        elif n==2:  return 3.0 * self.lateral_margin * self.width
        elif n==3:  return 4.5 * self.lateral_margin * self.width

    def z_wise_margin(self):
        return (self.vertical_margin - 0.45)*self.width

    def eval_geometry(self):
        # Update power transfert in case of hybridization
        self.aircraft.power_system.update_power_transfert()

        reference_thrust = self.aircraft.power_system.reference_thrust
        n_engine = self.aircraft.power_system.n_engine

        # info : reference_thrust is defined by thrust(mach=0.25, altp=0, disa=15) / 0.80
        mach = 0.25
        disa = 15.
        altp = 0.

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        vair = mach * earth.sound_speed(tamb)

        # tune_factor allows that output of unitary_thrust matches the definition of the reference thrust
        self.tune_factor = 1.
        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO",pw_offtake=self.reference_offtake)
        self.tune_factor = reference_thrust / (dict["fn"]/0.80)

        # Following computation as aim to model the decrease in nacelle dimension due to
        # the amount of power offtaken to drive an eventual electric chain
        total_thrust0 = reference_thrust*0.80
        core_thrust0 = total_thrust0*self.core_thrust_ratio
        fan_thrust0 = total_thrust0*(1.-self.core_thrust_ratio)
        fan_power0 = fan_thrust0*vair/self.propeller_efficiency

        fan_power = fan_power0 - self.reference_offtake
        fan_thrust = (fan_power/vair)*self.propeller_efficiency
        total_thrust = fan_thrust + core_thrust0

        self.thrust_factor = total_thrust / total_thrust0

        self.width = 0.5*self.engine_bpr**0.7 + 5.E-6*reference_thrust*self.thrust_factor
        self.length = 0.86*self.width + self.engine_bpr**0.37      # statistical regression

        knac = np.pi * self.width * self.length
        self.gross_wet_area = knac*(1.48 - 0.0076*knac)*n_engine       # statistical regression, all engines
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        reference_thrust = self.aircraft.power_system.reference_thrust
        n_engine = self.aircraft.power_system.n_engine
        self.engine_mass = (1250. + 0.021*reference_thrust*self.thrust_factor)*n_engine       # statistical regression, all engines
        self.pylon_mass = 0.0031*reference_thrust*self.thrust_factor*n_engine
        self.mass = self.engine_mass + self.pylon_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of a pure turbofan engine (semi-empirical model)
        """
        reference_thrust = self.aircraft.power_system.reference_thrust

        kth =  0.475*mach**2 + 0.091*(self.engine_bpr/10.)**2 \
             - 0.283*mach*self.engine_bpr/10. \
             - 0.633*mach - 0.081*self.engine_bpr/10. + 1.192

        rho,sig = earth.air_density(pamb, tamb)
        vair = mach * earth.sound_speed(tamb)

        total_thrust0 =   reference_thrust \
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

class OutboardWingMountedTf0Nacelle(SemiEmpiricTf0Nacelle,OutboradWingMountedNacelle):
    def __init__(self, aircraft):
        super(OutboardWingMountedTf0Nacelle, self).__init__(aircraft)

class InboardWingMountedTf0Nacelle(SemiEmpiricTf0Nacelle,InboradWingMountedNacelle):
    def __init__(self, aircraft):
        super(InboardWingMountedTf0Nacelle, self).__init__(aircraft)

class RearFuselageMountedTf0Nacelle(SemiEmpiricTf0Nacelle,RearFuselageMountedNacelle):
    def __init__(self, aircraft):
        super(RearFuselageMountedTf0Nacelle, self).__init__(aircraft)


class SemiEmpiricTfNacelle(Component):

    def __init__(self, aircraft):
        super(SemiEmpiricTfNacelle, self).__init__(aircraft)

        class_name = "SemiEmpiricTfNacelle"

        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.86, MCL=0.78, MCR=0.70, FID=0.10)
        self.reference_offtake = 0.
        self.tune_factor = 1.
        self.engine_bpr = get_init(class_name,"engine_bpr", val=self.__turbofan_bpr())
        self.core_thrust_ratio = get_init(class_name,"core_thrust_ratio")
        self.propeller_efficiency = get_init(class_name,"propeller_efficiency")
        self.fan_efficiency = get_init(class_name,"fan_efficiency")
        self.lateral_margin = get_init(class_name,"lateral_margin")
        self.vertical_margin = get_init(class_name,"vertical_margin")
        self.hub_width = get_init(class_name,"hub_width")

        self.fan_width = None
        self.nozzle_area = None
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

    def y_wise_margin(self, n):
        if n==1: return 1.5 * self.lateral_margin * self.width
        elif n==2:  return 3.0 * self.lateral_margin * self.width
        elif n==3:  return 4.5 * self.lateral_margin * self.width

    def z_wise_margin(self):
        return (self.vertical_margin - 0.45)*self.width

    def eval_geometry(self):
        # Update power transfert in case of hybridization, here : set power offtake
        self.aircraft.power_system.update_power_transfert()

        reference_thrust = self.aircraft.power_system.reference_thrust

        disa = self.aircraft.requirement.cruise_disa
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        # Reset tune factor
        self.tune_factor = 1.

        # Get fan shaft power in cruise condition
        shaft_power,core_thrust = self.fan_shaft_power(pamb,tamb,mach,"MCR",throttle=1.,pw_offtake=self.reference_offtake)

        # Design nacelle according to this shaft power in cruise condition
        self.turbofan_nacelle_design(pamb,tamb,mach,shaft_power)

        mach = 0.25
        disa = 15.
        altp = 0.
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        # Compute thrust of this nacelle in reference conditions
        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO",pw_offtake=self.reference_offtake)

        # Set tune factor so that output of unitary_thrust matches the definition of the reference thrust
        self.tune_factor = reference_thrust / (dict["fn"]/0.80)

        self.frame_origin = self.locate_nacelle()

    def fan_shaft_power(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Fan shaft power of a pure turbofan engine (semi-empirical model)
        """
        reference_thrust = self.aircraft.power_system.reference_thrust
        kth =  0.475*mach**2 + 0.091*(self.engine_bpr/10.)**2 \
             - 0.283*mach*self.engine_bpr/10. \
             - 0.633*mach - 0.081*self.engine_bpr/10. + 1.192

        rho,sig = earth.air_density(pamb, tamb)
        vair = mach * earth.sound_speed(tamb)

        total_thrust0 =   reference_thrust \
                        * self.tune_factor \
                        * kth \
                        * getattr(self.rating_factor,rating) \
                        * throttle \
                        * sig**0.75
        core_thrust0 = total_thrust0 * self.core_thrust_ratio        # Core thrust
        fan_thrust0 = total_thrust0 * (1.-self.core_thrust_ratio)   # Fan thrust
        fan_power0 = fan_thrust0*vair/self.propeller_efficiency     # Available total shaft power for one engine

        return fan_power0-pw_offtake, core_thrust0

    def turbofan_nacelle_design(self,Pamb,Tamb,Mach,shaft_power):
        """Electrofan nacelle design
        """
        n_engine = self.aircraft.power_system.n_engine

        r,gam,Cp,Cv = earth.gas_data()
        Vair = Mach * earth.sound_speed(Tamb)

        # Electrical nacelle geometry : e-nacelle diameter is size by cruise conditions
        deltaV = 2.*Vair*(self.fan_efficiency/self.propeller_efficiency - 1.)      # speed variation produced by the fan

        pw_input = self.fan_efficiency*shaft_power     # kinetic energy produced by the fan

        Vinlet = Vair
        Vjet = Vinlet + deltaV
        q1 = 2.*pw_input / (Vjet**2 - Vinlet**2)    # Here, it is total air flow, including core flow

        MachInlet = Mach     # The inlet is in free stream
        Ptot = earth.total_pressure(Pamb, MachInlet)        # Stagnation pressure at inlet position
        Ttot = earth.total_temperature(Tamb, MachInlet)     # Stagnation temperature at inlet position

        MachFan = 0.5       # required Mach number at fan position
        CQoA1 = self.corrected_air_flow(Ptot,Ttot,MachFan)        # Corrected air flow per area at fan position

        eFanArea = q1/CQoA1     # Fan area around the hub
        fan_width = np.sqrt(self.hub_width**2 + 4*eFanArea/np.pi)        # Fan diameter

        TtotJet = Ttot + shaft_power/(q1*Cp)        # Stagnation pressure increases due to introduced work
        Tstat = TtotJet - 0.5*Vjet**2/Cp            # static temperature

        VsndJet = np.sqrt(gam*r*Tstat)                      # Sound velocity at nozzle exhaust
        MachJet = Vjet/VsndJet                              # Mach number at nozzle output
        PtotJet = earth.total_pressure(Pamb, MachJet)       # total pressure at nozzle exhaust (P = Pamb)

        CQoA2 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at nozzle output
        qf = q1 * self.engine_bpr/(1.+self.engine_bpr)              # Here, it is fan air flow only
        nozzle_area = qf/CQoA2                                      # Fan nozzle area around the core nozzle

        self.fan_width = fan_width
        self.nozzle_area = nozzle_area

        self.width = 1.20*fan_width      # Surrounding structure
        self.length = 1.50*self.width

        self.gross_wet_area = np.pi*self.width*self.length*n_engine
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

    def corrected_air_flow(self,Ptot,Ttot,Mach):
        """Computes the corrected air flow per square meter
        """
        r,gam,Cp,Cv = earth.gas_data()
        f_m = Mach*(1. + 0.5*(gam-1)*Mach**2)**(-(gam+1.)/(2.*(gam-1.)))
        cqoa = (np.sqrt(gam/r)*Ptot/np.sqrt(Ttot))*f_m
        return cqoa

    def eval_mass(self):
        n_engine = self.aircraft.power_system.n_engine

        mach = 0.25
        disa = 15.
        altp = 0.
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO",pw_offtake=self.reference_offtake)
        eff_ref_thrust = dict["fn"]/0.8
        self.engine_mass = (1250. + 0.021*eff_ref_thrust)*n_engine     # statistical regression, all engines
        self.pylon_mass = 0.0031*eff_ref_thrust*n_engine
        self.mass = self.engine_mass + self.pylon_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.]) # statistical regression

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        return self.unitary_thrust_free_stream(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

    def unitary_thrust_free_stream(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of an electrofan engine (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(q,pw_shaft,pamb,Ttot,Vair):
            Vinlet = Vair
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q + Vinlet**2)   # Supposing adapted nozzle
            TtotJet = Ttot + pw_shaft/(q*Cp)            # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
            VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                      # Mach number at nozzle output, ignoring when Mach > 1
            PtotJet = earth.total_pressure(pamb, MachJet)               # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            q0 = CQoA1*self.nozzle_area
            qf = q * self.engine_bpr/(1.+self.engine_bpr)               # Here, it is fan air flow only
            y = q0 - qf
            return y

        pw_shaft,core_thrust = self.fan_shaft_power(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

        Ptot = earth.total_pressure(pamb, mach)        # Total pressure at inlet position
        Ttot = earth.total_temperature(tamb, mach)     # Total temperature at inlet position
        Vair = mach * earth.sound_speed(tamb)

        fct_arg = (pw_shaft,pamb,Ttot,Vair)

        CQoA0 = self.corrected_air_flow(Ptot,Ttot,mach)       # Corrected air flow per area at fan position
        q0init = CQoA0*(0.25*np.pi*self.fan_width**2)

        # Computation of the air flow swallowed by the inlet
        output_dict = fsolve(fct, x0=q0init, args=fct_arg, full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        q0 = output_dict[0][0]
        qf = q0 * self.engine_bpr/(1.+self.engine_bpr)               # Here, it is fan air flow only

        Vinlet = Vair
        pw_input = self.fan_efficiency*pw_shaft
        Vjet = np.sqrt(2.*pw_input/q0 + Vinlet**2)
        fan_thrust = qf*(Vjet - Vinlet)

        total_thrust = fan_thrust + core_thrust

        sfc_ref = ( 0.4 + 1./self.engine_bpr**0.895 )/36000.
        fuel_flow = sfc_ref * total_thrust

        return {"fn":total_thrust, "ff":fuel_flow, "t4":None}

    def unitary_sc(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        return self.unitary_sc_free_stream(pamb,tamb,mach,rating,thrust,pw_offtake=pw_offtake)

    def unitary_sc_free_stream(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary power required of an electrofan engine delivering a given thrust (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(x_in,thrust,pamb,Ttot,Vair):
            q = x_in[0]
            throttle = x_in[1]
            pw_shaft,core_thrust = self.fan_shaft_power(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)
            Vinlet = Vair
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q + Vinlet**2)   # Supposing adapted nozzle
            TtotJet = Ttot + pw_shaft/(q*Cp)            # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
            VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                      # Mach number at nozzle output, ignoring when Mach > 1
            PtotJet = earth.total_pressure(pamb, MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            q0 = CQoA1*self.nozzle_area
            qf = q * self.engine_bpr/(1.+self.engine_bpr)               # Here, it is fan air flow only
            fn = qf*(Vjet - Vinlet)
            return [q0-qf, thrust-(fn+core_thrust)]

        Ptot = earth.total_pressure(pamb, mach)        # Total pressure at inlet position
        Ttot = earth.total_temperature(tamb, mach)     # Total temperature at inlet position
        Vair = mach * earth.sound_speed(tamb)

        fct_arg = (thrust,pamb,Ttot,Vair)

        CQoA0 = self.corrected_air_flow(Ptot,Ttot,mach)       # Corrected air flow per area at fan position
        q0init = CQoA0*(0.25*np.pi*self.fan_width**2)

        x_init = [q0init,0.5]

        # Computation of both air flow and shaft power
        output_dict = fsolve(fct, x0=x_init, args=fct_arg, full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        q0 = output_dict[0][0]
        thtl = output_dict[0][1]

        sfc = ( 0.4 + 1./self.engine_bpr**0.895 )/36000.

        return {"sfc":sfc, "thtl":thtl, "t4":None}

class OutboardWingMountedTfNacelle(SemiEmpiricTfNacelle,OutboradWingMountedNacelle):
    def __init__(self, aircraft):
        super(OutboardWingMountedTfNacelle, self).__init__(aircraft)

class InboardWingMountedTfNacelle(SemiEmpiricTfNacelle,InboradWingMountedNacelle):
    def __init__(self, aircraft):
        super(InboardWingMountedTfNacelle, self).__init__(aircraft)

class RearFuselageMountedTfNacelle(SemiEmpiricTfNacelle,RearFuselageMountedNacelle):
    def __init__(self, aircraft):
        super(RearFuselageMountedTfNacelle, self).__init__(aircraft)

class PodTailConeMountedTfNacelle(SemiEmpiricTfNacelle,PodTailConeMountedNacelle):
    def __init__(self, aircraft):
        super(PodTailConeMountedTfNacelle, self).__init__(aircraft)
        self.bli_effect = get_init("PodTailConeMountedNacelle","bli_effect")
        self.hub_width = get_init("PodTailConeMountedNacelle","hub_width")
        self.body_width = None
        self.body_length = None
        self.bnd_layer = None

    def eval_geometry(self):
        self.body_width = self.aircraft.airframe.tank.width
        self.body_length = self.aircraft.airframe.tank.length
        self.bnd_layer = self.tail_cone_boundary_layer(self.body_width,self.hub_width)

        # Update power transfert in case of hybridization, here : set power offtake
        self.aircraft.power_system.update_power_transfert()

        disa = self.aircraft.requirement.cruise_disa
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        # Reset tune factor
        self.tune_factor = 1.

        # Get fan shaft power in cruise condition
        shaft_power,core_thrust = self.fan_shaft_power(pamb,tamb,mach,"MCR",throttle=1.,pw_offtake=self.reference_offtake)

        # Design nacelle according to this shaft power in cruise condition
        self.turbofan_nacelle_design(pamb,tamb,mach,shaft_power)

        mach = 0.25
        disa = 15.
        altp = 0.
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        # Compute thrust of this nacelle in reference conditions
        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO",pw_offtake=self.reference_offtake)

        # Set tune factor so that output of unitary_thrust matches the definition of the reference thrust
        self.tune_factor = self.reference_thrust / (dict["fn"]/0.80)

        self.frame_origin = self.locate_nacelle()

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
        if self.bli_effect=="yes":
            dict_bli = self.unitary_thrust_bli(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)
            dict_fs = self.unitary_thrust_free_stream(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)
            return {"fn":dict_bli["fn"], "ff":dict_fs["ff"], "T4":None}
        else:
            return self.unitary_thrust_free_stream(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

    def unitary_thrust_bli(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of an electrofan engine (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(y,pw_shaft,pamb,rho,Ttot,Vair,r1,d1):
            q0,q1,q2,Vinlet,dVbli = self.air_flow(rho,Vair,r1,d1,y)
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q1 + Vinlet**2)      # Supposing adapted nozzle
            TtotJet = Ttot + pw_shaft/(q1*Cp)               # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp             # Static temperature
            VsndJet = earth.sound_speed(TstatJet)           # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                          # Mach number at nozzle output, ignoring when Mach > 1
            PtotJet = earth.total_pressure(pamb,MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            q = CQoA1*self.nozzle_area
            qf = q1 * self.engine_bpr/(1.+self.engine_bpr)               # Here, it is fan air flow only
            return q - qf

        pw_shaft,core_thrust = self.fan_shaft_power(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

        Re = earth.reynolds_number(pamb,tamb,mach)
        rho,sig = earth.air_density(pamb,tamb)
        Vair = mach * earth.sound_speed(tamb)
        Ttot = earth.total_temperature(tamb,mach)     # Total temperature at inlet position

        d0 = self.boundary_layer(Re,self.body_length)      # theorical thickness of the boundary layer without taking account of fuselage tapering
        r1 = 0.5*self.hub_width      # Radius of the hub of the eFan nacelle
        d1 = math.lin_interp_1d(d0,self.bnd_layer[:,0],self.bnd_layer[:,1])     # Using the precomputed relation

        fct_arg = (pw_shaft,pamb,rho,Ttot,Vair,r1,d1)

        # Computation of y1 : thikness of the vein swallowed by the inlet
        output_dict = fsolve(fct, x0=0.5, args=fct_arg, full_output=True)

        y1 = output_dict[0][0]
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        q0,q1,q2,Vinlet,dVbli = self.air_flow(rho,Vair,r1,d1,y1)

        pw_input = self.fan_efficiency*pw_shaft
        Vjet = np.sqrt(2.*pw_input/q1 + Vinlet**2)
        fan_thrust = q1*(Vjet - Vinlet)

        total_thrust = fan_thrust + core_thrust

        return {"fn":total_thrust}

    def unitary_sc(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        if self.bli_effect=="yes":
            dict_bli = self.unitary_sc_bli(pamb,tamb,mach,rating,thrust,pw_offtake=pw_offtake)
            throttle = dict_bli["thtl"]
            dict_fs = self.unitary_thrust_free_stream(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)
            return {"sfc":dict_fs["ff"]/thrust, "thtl":throttle, "t4":None}
        else:
            return self.unitary_sc_free_stream(pamb,tamb,mach,rating,thrust,pw_offtake=pw_offtake)

    def unitary_sc_bli(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary specific consumption of a turbofan with bli
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(x_in,thrust,pamb,rho,Ttot,Vair,r1,d1):
            y = x_in[0]
            throttle = x_in[1]
            q0,q1,q2,Vinlet,dVbli = self.air_flow(rho,Vair,r1,d1,y)
            pw_shaft,core_thrust = self.fan_shaft_power(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q1 + Vinlet**2)  # Supposing adapted nozzle
            TtotJet = Ttot + pw_shaft/(q1*Cp)           # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
            VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                      # Mach number at nozzle output, ignoring when Mach > 1
            PtotJet = earth.total_pressure(pamb, MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            q = CQoA1*self.nozzle_area
            qf = q1 * self.engine_bpr/(1.+self.engine_bpr)               # Here, it is fan air flow only
            fn = qf*(Vjet - Vinlet)
            return [q-qf, thrust-(fn+core_thrust)]

        Re = earth.reynolds_number(pamb,tamb,mach)
        rho,sig = earth.air_density(pamb,tamb)
        Vair = mach * earth.sound_speed(tamb)
        Ttot = earth.total_temperature(tamb,mach)     # Total temperature at inlet position

        d0 = self.boundary_layer(Re,self.body_length)      # theorical thickness of the boundary layer without taking account of fuselage tapering
        r1 = 0.5*self.hub_width      # Radius of the hub of the eFan nacelle
        d1 = math.lin_interp_1d(d0,self.bnd_layer[:,0],self.bnd_layer[:,1])     # Using the precomputed relation

        fct_arg = (thrust,pamb,rho,Ttot,Vair,r1,d1)

        x_init = [0.5,0.5]

        # Computation of both air flow and shaft power
        output_dict = fsolve(fct, x0=x_init, args=fct_arg, full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        y1 = output_dict[0][0]
        thtl = output_dict[0][1]

        return {"thtl":thtl}


class SemiEmpiricTpNacelle(Component):

    def __init__(self, aircraft):
        super(SemiEmpiricTpNacelle, self).__init__(aircraft)

        class_name = "SemiEmpiricTpNacelle"

        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.95, MCL=0.90, MCR=0.70, FID=0.10)
        self.propeller_efficiency = get_init(class_name,"propeller_efficiency")
        self.propeller_disk_load = get_init(class_name,"propeller_disk_load")
        self.lateral_margin = get_init(class_name,"lateral_margin")
        self.hub_width = get_init(class_name,"hub_width")
        self.engine_bpr = 100.

        self.propeller_width = None
        self.width = None
        self.length = None
        self.propeller_mass = None

        self.propeller_mass = 0.
        self.engine_mass = 0.
        self.pylon_mass = 0.

        self.frame_origin = np.full(3,None)

    def y_wise_margin(self, n):
        if n==1: return 0.6 * self.lateral_margin * self.propeller_width
        elif n==2:  return 1.8 * self.lateral_margin * self.propeller_width
        elif n==3:  return 3.0 * self.lateral_margin * self.propeller_width

    def z_wise_margin(self):
        return 0.

    def eval_geometry(self):
        reference_power = self.aircraft.power_system.reference_power
        n_engine = self.aircraft.power_system.n_engine

        # info : reference_thrust is defined by thrust(mach=0.25, altp=0, disa=15) / 0.80
        mach = 0.25
        disa = 15.
        altp = 0.

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO")
        self.reference_thrust = dict["fn"] / 0.80

        self.propeller_width = np.sqrt((4./np.pi)*(self.reference_thrust/self.propeller_disk_load))      # Assuming 3000 N/m2

        self.width = 0.25*(reference_power/1.e3)**0.2        # statistical regression
        self.length = 0.84*(reference_power/1.e3)**0.2       # statistical regression

        self.gross_wet_area = 2.8*(self.width*self.length)*n_engine     # statistical regression
        self.net_wet_area = 0.80*self.gross_wet_area
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        reference_power = self.aircraft.power_system.reference_power
        n_engine = self.aircraft.power_system.n_engine

        self.engine_mass = (0.633*(reference_power/1.e3)**0.9)*n_engine       # statistical regression
        self.propeller_mass = (165./1.5e6)*reference_power * n_engine
        self.mass = self.engine_mass + self.propeller_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of a pure turboprop engine (semi-empirical model)
        """
        reference_power = self.aircraft.power_system.reference_power

        factor = self.rating_factor
        eta_prop = self.propeller_efficiency

        psfc_ref = unit.kgpWps_lbpshpph(0.4)   # 0.4 lb/shp/h

        rho,sig = earth.air_density(pamb,tamb)
        Vsnd = earth.sound_speed(tamb)
        Vair = Vsnd*mach

        shaft_power = throttle*getattr(factor,rating)*reference_power*sig**0.5 - pw_offtake

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

        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.90, MCL=0.90, MCR=0.90, FID=0.10)
        self.propeller_efficiency = get_init(class_name,"propeller_efficiency")
        self.propeller_disk_load = get_init(class_name,"propeller_disk_load")
        self.motor_efficiency = get_init(class_name,"motor_efficiency")
        self.controller_efficiency = get_init(class_name,"controller_efficiency")
        self.controller_pw_density = get_init(class_name,"controller_pw_density")
        self.nacelle_pw_density = get_init(class_name,"nacelle_pw_density")
        self.motor_pw_density = get_init(class_name,"motor_pw_density")
        self.lateral_margin = get_init(class_name,"lateral_margin")
        self.hub_width = get_init(class_name,"hub_width")
        self.engine_bpr = 100.

        self.propeller_width = None
        self.width = None
        self.length = None

        self.propeller_mass = 0.
        self.engine_mass = 0.
        self.pylon_mass = 0.

        self.frame_origin = np.full(3,None)

    def y_wise_margin(self, n):
        if n==1: return 0.6 * self.lateral_margin * self.propeller_width
        elif n==2:  return 1.8 * self.lateral_margin * self.propeller_width
        elif n==3:  return 3.0 * self.lateral_margin * self.propeller_width

    def z_wise_margin(self):
        return 0.

    def eval_geometry(self):
        reference_power = self.aircraft.power_system.reference_power
        n_engine = self.aircraft.power_system.n_engine

        # info : reference_thrust is defined by thrust(mach=0.25, altp=0, disa=15) / 0.80
        mach = 0.25
        disa = 15.
        altp = 0.

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO")
        self.reference_thrust = dict["fn"] / 0.80

        self.propeller_width = np.sqrt((4./np.pi)*(self.reference_thrust/self.propeller_disk_load))      # Assuming 3000 N/m2

        self.width = 0.15*(reference_power/1.e3)**0.2        # statistical regression
        self.length = 0.55*(reference_power/1.e3)**0.2       # statistical regression

        self.gross_wet_area = 2.8*(self.width*self.length)*n_engine     # statistical regression
        self.net_wet_area = 0.80*self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        reference_power = self.aircraft.power_system.reference_power
        n_engine = self.aircraft.power_system.n_engine

        self.engine_mass = (  1./self.controller_pw_density + 1./self.motor_pw_density
                            + 1./self.nacelle_pw_density
                           ) * reference_power * n_engine
        self.propeller_mass = (165./1.5e6)*reference_power * n_engine
        self.mass = self.engine_mass + self.propeller_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of a pure turboprop engine (semi-empirical model)
        """
        reference_power = self.aircraft.power_system.reference_power
        Vsnd = earth.sound_speed(tamb)
        Vair = Vsnd*mach
        pw_shaft = reference_power*getattr(self.rating_factor,rating)*throttle - pw_offtake
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

class ExternalWingMountedEpNacelle(SemiEmpiricEpNacelle,ExternalWingMountedNacelle):
    def __init__(self, aircraft):
        super(ExternalWingMountedEpNacelle, self).__init__(aircraft)

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

        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.90, MCL=0.90, MCR=0.90, FID=0.10)
        self.propeller_efficiency = get_init(class_name,"propeller_efficiency")
        self.fan_efficiency = get_init(class_name,"fan_efficiency")
        self.motor_efficiency = get_init(class_name,"motor_efficiency")
        self.controller_efficiency = get_init(class_name,"controller_efficiency")
        self.controller_pw_density = get_init(class_name,"controller_pw_density")
        self.nacelle_pw_density = get_init(class_name,"nacelle_pw_density")
        self.motor_pw_density = get_init(class_name,"motor_pw_density")
        self.lateral_margin = get_init(class_name,"lateral_margin")
        self.vertical_margin = get_init(class_name,"vertical_margin")
        self.hub_width = get_init(class_name,"hub_width")

        self.fan_width = None
        self.nozzle_area = None
        self.width = None
        self.length = None

        self.propeller_mass = 0.
        self.engine_mass = 0.
        self.pylon_mass = 0.

        self.frame_origin = np.full(3,None)

    def y_wise_margin(self, n):
        if n==1: return 1.5 * self.lateral_margin * self.width
        elif n==2:  return 3.0 * self.lateral_margin * self.width
        elif n==3:  return 4.5 * self.lateral_margin * self.width

    def z_wise_margin(self):
        return (self.vertical_margin - 0.45)*self.width

    def eval_geometry(self):
        reference_power = self.aircraft.power_system.reference_power

        # Electric fan geometry is design for MCR in cruise condition
        disa = self.aircraft.requirement.cruise_disa
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        shaft_power = reference_power*self.rating_factor.MCR

        self.efan_nacelle_design(pamb,tamb,mach,shaft_power)

        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        reference_power = self.aircraft.power_system.reference_power
        n_engine = self.aircraft.power_system.n_engine

        self.engine_mass = (  1./self.controller_pw_density + 1./self.motor_pw_density
                            + 1./self.nacelle_pw_density
                          ) * reference_power * n_engine
        self.pylon_mass = 0.0031*self.reference_thrust*n_engine
        self.mass = self.engine_mass + self.pylon_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression

    def efan_nacelle_design(self,Pamb,Tamb,Mach,shaft_power):
        """Electrofan nacelle design
        """
        n_engine = self.aircraft.power_system.n_engine

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

        TtotJet = Ttot + shaft_power/(q1*Cp)    # Stagnation pressure increases due to introduced work
        Tstat = TtotJet - 0.5*Vjet**2/Cp        # static temperature

        VsndJet = np.sqrt(gam*r*Tstat)                      # Sound velocity at nozzle exhaust
        MachJet = Vjet/VsndJet                              # Mach number at nozzle output
        PtotJet = earth.total_pressure(Pamb, MachJet)       # total pressure at nozzle exhaust (P = Pamb)

        CQoA2 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)     # Corrected air flow per area at nozzle output
        nozzle_area = q1/CQoA2        # Fan area around the hub

        self.fan_width = fan_width
        self.nozzle_area = nozzle_area

        self.width = 1.20*fan_width      # Surrounding structure
        self.length = 1.50*self.width

        self.gross_wet_area = np.pi*self.width*self.length*n_engine
        self.net_wet_area = self.gross_wet_area

    def corrected_air_flow(self,Ptot,Ttot,Mach):
        """Computes the corrected air flow per square meter
        """
        r,gam,Cp,Cv = earth.gas_data()
        f_m = Mach*(1. + 0.5*(gam-1)*Mach**2)**(-(gam+1.)/(2.*(gam-1.)))
        cqoa = (np.sqrt(gam/r)*Ptot/np.sqrt(Ttot))*f_m
        return cqoa

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        return self.unitary_thrust_free_stream(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

    def unitary_thrust_free_stream(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of an electrofan engine (semi-empirical model)
        """
        reference_power = self.aircraft.power_system.reference_power

        r,gam,Cp,Cv = earth.gas_data()

        def fct(q,pw_shaft,pamb,Ttot,Vair):
            Vinlet = Vair
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q + Vinlet**2)   # Supposing adapted nozzle
            TtotJet = Ttot + pw_shaft/(q*Cp)            # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
            VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                      # Mach number at nozzle output, ignoring when Mach > 1
            PtotJet = earth.total_pressure(pamb, MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            q0 = CQoA1*self.nozzle_area
            y = q0 - q
            return y

        pw_shaft = reference_power*getattr(self.rating_factor,rating)*throttle - pw_offtake
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
        return self.unitary_sc_free_stream(pamb,tamb,mach,rating,thrust,pw_offtake=pw_offtake)

    def unitary_sc_free_stream(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary power required of an electrofan engine delivering a given thrust (semi-empirical model)
        """
        reference_power = self.aircraft.power_system.reference_power

        r,gam,Cp,Cv = earth.gas_data()

        def fct(x_in,thrust,pamb,Ttot,Vair):
            q = x_in[0]
            pw_shaft = x_in[1]
            Vinlet = Vair
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q + Vinlet**2)   # Supposing adapted nozzle
            TtotJet = Ttot + pw_shaft/(q*Cp)            # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
            VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                      # Mach number at nozzle output, ignoring when Mach > 1
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
        PWinit = reference_power*getattr(self.rating_factor,rating) - pw_offtake
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

        throttle = (Pw+pw_offtake)/(reference_power*getattr(self.rating_factor,rating))
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
        self.bli_effect = get_init("FuselageTailConeMountedNacelle","bli_effect")
        self.hub_width = get_init("FuselageTailConeMountedNacelle","hub_width")
        self.body_width = None
        self.body_length = None
        self.bnd_layer = None

    def eval_geometry(self):
        self.reference_power = self.aircraft.airframe.system.chain_power
        self.body_width = self.aircraft.airframe.body.width
        self.body_length = self.aircraft.airframe.body.length
        self.bnd_layer = self.tail_cone_boundary_layer(self.body_width,self.hub_width)

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
        if self.bli_effect=="yes":
            return self.unitary_thrust_bli(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)
        else:
            return self.unitary_thrust_free_stream(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

    def unitary_thrust_bli(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of an electrofan engine (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(y,pw_shaft,pamb,rho,Ttot,Vair,r1,d1):
            q0,q1,q2,Vinlet,dVbli = self.air_flow(rho,Vair,r1,d1,y)
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q1 + Vinlet**2)      # Supposing adapted nozzle
            TtotJet = Ttot + pw_shaft/(q1*Cp)               # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp             # Static temperature
            VsndJet = earth.sound_speed(TstatJet)           # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                          # Mach number at nozzle output, ignoring when Mach > 1
            PtotJet = earth.total_pressure(pamb,MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            q = CQoA1*self.nozzle_area
            return q1 - q

        pw_shaft = self.reference_power*getattr(self.rating_factor,rating)*throttle - pw_offtake
        pw_elec = pw_shaft / (self.controller_efficiency*self.motor_efficiency)

        Re = earth.reynolds_number(pamb,tamb,mach)
        rho,sig = earth.air_density(pamb,tamb)
        Vair = mach * earth.sound_speed(tamb)
        Ttot = earth.total_temperature(tamb,mach)     # Total temperature at inlet position

        d0 = self.boundary_layer(Re,self.body_length)      # theorical thickness of the boundary layer without taking account of fuselage tapering
        r1 = 0.5*self.hub_width      # Radius of the hub of the eFan nacelle
        d1 = math.lin_interp_1d(d0,self.bnd_layer[:,0],self.bnd_layer[:,1])     # Using the precomputed relation

        fct_arg = (pw_shaft,pamb,rho,Ttot,Vair,r1,d1)

        # Computation of y1 : thikness of the vein swallowed by the inlet
        output_dict = fsolve(fct, x0=0.5, args=fct_arg, full_output=True)

        y1 = output_dict[0][0]
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        q0,q1,q2,Vinlet,dVbli = self.air_flow(rho,Vair,r1,d1,y1)

        pw_input = self.fan_efficiency*pw_shaft
        Vjet = np.sqrt(2.*pw_input/q1 + Vinlet**2)
        eFn = q1*(Vjet - Vinlet)

        return {"fn":eFn, "pw":pw_elec, "dv_bli":dVbli}

    def unitary_sc(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        if self.bli_effect=="yes":
            return self.unitary_sc_bli(pamb,tamb,mach,rating,thrust,pw_offtake=pw_offtake)
        else:
            return self.unitary_sc_free_stream(pamb,tamb,mach,rating,thrust,pw_offtake=pw_offtake)

    def unitary_sc_bli(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary power required of an electrofan engine delivering a given thrust (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(x_in,thrust,pamb,rho,Ttot,Vair,r1,d1):
            y = x_in[0]
            pw_shaft = x_in[1]
            q0,q1,q2,Vinlet,dVbli = self.air_flow(rho,Vair,r1,d1,y)
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q1 + Vinlet**2)  # Supposing adapted nozzle
            TtotJet = Ttot + pw_shaft/(q1*Cp)           # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
            VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                      # Mach number at nozzle output, ignoring when Mach > 1
            PtotJet = earth.total_pressure(pamb, MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            q = CQoA1*self.nozzle_area
            eFn = q*(Vjet - Vinlet)
            return [q1-q, thrust-eFn]

        Re = earth.reynolds_number(pamb,tamb,mach)
        rho,sig = earth.air_density(pamb,tamb)
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










