#!/usr/bin/env python3
"""
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC

.. note:: All physical parameters are given in SI units.
"""

import numpy as np
from scipy.optimize import fsolve

from marilib.utils import earth, unit, math

from marilib.aircraft.airframe.component import Nacelle



class InboardWingMountedNacelle(Nacelle):

    def __init__(self, aircraft):
        super(InboardWingMountedNacelle, self).__init__(aircraft)

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

        return np.array([x_int, y_int*self.get_side(), z_int])

    def get_component_type(self):
        return "wing_nacelle"

class OutboardWingMountedNacelle(Nacelle):

    def __init__(self, aircraft):
        super(OutboardWingMountedNacelle, self).__init__(aircraft)

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

        return np.array([x_ext, y_ext*self.get_side(), z_ext])

    def get_component_type(self):
        return "wing_nacelle"

class ExternalWingMountedNacelle(Nacelle):

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

        return np.array([x_ext, y_ext*self.get_side(), z_ext])

    def get_component_type(self):
        return "wing_nacelle"

class RearFuselageMountedNacelle(Nacelle):

    def __init__(self, aircraft):
        super(RearFuselageMountedNacelle, self).__init__(aircraft)

    def locate_nacelle(self):
        body_width = self.aircraft.airframe.body.width
        body_height = self.aircraft.airframe.body.height
        body_length = self.aircraft.airframe.body.length

        y_int = 0.5 * body_width + 0.7 * self.width      # statistical regression
        x_int = 0.80 * body_length - self.length
        z_int = body_height

        return np.array([x_int, y_int*self.get_side(), z_int])

    def get_component_type(self):
        return "body_nacelle"

class BodyTailConeMountedNacelle(Nacelle):

    def __init__(self, aircraft):
        super(BodyTailConeMountedNacelle, self).__init__(aircraft)

        self.tail_cone_height_ratio = aircraft.get_init("BodyTailConeMountedNacelle","tail_cone_height_ratio")
        self.specific_nacelle_cost = aircraft.get_init("BodyTailConeMountedNacelle","specific_nacelle_cost")
        self.bli_effect = aircraft.get_init("BodyTailConeMountedNacelle","bli_effect")
        self.hub_width = aircraft.get_init("BodyTailConeMountedNacelle","hub_width")
        self.body_width = None
        self.body_length = None
        self.bnd_layer = None

    def locate_nacelle(self):
        self.body_width = self.aircraft.airframe.body.width
        self.body_length = self.aircraft.airframe.body.length
        self.bnd_layer = self.aircraft.aerodynamics.tail_cone_boundary_layer(self.body_width,self.hub_width)

        body_origin = self.aircraft.airframe.body.frame_origin
        body_height = self.aircraft.airframe.body.height
        body_length = self.aircraft.airframe.body.length

        y_axe = body_origin[1]
        x_axe = body_origin[0] + body_length
        z_axe = body_origin[2] + self.tail_cone_height_ratio * body_height

        return np.array([x_axe, y_axe, z_axe])

    def get_component_type(self):
        return "body_tail_nacelle"

class PodTailConeMountedNacelle(Nacelle):

    def __init__(self, aircraft):
        super(PodTailConeMountedNacelle, self).__init__(aircraft)

        self.lateral_margin = aircraft.get_init("PodTailConeMountedNacelle","lateral_margin")
        self.x_loc_ratio = aircraft.get_init("PodTailConeMountedNacelle","x_loc_ratio")
        self.z_loc_ratio = aircraft.get_init("PodTailConeMountedNacelle","z_loc_ratio")
        self.specific_nacelle_cost = aircraft.get_init("PodTailConeMountedNacelle","specific_nacelle_cost")
        self.hub_width = aircraft.get_init("PodTailConeMountedNacelle","hub_width")
        self.bli_effect = aircraft.get_init("PodTailConeMountedNacelle","bli_effect")
        self.body_width = None
        self.body_length = None
        self.bnd_layer = None

    def locate_nacelle(self):
        self.body_width = self.aircraft.airframe.tank.width
        self.body_length = self.aircraft.airframe.tank.length
        self.bnd_layer = self.aircraft.aerodynamics.tail_cone_boundary_layer(self.body_width,self.hub_width)

        body_width = self.aircraft.airframe.body.width
        tank_width = self.aircraft.airframe.tank.width

        # Locate nacelle
        x_int = self.aircraft.airframe.tank.frame_origin[0] + self.aircraft.airframe.tank.length
        y_int = self.aircraft.airframe.tank.frame_origin[1]
        z_int = self.aircraft.airframe.tank.frame_origin[2]

        return np.array([x_int, y_int*self.get_side(), z_int])

    def get_component_type(self):
        return "pod_tail_nacelle"

class PiggyBackTailConeMountedNacelle(Nacelle):

    def __init__(self, aircraft):
        super(PiggyBackTailConeMountedNacelle, self).__init__(aircraft)

        self.specific_nacelle_cost = aircraft.get_init("PiggyBackTailConeMountedNacelle","specific_nacelle_cost")
        self.hub_width = aircraft.get_init("PiggyBackTailConeMountedNacelle","hub_width")
        self.bli_effect = aircraft.get_init("PiggyBackTailConeMountedNacelle","bli_effect")
        self.body_width = None
        self.body_length = None
        self.bnd_layer = None

    def locate_nacelle(self):
        self.body_width = self.aircraft.airframe.tank.width
        self.body_length = self.aircraft.airframe.tank.length
        self.bnd_layer = self.aircraft.aerodynamics.tail_cone_boundary_layer(self.body_width,self.hub_width)

        tank_frame = self.aircraft.airframe.tank.frame_origin

        # Locate nacelle
        y_axe = tank_frame[1]
        x_axe = tank_frame[0] + self.aircraft.airframe.tank.length
        z_axe = tank_frame[2]

        return np.array([x_axe, y_axe, z_axe])

    def get_component_type(self):
        return "piggyback_tail_nacelle"



class RatingFactor(object):
    def __init__(self, MTO=None, MCN=None, MCL=None, MCR=None, FID=None):
        self.MTO = MTO
        self.MCN = MCN
        self.MCL = MCL
        self.MCR = MCR
        self.FID = FID



class SemiEmpiricTf0Nacelle(object):

    def __init__(self, aircraft):
        super(SemiEmpiricTf0Nacelle, self).__init__(aircraft)

        class_name = "SemiEmpiricTf0Nacelle"

        ne = self.aircraft.arrangement.number_of_engine
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range

        self.eis_date = aircraft.get_init(class_name,"eis_date")
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.86, MCL=0.78, MCR=0.70, FID=0.05)
        self.reference_offtake = 0.
        self.tune_factor = 1.
        self.engine_bpr = aircraft.get_init(class_name,"engine_bpr", val=self.__turbofan_bpr())
        self.engine_opr = aircraft.get_init(class_name,"engine_opr", val=self.__turbofan_opr())
        self.core_thrust_ratio = aircraft.get_init(class_name,"core_thrust_ratio")
        self.propeller_efficiency = aircraft.get_init(class_name,"propeller_efficiency")
        self.lateral_margin = aircraft.get_init(class_name,"lateral_margin")
        self.vertical_margin = aircraft.get_init(class_name,"vertical_margin")

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

    def __turbofan_opr(self):
        opr = 50.
        return opr

    def y_wise_margin(self, n):
        if n==1: return 1.5 * self.lateral_margin * self.width
        elif n==2:  return 3.0 * self.lateral_margin * self.width
        elif n==3:  return 4.5 * self.lateral_margin * self.width

    def z_wise_margin(self):
        return (self.vertical_margin - 0.45)*self.width

    def eval_geometry(self):
        # Set nominal power offtake
        self.reference_offtake = self.aircraft.airframe.system.get_reference_offtake()

        reference_thrust = self.aircraft.power_system.get_reference_thrust()

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
        self.gross_wet_area = knac*(1.48 - 0.0076*knac)       # statistical regression, all engines
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        reference_thrust = self.aircraft.power_system.reference_thrust
        self.engine_mass = (1250. + 0.021*reference_thrust*self.thrust_factor)       # statistical regression, all engines
        self.pylon_mass = 0.0031*reference_thrust*self.thrust_factor
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

class OutboardWingMountedTf0Nacelle(SemiEmpiricTf0Nacelle,OutboardWingMountedNacelle):
    def __init__(self, aircraft, side):
        super(OutboardWingMountedTf0Nacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

class InboardWingMountedTf0Nacelle(SemiEmpiricTf0Nacelle,InboardWingMountedNacelle):
    def __init__(self, aircraft, side):
        super(InboardWingMountedTf0Nacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

class RearFuselageMountedTf0Nacelle(SemiEmpiricTf0Nacelle,RearFuselageMountedNacelle):
    def __init__(self, aircraft, side):
        super(RearFuselageMountedTf0Nacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)




class SemiEmpiricTfNacelle(object):

    def __init__(self, aircraft):
        super(SemiEmpiricTfNacelle, self).__init__(aircraft)

        class_name = "SemiEmpiricTfNacelle"

        self.eis_date = aircraft.get_init(class_name,"eis_date")
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.86, MCL=0.78, MCR=0.70, FID=0.05)
        self.reference_offtake = 0.
        self.tune_factor = 1.

        self.engine_bpr = aircraft.get_init(class_name,"engine_bpr", val=self.__turbofan_bpr())
        self.engine_opr = aircraft.get_init(class_name,"engine_opr", val=self.__turbofan_opr())
        self.core_thrust_ratio = aircraft.get_init(class_name,"core_thrust_ratio")
        self.propeller_efficiency = aircraft.get_init(class_name,"propeller_efficiency")
        self.fan_efficiency = aircraft.get_init(class_name,"fan_efficiency")
        self.design_delta_air_speed = aircraft.get_init(class_name,"design_delta_air_speed")

        self.lateral_margin = aircraft.get_init(class_name,"lateral_margin")
        self.vertical_margin = aircraft.get_init(class_name,"vertical_margin")
        self.hub_width = aircraft.get_init(class_name,"hub_width")

        self.engine_fpr = None
        self.fan_width = None
        self.nozzle_width = None
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

    def __turbofan_opr(self):
        opr = 50.
        return opr

    def y_wise_margin(self, n):
        if n==1: return 1.5 * self.lateral_margin * self.width
        elif n==2:  return 3.0 * self.lateral_margin * self.width
        elif n==3:  return 4.5 * self.lateral_margin * self.width

    def z_wise_margin(self):
        return (self.vertical_margin - 0.45)*self.width

    def eval_geometry(self):
        # Set nominal power offtake
        self.reference_offtake = self.aircraft.airframe.system.get_reference_offtake()

        reference_thrust = self.aircraft.power_system.get_reference_thrust()

        disa = self.aircraft.requirement.cruise_disa
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        # Reset tune factor
        self.tune_factor = 1.

        # Get fan shaft power in cruise condition
        shaft_power,core_thrust,fuel_flow = self.fan_shaft_power(pamb,tamb,mach,"MCR",throttle=1.,pw_offtake=self.reference_offtake)

        # Design nacelle according to this shaft power in cruise condition
        self.turbofan_nacelle_design(pamb,tamb,mach,shaft_power)

        self.frame_origin = self.locate_nacelle()

        mach = 0.25
        disa = 15.
        altp = 0.
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        # Compute thrust of this nacelle in reference conditions
        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO",pw_offtake=self.reference_offtake)

        # Set tune factor so that output of unitary_thrust matches the definition of the reference thrust
        self.tune_factor = reference_thrust / (dict["fn"]/0.80)

    def fan_shaft_power(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Fan shaft power of a pure turbofan engine (semi-empirical model)
        """
        sfc_ref = ( 0.4 + 1./self.engine_bpr**0.895 )/36000.

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
        fuel_flow = sfc_ref * total_thrust0                         # Fuel flow
        core_thrust0 = total_thrust0 * self.core_thrust_ratio       # Core thrust
        fan_thrust0 = total_thrust0 * (1.-self.core_thrust_ratio)   # Fan thrust
        fan_power0 = fan_thrust0*vair/self.propeller_efficiency     # Available total shaft power for one engine

        return fan_power0-pw_offtake, core_thrust0, fuel_flow

    def turbofan_nacelle_design(self,Pamb,Tamb,Mach,shaft_power):
        """Electrofan nacelle design
        """
        r,gam,Cp,Cv = earth.gas_data()
        Vair = Mach * earth.sound_speed(Tamb)
        DeltaV = self.design_delta_air_speed

        r,gam,Cp,Cv = earth.gas_data()

        def fct(x,pw_shaft,Pamb,Ttot,Vair):
            nozzle_area = x[0]
            q = x[1]
            TtotJet = Ttot + pw_shaft/(q*Cp)
            PtotJet = Ptot * (1. + self.fan_efficiency*(TtotJet/Ttot-1.))**(gam/(gam-1.))
            MachJet = np.sqrt(((PtotJet/Pamb)**((gam-1.)/gam) - 1.) * (2./(gam-1.)))
            TstaJet = TtotJet / (1.+0.5*(gam-1.)*MachJet**2)
            Vjet = MachJet * np.sqrt(gam*r*TstaJet)
            qf = nozzle_area * self.corrected_air_flow(PtotJet,TtotJet,MachJet)
            qc = qf / self.engine_bpr
            return [qf+qc - q, Vjet - (Vair+DeltaV)]

        MachInlet = Mach     # The inlet is in free stream
        Ptot = earth.total_pressure(Pamb, MachInlet)        # Stagnation pressure at inlet position
        Ttot = earth.total_temperature(Tamb, MachInlet)     # Stagnation temperature at inlet position

        fct_arg = (shaft_power,Pamb,Ttot,Vair)

        m_dot_i = (2.*0.9*shaft_power) / ((Vair+DeltaV)**2 - Vair**2)
        MachJet_i = (Vair+DeltaV)/np.sqrt(gam*r*Tamb)
        PtotJet_i = Pamb * (1.+0.5*(gam-1.)*MachJet_i**2)
        TtotJet_i = Ttot*(((PtotJet_i/Ptot)**((gam-1.)/gam) - 1.)/self.fan_efficiency + 1.)
        m_dot_o_a = self.corrected_air_flow(PtotJet_i,TtotJet_i,MachJet_i)       # Corrected air flow per area at fan position
        nozzle_area_i = m_dot_i / m_dot_o_a
        x_init = [nozzle_area_i, m_dot_i]

        # Computation of the air flow swallowed by the inlet
        output_dict = fsolve(fct, x0=x_init, args=fct_arg, full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.nozzle_area = output_dict[0][0]
        m_dot_design = output_dict[0][1]

        MachFan = 0.5       # required Mach number at fan position
        m_dot_o_a = self.corrected_air_flow(Ptot,Ttot,MachFan)        # Corrected air flow per area at fan position
        fan_area = m_dot_design / m_dot_o_a
        self.fan_width = np.sqrt(self.hub_width**2 + 4.*fan_area/np.pi)        # Fan diameter
        self.nozzle_width = np.sqrt(self.hub_width**2 + 4.*self.nozzle_area/np.pi)        # Nozzle diameter

        TtotJet = Ttot + shaft_power/(m_dot_design*Cp)
        PtotJet = Ptot * (1. + self.fan_efficiency*(TtotJet/Ttot-1.))**(gam/(gam-1.))
        self.engine_fpr = PtotJet / Ptot









        self.width = self.fan_width + 0.30      # Surrounding structure
        self.length = 1.50*self.width

        self.gross_wet_area = np.pi*self.width*self.length
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
        mach = 0.25
        disa = 15.
        altp = 0.
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        dict = self.unitary_thrust(pamb,tamb,mach,rating="MTO",pw_offtake=self.reference_offtake)
        eff_ref_thrust = dict["fn"]/0.8
        self.engine_mass = (1250. + 0.021*eff_ref_thrust)     # statistical regression, all engines
        self.pylon_mass = 0.0031*eff_ref_thrust
        self.mass = self.engine_mass + self.pylon_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.]) # statistical regression

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        return self.unitary_thrust_free_stream(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

    def unitary_thrust_free_stream(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of an electrofan engine (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(q,pw_shaft,pamb,Ttot,Vair):
            TtotJet = Ttot + pw_shaft/(q*Cp)
            PtotJet = Ptot * (1. + self.fan_efficiency*(TtotJet/Ttot-1.))**(gam/(gam-1.))
            MachJet = np.sqrt(((PtotJet/pamb)**((gam-1.)/gam) - 1.) * (2./(gam-1.)))
            TstaJet = TtotJet / (1.+0.5*(gam-1.)*MachJet**2)
            qf = self.nozzle_area * self.corrected_air_flow(PtotJet,TtotJet,MachJet)
            qc = qf / self.engine_bpr
            return qf+qc - q

        pw_shaft,core_thrust,fuel_flow = self.fan_shaft_power(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

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
        TtotJet = Ttot + pw_shaft/(q0*Cp)
        PtotJet = Ptot * (1. + self.fan_efficiency*(TtotJet/Ttot-1.))**(gam/(gam-1.))
        MachJet = np.sqrt(((PtotJet/pamb)**((gam-1.)/gam) - 1.) * (2./(gam-1.)))
        TstaJet = TtotJet / (1.+0.5*(gam-1.)*MachJet**2)
        Vjet = MachJet * np.sqrt(gam*r*TstaJet)
        qf = self.nozzle_area * self.corrected_air_flow(PtotJet,TtotJet,MachJet)
        fan_thrust = qf*(Vjet - Vair)

        total_thrust = fan_thrust + core_thrust

        return {"fn":total_thrust, "ff":fuel_flow, "t4":None}

    def unitary_sc(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        return self.unitary_sc_free_stream(pamb,tamb,mach,rating,thrust,pw_offtake=pw_offtake)

    def unitary_sc_free_stream(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary power required of a turbofan engine delivering a given thrust (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(x_in,thrust,pamb,Ttot,Vair):
            q = x_in[0]
            throttle = x_in[1]
            pw_shaft,core_thrust,fuel_flow = self.fan_shaft_power(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)
            TtotJet = Ttot + pw_shaft/(q*Cp)
            PtotJet = Ptot * (1. + self.fan_efficiency*(TtotJet/Ttot-1.))**(gam/(gam-1.))
            MachJet = np.sqrt(((PtotJet/pamb)**((gam-1.)/gam) - 1.) * (2./(gam-1.)))
            TstaJet = TtotJet / (1.+0.5*(gam-1.)*MachJet**2)
            Vjet = MachJet * np.sqrt(gam*r*TstaJet)
            qf = self.nozzle_area * self.corrected_air_flow(PtotJet,TtotJet,MachJet)
            fan_thrust = qf*(Vjet - Vair)
            qc = qf / self.engine_bpr
            return [qf+qc-q, thrust-(fan_thrust+core_thrust)]

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

        throttle = output_dict[0][1]

        pw_shaft,core_thrust,fuel_flow = self.fan_shaft_power(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

        sfc = fuel_flow / thrust

        return {"sfc":sfc, "thtl":throttle, "t4":None}

class OutboardWingMountedTfNacelle(SemiEmpiricTfNacelle,OutboardWingMountedNacelle):
    def __init__(self, aircraft, side):
        super(OutboardWingMountedTfNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

class InboardWingMountedTfNacelle(SemiEmpiricTfNacelle,InboardWingMountedNacelle):
    def __init__(self, aircraft, side):
        super(InboardWingMountedTfNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

class RearFuselageMountedTfNacelle(SemiEmpiricTfNacelle,RearFuselageMountedNacelle):
    def __init__(self, aircraft, side):
        super(RearFuselageMountedTfNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)


class SemiEmpiricTfBliNacelle(SemiEmpiricTfNacelle):
    def __init__(self, aircraft):
        super(SemiEmpiricTfBliNacelle, self).__init__(aircraft)

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

    def boundary_layer(self,re,x_length):
        """Thickness of a turbulent boundary layer which developped turbulently from its starting point
        """
        return (0.385*x_length)/(re*x_length)**(1./5.)

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        if self.bli_effect=="yes":
            dict_bli = self.unitary_thrust_bli(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)
            dict_fs = self.unitary_thrust_free_stream(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)
            return {"fn":dict_bli["fn"], "ff":dict_fs["ff"], "t4":None}
        else:
            return self.unitary_thrust_free_stream(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

    def unitary_thrust_bli(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of an electrofan engine (semi-empirical model)
        """
        r,gam,Cp,Cv = earth.gas_data()

        def fct(y,pw_shaft,pamb,rho,Ptot,Ttot,Vair,r1,d1):
            q0,q1,q2,Vinlet,dVbli = self.air_flow(rho,Vair,r1,d1,y)
            TtotJet = Ttot + pw_shaft/(q1*Cp)
            PtotJet = Ptot * (1. + self.fan_efficiency*(TtotJet/Ttot-1.))**(gam/(gam-1.))
            MachJet = np.sqrt(((PtotJet/pamb)**((gam-1.)/gam) - 1.) * (2./(gam-1.)))
            TstaJet = TtotJet / (1.+0.5*(gam-1.)*MachJet**2)
            qf = self.nozzle_area * self.corrected_air_flow(PtotJet,TtotJet,MachJet)
            qc = qf / self.engine_bpr
            return qf+qc - q1

        pw_shaft,core_thrust,fuel_flow = self.fan_shaft_power(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

        Re = earth.reynolds_number(pamb,tamb,mach)
        rho,sig = earth.air_density(pamb,tamb)
        Vair = mach * earth.sound_speed(tamb)
        Ptot = earth.total_pressure(pamb,mach)        # Total pressure at inlet position
        Ttot = earth.total_temperature(tamb,mach)     # Total temperature at inlet position

        d0 = self.boundary_layer(Re,self.body_length)      # theorical thickness of the boundary layer without taking account of fuselage tapering
        r1 = 0.5*self.hub_width      # Radius of the hub of the eFan nacelle
        d1 = math.lin_interp_1d(d0,self.bnd_layer[:,0],self.bnd_layer[:,1])     # Using the precomputed relation

        fct_arg = (pw_shaft,pamb,rho,Ptot,Ttot,Vair,r1,d1)

        # Computation of y1 : thikness of the vein swallowed by the inlet
        output_dict = fsolve(fct, x0=0.5, args=fct_arg, full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        y1 = output_dict[0][0]

        q0,q1,q2,Vinlet,dVbli = self.air_flow(rho,Vair,r1,d1,y1)

        TtotJet = Ttot + pw_shaft/(q0*Cp)
        PtotJet = Ptot * (1. + self.fan_efficiency*(TtotJet/Ttot-1.))**(gam/(gam-1.))
        MachJet = np.sqrt(((PtotJet/pamb)**((gam-1.)/gam) - 1.) * (2./(gam-1.)))
        TstaJet = TtotJet / (1.+0.5*(gam-1.)*MachJet**2)
        Vjet = MachJet * np.sqrt(gam*r*TstaJet)
        qf = self.nozzle_area * self.corrected_air_flow(PtotJet,TtotJet,MachJet)
        fan_thrust = qf*(Vjet - Vair)

        total_thrust = fan_thrust + core_thrust

        return {"fn":total_thrust, "ff":fuel_flow, "t4":None}

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

        def fct(x_in,thrust,pamb,rho,Ptot,Ttot,Vair,r1,d1):
            y = x_in[0]
            throttle = x_in[1]
            q0,q1,q2,Vinlet,dVbli = self.air_flow(rho,Vair,r1,d1,y)
            pw_shaft,core_thrust,fuel_flow = self.fan_shaft_power(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)
            TtotJet = Ttot + pw_shaft/(q1*Cp)
            PtotJet = Ptot * (1. + self.fan_efficiency*(TtotJet/Ttot-1.))**(gam/(gam-1.))
            MachJet = np.sqrt(((PtotJet/pamb)**((gam-1.)/gam) - 1.) * (2./(gam-1.)))
            TstaJet = TtotJet / (1.+0.5*(gam-1.)*MachJet**2)
            Vjet = MachJet * np.sqrt(gam*r*TstaJet)
            qf = self.nozzle_area * self.corrected_air_flow(PtotJet,TtotJet,MachJet)
            fan_thrust = qf*(Vjet - Vair)
            qc = qf / self.engine_bpr
            return [qf+qc-q1, thrust-(fan_thrust+core_thrust)]

        Re = earth.reynolds_number(pamb,tamb,mach)
        rho,sig = earth.air_density(pamb,tamb)
        Vair = mach * earth.sound_speed(tamb)
        Ptot = earth.total_pressure(pamb,mach)        # Total pressure at inlet position
        Ttot = earth.total_temperature(tamb,mach)     # Total temperature at inlet position

        d0 = self.boundary_layer(Re,self.body_length)      # theorical thickness of the boundary layer without taking account of fuselage tapering
        r1 = 0.5*self.hub_width      # Radius of the hub of the eFan nacelle
        d1 = math.lin_interp_1d(d0,self.bnd_layer[:,0],self.bnd_layer[:,1])     # Using the precomputed relation

        fct_arg = (thrust,pamb,rho,Ptot,Ttot,Vair,r1,d1)

        x_init = [0.75,0.75]

        # Computation of both air flow and shaft power
        output_dict = fsolve(fct, x0=x_init, args=fct_arg, full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        throttle = output_dict[0][1]

        pw_shaft,core_thrust,fuel_flow = self.fan_shaft_power(pamb,tamb,mach,rating,throttle=throttle,pw_offtake=pw_offtake)

        sfc = fuel_flow / thrust

        return {"sfc":sfc, "thtl":throttle, "t4":None}

class PodTailConeMountedTfNacelle(SemiEmpiricTfBliNacelle,PodTailConeMountedNacelle):
    def __init__(self, aircraft, side):
        super(PodTailConeMountedTfNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

class BodyTailConeMountedTfNacelle(SemiEmpiricTfBliNacelle,BodyTailConeMountedNacelle):
    def __init__(self, aircraft):
        super(BodyTailConeMountedTfNacelle, self).__init__(aircraft)

class PiggyBackTailConeMountedTfNacelle(SemiEmpiricTfBliNacelle,PiggyBackTailConeMountedNacelle):
    def __init__(self, aircraft):
        super(PiggyBackTailConeMountedTfNacelle, self).__init__(aircraft)



class SemiEmpiricEfNacelle(object):

    def __init__(self, aircraft):
        super(SemiEmpiricEfNacelle, self).__init__(aircraft)

        class_name = "SemiEmpiricEfNacelle"

        self.eis_date = 2020
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.90, MCL=0.90, MCR=0.90, FID=0.05)
        self.propeller_efficiency = aircraft.get_init(class_name,"propeller_efficiency")
        self.fan_efficiency = aircraft.get_init(class_name,"fan_efficiency")
        self.motor_efficiency = aircraft.get_init(class_name,"motor_efficiency")
        self.controller_efficiency = aircraft.get_init(class_name,"controller_efficiency")
        self.controller_pw_density = aircraft.get_init(class_name,"controller_pw_density")
        self.nacelle_pw_density = aircraft.get_init(class_name,"nacelle_pw_density")
        self.motor_pw_density = aircraft.get_init(class_name,"motor_pw_density")
        self.lateral_margin = aircraft.get_init(class_name,"lateral_margin")
        self.vertical_margin = aircraft.get_init(class_name,"vertical_margin")
        self.hub_width = aircraft.get_init(class_name,"hub_width")

        self.engine_fpr = None
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
        reference_power = self.aircraft.power_system.get_reference_power(self.get_component_type())

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
        reference_power = self.aircraft.power_system.get_reference_power(self.get_component_type())

        self.engine_mass = (  1./self.controller_pw_density + 1./self.motor_pw_density
                            + 1./self.nacelle_pw_density
                          ) * reference_power
        self.pylon_mass = 0.0031*(0.82/87.26)*reference_power
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

        TtotJet = Ttot + shaft_power/(q1*Cp)    # Stagnation pressure increases due to introduced work
        Tstat = TtotJet - 0.5*Vjet**2/Cp        # static temperature

        VsndJet = np.sqrt(gam*r*Tstat)                      # Sound velocity at nozzle exhaust
        MachJet = Vjet/VsndJet                              # Mach number at nozzle output
        PtotJet = earth.total_pressure(Pamb, MachJet)       # total pressure at nozzle exhaust (P = Pamb)

        CQoA2 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)     # Corrected air flow per area at nozzle output
        nozzle_area = q1/CQoA2        # Fan area around the hub

        self.engine_fpr = PtotJet / Ptot

        self.fan_width = fan_width
        self.nozzle_area = nozzle_area

        self.width = fan_width + 0.30      # Surrounding structure
        self.length = 1.50*self.width

        self.gross_wet_area = np.pi*self.width*self.length
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
        reference_power = self.aircraft.power_system.get_reference_power(self.get_component_type())

        r,gam,Cp,Cv = earth.gas_data()

        def fct(q,pw_shaft,pamb,Ttot,Vair):
            Vinlet = Vair
            pw_input = self.fan_efficiency*pw_shaft
            Vjet = np.sqrt(2.*pw_input/q + Vinlet**2)   # Supposing adapted nozzle
            TtotJet = Ttot + pw_shaft/(q*Cp)            # Stagnation temperature increases due to introduced work
            TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
            VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
            MachJet = Vjet/VsndJet                      # Mach number at nozzle output, ignoring when Mach > 1
            # if MachJet>1.:
            #     qf = pw_shaft * (1.-gam+(1.+gam)*self.fan_efficiency) / (r*gam*Ttot-0.5*(gam+1.)*Vinlet**2)
            # else:
            PtotJet = earth.total_pressure(pamb, MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            qf = CQoA1*self.nozzle_area
            return qf-q

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
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        q0 = output_dict[0][0]

        Vinlet = Vair
        pw_input = self.fan_efficiency*pw_shaft
        Vjet = np.sqrt(2.*pw_input/q0 + Vinlet**2)
        eFn = q0*(Vjet - Vinlet)

        return {"fn":eFn, "pw":pw_elec, "sec":pw_elec/eFn}

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
            # if MachJet>1.:
            #     qf = pw_shaft * (1.-gam+(1.+gam)*self.fan_efficiency) / (r*gam*Ttot-0.5*(gam+1.)*Vinlet**2)
            # else:
            PtotJet = earth.total_pressure(pamb, MachJet)    # total pressure at nozzle exhaust (P = pamb)
            CQoA1 = self.corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
            qf = CQoA1*self.nozzle_area
            eFn = q*(Vjet - Vinlet)
            return [qf-q, thrust-eFn]

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

class OutboardWingMountedEfNacelle(SemiEmpiricEfNacelle,OutboardWingMountedNacelle):
    def __init__(self, aircraft, side):
        super(OutboardWingMountedEfNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

class InboardWingMountedEfNacelle(SemiEmpiricEfNacelle,InboardWingMountedNacelle):
    def __init__(self, aircraft, side):
        super(InboardWingMountedEfNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

class RearFuselageMountedEfNacelle(SemiEmpiricEfNacelle,RearFuselageMountedNacelle):
    def __init__(self, aircraft, side):
        super(RearFuselageMountedEfNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)


class SemiEmpiricEfBliNacelle(SemiEmpiricEfNacelle):
    def __init__(self, aircraft):
        super(SemiEmpiricEfBliNacelle, self).__init__(aircraft)

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

    def boundary_layer(self,re,x_length):
        """Thickness of a turbulent boundary layer which developped turbulently from its starting point
        """
        return (0.385*x_length)/(re*x_length)**(1./5.)

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
            qf = CQoA1*self.nozzle_area
            return qf - q1

        reference_power = self.aircraft.power_system.get_reference_power(self.get_component_type())
        pw_shaft = reference_power*getattr(self.rating_factor,rating)*throttle - pw_offtake
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
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        y1 = output_dict[0][0]

        q0,q1,q2,Vinlet,dVbli = self.air_flow(rho,Vair,r1,d1,y1)

        pw_input = self.fan_efficiency*pw_shaft
        Vjet = np.sqrt(2.*pw_input/q1 + Vinlet**2)
        eFn = q1*(Vjet - Vinlet)

        return {"fn":eFn, "pw":pw_elec, "sec":pw_elec/eFn, "dv_bli":dVbli}

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

        reference_power = self.aircraft.power_system.get_reference_power()

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

        throttle = (Pw+pw_offtake)/(self.reference_power*getattr(self.rating_factor,rating))
        pw_elec = Pw / (self.controller_efficiency*self.motor_efficiency)
        sec = pw_elec/eFn

        return {"sec":sec, "thtl":throttle}

class PodTailConeMountedEfNacelle(SemiEmpiricEfBliNacelle,PodTailConeMountedNacelle):
    def __init__(self, aircraft, side):
        super(PodTailConeMountedEfNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

class BodyTailConeMountedEfNacelle(SemiEmpiricEfBliNacelle,BodyTailConeMountedNacelle):
    def __init__(self, aircraft):
        super(BodyTailConeMountedEfNacelle, self).__init__(aircraft)

class PiggyBackTailConeMountedEfNacelle(SemiEmpiricEfBliNacelle,PiggyBackTailConeMountedNacelle):
    def __init__(self, aircraft):
        super(PiggyBackTailConeMountedEfNacelle, self).__init__(aircraft)



class SemiEmpiricTpNacelle(object):

    def __init__(self, aircraft):
        super(SemiEmpiricTpNacelle, self).__init__(aircraft)

        class_name = "SemiEmpiricTpNacelle"

        self.eis_date = 2020
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.95, MCL=0.90, MCR=0.70, FID=0.05)
        self.propeller_efficiency = aircraft.get_init(class_name,"propeller_efficiency")
        self.propeller_disk_load = aircraft.get_init(class_name,"propeller_disk_load")
        self.lateral_margin = aircraft.get_init(class_name,"lateral_margin")
        self.hub_width = aircraft.get_init(class_name,"hub_width")
        self.engine_bpr = 100.
        self.psfc_reference = aircraft.get_init(class_name,"psfc_reference")

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
        reference_power = self.aircraft.power_system.get_reference_power()

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

        self.gross_wet_area = 2.8*(self.width*self.length)     # statistical regression
        self.net_wet_area = 0.80*self.gross_wet_area
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        reference_power = self.aircraft.power_system.reference_power

        self.engine_mass = (0.633*(reference_power/1.e3)**0.9)       # statistical regression
        self.propeller_mass = (165./1.5e6)*reference_power
        self.mass = self.engine_mass + self.propeller_mass
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression

    def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.):
        """Unitary thrust of a pure turboprop engine (semi-empirical model)
        """
        reference_power = self.aircraft.power_system.reference_power

        factor = self.rating_factor
        eta_prop = self.propeller_efficiency

        rho,sig = earth.air_density(pamb,tamb)
        Vair = mach*earth.sound_speed(tamb)

        shaft_power = throttle*getattr(factor,rating)*reference_power*sig**0.5 - pw_offtake

        fn = eta_prop*shaft_power/Vair
        ff = self.psfc_reference*(shaft_power + pw_offtake)

        return {"fn":fn, "ff":ff, "pw":shaft_power, "t4":None}

    def unitary_sc(self,pamb,tamb,mach,rating,thrust,pw_offtake=0.):
        """Unitary thrust of a pure turbofan engine (semi-empirical model)
        """
        dict = self.unitary_thrust(pamb,tamb,mach,rating,pw_offtake=pw_offtake)
        throttle = thrust/dict["fn"]
        sfc = dict["ff"]/dict["pw"]     # Power SFC
        t41 = dict["t4"]
        return {"sfc":sfc, "thtl":throttle, "t4":t41}

class OutboardWingMountedTpNacelle(SemiEmpiricTpNacelle,OutboardWingMountedNacelle):
    def __init__(self, aircraft, side):
        super(OutboardWingMountedTpNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

class InboardWingMountedTpNacelle(SemiEmpiricTpNacelle,InboardWingMountedNacelle):
    def __init__(self, aircraft, side):
        super(InboardWingMountedTpNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)



class SemiEmpiricEpNacelle(object):

    def __init__(self, aircraft):
        super(SemiEmpiricEpNacelle, self).__init__(aircraft)

        class_name = "SemiEmpiricEpNacelle"

        self.eis_date = 2020
        self.rating_factor = RatingFactor(MTO=1.00, MCN=0.90, MCL=0.90, MCR=0.90, FID=0.05)
        self.propeller_efficiency = aircraft.get_init(class_name,"propeller_efficiency")
        self.propeller_disk_load = aircraft.get_init(class_name,"propeller_disk_load")
        self.motor_efficiency = aircraft.get_init(class_name,"motor_efficiency")
        self.controller_efficiency = aircraft.get_init(class_name,"controller_efficiency")
        self.controller_pw_density = aircraft.get_init(class_name,"controller_pw_density")
        self.nacelle_pw_density = aircraft.get_init(class_name,"nacelle_pw_density")
        self.motor_pw_density = aircraft.get_init(class_name,"motor_pw_density")
        self.lateral_margin = aircraft.get_init(class_name,"lateral_margin")
        self.hub_width = aircraft.get_init(class_name,"hub_width")
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
        reference_power = self.aircraft.power_system.get_reference_power()

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

        self.gross_wet_area = 2.8*(self.width*self.length)     # statistical regression
        self.net_wet_area = 0.80*self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.locate_nacelle()

    def eval_mass(self):
        reference_power = self.aircraft.power_system.reference_power

        self.engine_mass = (  1./self.controller_pw_density + 1./self.motor_pw_density
                            + 1./self.nacelle_pw_density
                           ) * reference_power
        self.propeller_mass = (165./1.5e6)*reference_power
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
        return {"fn":fn, "pw":pw_elec, "sec":pw_elec/fn}

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
    def __init__(self, aircraft, side):
        super(ExternalWingMountedEpNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

class OutboardWingMountedEpNacelle(SemiEmpiricEpNacelle,OutboardWingMountedNacelle):
    def __init__(self, aircraft, side):
        super(OutboardWingMountedEpNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

class InboardWingMountedEpNacelle(SemiEmpiricEpNacelle,InboardWingMountedNacelle):
    def __init__(self, aircraft, side):
        super(InboardWingMountedEpNacelle, self).__init__(aircraft)
        self.airplane_side = side
    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)











