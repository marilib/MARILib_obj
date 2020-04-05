#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""


import numpy as np

from aircraft.tool import unit
import earth

from engine.ExergeticEngine import Turbofan

from aircraft.airframe.component import Component


class Exergetic_turbofan_nacelle(Component):

    def __init__(self, aircraft):
        super(Exergetic_turbofan_nacelle, self).__init__(aircraft)

        ne = self.aircraft.arrangement.number_of_engine
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range

        self.n_engine = {"twin":2, "quadri":4}.get(ne, "number of engine is unknown")
        self.reference_thrust = (1.e5 + 177.*n_pax_ref*design_range*1.e-6)/self.n_engine
        self.reference_offtake = 0.
        self.reference_wBleed = 0.
        self.rating_factor = {"MTO":1.00, "MCN":0.86, "MCL":0.78, "MCR":0.70, "FID":0.10}
        self.engine_bpr = self.__turbofan_bpr__()
        self.engine_fpr = 1.66
        self.engine_lpc_pr = 2.4
        self.engine_hpc_pr = 14.
        self.engine_t41 = 1750.

        self.width = None
        self.length = None

        self.frame_origin = np.full(3,None)

    def __turbofan_bpr__(self):
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        if (80<n_pax_ref):
            bpr = 9.
        else:
            bpr = 5.
        return bpr

     def __locate_nacelle__(self):
        return np.full(3,None)

    def eval_geometry(self):

        # info : reference_thrust is defined by thrust(mach=0.25, altp=0, disa=15) / 0.80
        mach = 0.25
        disa = 15.
        altp = 0.

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        vair = mach*earth.sound_speed(tamb)



        TF = Turbofan()

        # Set the flight conditions to 35000ft, ISA, Mn 0.78 as static temperature, static pressure and Mach number
        TF.set_flight(218.808,
                      23842.272,
                      0.78)

        # Set the losses for all components
        TF.ex_loss = {"inlet": 0., "LPC": 0.132764781, "HPC": 0.100735895, "Burner": 0.010989737,
                      "HPT": 0.078125215, "LPT": 0.104386722, "Fan": 0.074168491, "SE": 0.0, "PE": 0.}

        # Design for a given thrust (Newton), BPR, FPR, LPC PR, HPC PR, T41 (Kelvin)
        s, c, p = TF.design(21000.,
                            self.engine_bpr,
                            self.engine_fpr,
                            self.engine_lpc_pr,
                            self.engine_hpc_pr,
                            Ttmax = self.engine_t41,
                            HPX = self.reference_offtake,
                            wBleed = self.reference_wBleed)



        self.width = 0.5*self.engine_bpr**0.7 + 5.E-6*self.reference_thrust
        self.length = 0.86*self.width + self.engine_bpr**0.37      # statistical regression

        knac = np.pi * self.width * self.length
        self.gross_wet_area = knac*(1.48 - 0.0076*knac)*2.       # statistical regression, two engines
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.15

        self.frame_origin = self.__locate_nacelle__()

    def eval_mass(self):
        self.mass = (1250. + 0.021*self.reference_thrust)*2.       # statistical regression, two engines
        self.cg = self.frame_origin + 0.7 * np.array([self.length, 0., 0.])      # statistical regression

   def unitary_thrust(self,pamb,tamb,mach,rating,throttle=1.,pw_offtake=0.,nei=0.):
        """Unitary thrust of a pure turbofan engine (semi-empirical model)
        """


        return total_thrust, fuel_flow






