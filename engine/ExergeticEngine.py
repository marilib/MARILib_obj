# -*- coding: utf-8 -*-
"""
The ExergeticEngine module contains the base classes to simulate aircraft engines.

It currently provide the following architectures:
    * Turbojet: single shaft, single flow
    * Turbofan: two shafts, two flows
    * Turboprop: Three shafts, single flow
    * Electric Fan: single shaft, single flow, electrically driven fan

Other architectures may be added as necessary as follows:
    * create a new class based on :class:`ExergeticEngine`
    * overload the following methods:
        * cycle
        * design_guess
        * design_equations
        * off_design_equations
        * draw_sankey

The class ExergeticEngine contains the basic operations (compression, espansion, burning, ...), from which a full
thermodynamic cycle may be built. Every operation is expressed in term of power exchange (enthalpy) and exergy loss.

Exergy is defined as the maximum useful work you can get out of a system in a given environment. In thermodynamics, the
total energy is the total enthalpy. It can be made into kinetic power, fully usable, and thermal power, only recoverable
via a carnot ideal cycle.
Hence a general definition for the specific exergy is: :math:`\\Delta \\varepsilon = \\Delta H_t - T_{s0} \\Delta s`,
in J.kg-1.

With the assumption of perfect gas and fixed heat capacity ratio,
:math:`\\Delta \\varepsilon = \\Delta H_t - H_{s0} [ \\ln{\\tau} - \\frac{\\gamma - 1}{\\gamma}\\ln{\\pi}]`

where :math:`\\tau` is the total temperature ratio, and :math:`\\pi` the total pressure ratio.

The model is implemented with the following assumptions:
    * Perfect gas
    * Constant heat capacity ratio
    * The International Unit system is used everywhere.

.. warning:: This has been written for educational purpose only.
   There is no guarantee of any kind.

"""

import copy
import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from matplotlib.sankey import Sankey


# global constants
gamma = 1.4  # heat capacity ratio, assumed constant
R = 287.0486821  # Gas constant
Cp = gamma/(gamma-1.) * R
flhv = 42.798e6  # fuel lower heating value, 18400. btu.ln-1, in J.kg-1, for Jet-A1 (kerosene)


class ExergeticEngine(object):
    """
    The ExergeticEngine provides the base methods to simulate an engine cycle.
    It also implements a single flow, single shaft turbojet.
    """
    def __init__(self):
        """
        Constructor for the ExergeticEngine class.

        The Exergetic engine is initialized with the following data:
            * Cycle date: component losses and off-design slopes
            * Design point state
            * Ambient conditions
        """
        # dictionary with the various component efficiencies
        self.ex_loss = dict()
        # dictionary with the parameters used for off-design computations
        self.slope = {"HPC": 0.543, "HPT": -27.4}
        # amount of flow diverted (ratio of core flow) to cool the HPT inlet vane
        self.cooling_flow = 0.

        # to store the design state
        self.design_state = None
        self.dp = None

        # Ambient conditions (sea level, ISA+15degK, Mn 0.25)
        self.Ts0 = 288.15
        self.Ps0 = 101325.
        self.Mach = 0.25

        # derived parameters
        self.Tt0 = (1 + self.Mach*self.Mach*(gamma - 1.)/2.) * self.Ts0
        self.Pt0 = (1 + self.Mach*self.Mach*(gamma - 1.)/2.)**(gamma/(gamma-1.)) * self.Ps0
        self.V0 = np.sqrt(gamma * R * self.Ts0) * self.Mach
        self.Hs0 = Cp * self.Ts0
        self.Ht0 = Cp * self.Tt0
        self.Ex0 = self.Ht0 - self.Hs0

    def get_Pt(self, Ht, Ex):
        """
        Compute the total pressure, total temperature, and the specific referred mass flow based on
        specific total Enthalpy H and specific Exergy Ex.

        :math:`T_t = \\frac{H_t}{C_p}`

        :math:`P_t = P_{t0} (\\frac{H_t}{H_{t0}} \\exp(\\frac{\\varepsilon - \\varepsilon_0 - H_t + H_{t0}}{H_{s0}}))^\\frac{\\gamma}{\\gamma-1}`

        :param Ht: Specific total enthalpy, J/kg
        :param Ex: Specific exergy, J/kg
        :type H: float
        :type Ex: float
        :return: Total pressure (Pa), total temperature (K), specific referred mass flow (no dim)
        :rtype: (float, float, float)
        """
        Tt = Ht / Cp
        Pt = self.Pt0 * ((Ht / self.Ht0) * np.exp((Ex - self.Ex0 - Ht + self.Ht0) / self.Hs0))**(gamma / (gamma - 1.))
        return Pt, Tt, np.sqrt(Tt / 288.15) / (Pt / 101325.)

    def get_speed(self, Ht, Ex):
        """
        Compute the isentropic speed obtained by expanding the flow to ambient static pressure self.Ps0,
        the necessary specific effective area, as well as the total pressure and temperature.

        :param Ht: enthalpy (J.kg-1)
        :param Ex: exergy (J.kg-1)
        :type H: float
        :type Ex: float
        :return: V, the speed (e.g. specific thrust) in m/s, and A, the specific flow area in m2.s.kg-1,
            total pressure (Pa) and temperature (K)
        :rtype: (float, float, float, float))
        """
        # get the total conditions
        Pt, Tt, Wr = self.get_Pt(Ht, Ex)
        # Get the static temperature by expansion to Ps0
        Ts = Tt * (self.Ps0 / Pt)**((gamma - 1.) / gamma)
        # If the total pressure is higher that the static pressure, the total temperature is also higher than the
        # static temperature. The test is done on the temperatures as this is what is involved in the computations.
        if Tt > Ts:
            # Mach number
            Mach = np.sqrt(2. / (gamma - 1.) * (Tt / Ts - 1.))
            # Speed resulting from isentropic expension to Ps0
            Vis = np.sqrt(gamma * R * Ts) * Mach
            # V is the effective speed in the exhaust section throat.
            # At first, we assume no chock, so V = Vis.
            V = Vis
            Ps = self.Ps0
            # If the isentropic Mach is higher than 1., then the exhaust is chocked
            # we compute the new static conditions just after the chock
            if Mach > 1.:
                Mach = 1.
                Ts = Tt / (1. + (gamma-1.)/2.)
                Ps = Pt / (1. + (gamma-1.)/2.)**(gamma/(gamma-1.))
                V = np.sqrt(gamma * R * Ts)
            # Air density
            rho = Ps / (R * Ts)
            # Specific throat area, i.e area for 1 kg.s-1
            A = 1. / (rho * V)
            return Vis, A, Pt, Tt
        else:
            return 0., np.inf, Pt, Tt

    def from_PR_to_tau_pol(self, PR, eta_pol):
        """
        Auxiliary function to help convert from usual efficiencies to exergy destruction.

        This function is not used in the cycle simulation, but is there to help the user to help converting from the
        usual performance metrics to the one used in this cycle modeling.

        The returned values are :math:`\\tau`, the total temperature ratio, and Ex_loss, the exergy destruction
        expressed as a fraction of the static enthalpy times log(tau).

        The returned tau value can be used to initialize the design point computation.

        The returned Ex_loss value is positive, and may be used to set the component loss.

        :param PR: pressure ratio
        :param eta_pol: polytropic efficiency
        :type PR: float
        :type eta_pol: float
        :return: tau, Ex_loss
        :rtype: (float, float)
        """
        # total temperature ratio for isentropic compression or expansion
        tau_is = PR**((gamma-1.)/gamma)
        if PR > 1.:
            # Compression
            tau = tau_is**(1. / eta_pol)
            Ex_loss = 1. - eta_pol
        else:
            # Expansion
            tau = tau_is**eta_pol
            Ex_loss = 1. / eta_pol - 1.
        # Ex_loss = 1. - (gamma - 1.) / gamma * np.log(PR) / np.log(tau)

        return tau, Ex_loss

    def from_PR_to_tau_is(self, PR, eta_is):
        """
        Auxiliary function to help convert from usual efficiencies to exergy destruction.

        This function is not used in the cycle simulation, but is there to help the user to help converting from the
        usual performance metrics to the one used in this cycle modeling.

        The returned values are :math:`\\tau`, the total temperature ratio, and Ex_loss, the exergy destruction
        expressed as a fraction of the static enthalpy times log(tau).

        The returned tau value can be used to initialize the design point computation.

        The returned Ex_loss value is positive, and may be used to set the component loss.

        :param PR: pressure ratio
        :param eta_is: isentropic efficiency
        :type PR: float
        :type eta_is: float
        :return: tau, Ex_loss
        :rtype: (float, float)
        """
        tau_is = PR**((gamma-1.)/gamma)
        if PR > 1.:
            # Compression
            tau = (tau_is - 1.) / eta_is + 1.
            Ex_loss = 1. - (gamma - 1.) / gamma * np.log(PR) / np.log(tau)
        else:
            # Expansion
            tau = eta_is * (tau_is - 1.) + 1.
            Ex_loss = (gamma - 1.) / gamma * np.log(PR) / np.log(tau) - 1.

        return tau, Ex_loss

    def from_PR_loss_to_Ex_loss(self, PR):
        """
        Auxiliary function to help convert from usual efficiencies to exergy destruction.

        This function is not used in the cycle simulation, but is there to help the user to help converting from the
        usual performance metrics to the one used in this cycle modeling.

        The returned values is Ex_loss, the exergy destruction expressed
        as a fraction of the static enthalpy for the given total pressure loss PR.

        :param PR: pressure loss
        :type PR: float
        :return: Exergy loss
        :rtype: float
        """
        return -(gamma - 1.) / gamma * np.log(PR)

    def set_flight(self, Ts0, Ps0, Mach):
        """
        Set the flight conditions in the object.

        The input are the static temperature and pressure, and the flight Mach number.

        :param Ts0: ambient static temperature (Kelvin)
        :param Ps0: ambient static pressure (Pascal)
        :param Mach: Mach number
        :type Ts0: float
        :type Ps0: float
        :type Mach: float
        :return: None
        """
        self.Ts0 = Ts0
        self.Ps0 = Ps0
        self.Mach = Mach
        self.Tt0 = (1 + self.Mach * self.Mach * (gamma - 1.) / 2.) * self.Ts0
        self.Pt0 = (1 + self.Mach * self.Mach * (gamma - 1.) / 2.)**(gamma/(gamma-1.)) * self.Ps0
        self.V0 = np.sqrt(gamma * R * self.Ts0) * self.Mach
        self.Hs0 = Cp * self.Ts0
        self.Ht0 = Cp * self.Tt0
        self.Ex0 = self.Ht0 - self.Hs0

    def free_stream(self, w):
        """
        Defines the source component, i.e. the one providing the inlet conditions.

        This is the first method to be called in a cycle simulation.

        It defines the engine ram drag and initial kinetic power, as well as all the necessary parameters for the
        following components.

        :param w: freestream mass flow, in kg.s-1
        :type w: float
        :return: the freestream state
        :rtype: dict
        """
        Ht = self.Ht0
        Ex = self.Ex0
        V0, A, Pt, Tt = self.get_speed(Ht, Ex)
        Wr = np.sqrt(Tt / 288.15) / (Pt / 101325.)
        station = {"Ht": Ht, "Ex": Ex, "Pt": Pt, "Tt": Tt, "w": w, "F": - w * V0, "A": A * w, "V0": V0, "Hs": self.Hs0,
                   'Wr': w * Wr, 'Nk': 1. / np.sqrt(Tt), "Pu": - 0.5 * w * V0**2,
                   'PEx': - w * Ex, "losses": {}}
        return station

    def wake_ingestion(self, d_bli, input_flow, name):
        """
        In case you want to simulate a wake ingestion, this is the component to be used, just after free_stream.

        d_bli is the irreversible drag of the body generating the wake (friction, viscous pressure, compressibility),
        in Newtons, for the flight conditions defined for the simulation.

        The method checks if the mass flow is sufficient to ingest the full boundary layer generated by the front body.
        If it is not, the actually ingested drag and kinetic power is estimated.

        :param d_bli: drag available for ingestion, in case of 360° Boundary Layer Ingestion, in Newtons
        :param input_flow: the free stream flow description
        :param name: a name for this component
        :type d_bli: float
        :type input_flow: dict
        :type name: basestring
        :return: the station at the inlet entry
        :rtype: dict
        """

        Ht = input_flow['Ht']
        Exi = input_flow['Ex']
        w = input_flow['w']
        if d_bli > 0:
            wd = 9. * d_bli / self.V0  # assuming speed profile to power 1/7 for the boundary layer
            # If the engine mass flow w is higher than the boundary layer mass flow,
            # the engine is ingesting the full wake
            if w > wd:
                f = wd * self.V0 / 9.
                pu = 0.5 * wd * self.V0**2 * (2. / 10.)
            else:
                # otherwise, the engine ingest only a part of the wake, and a part of the associated drag and kinetic
                # power.
                f = wd * self.V0 * (1 - 8. / 9. * ((w / wd)**(1. / 8.)))
                pu = 0.5 * wd * self.V0**2 * (1 - 8. / 10. * ((w / wd)**(2. / 8.)))
            # The exergy loss os approximated to the kinetic power loss.
            Ex = Exi - pu / w
        else:
            f = 0.
            pu = 0.
            Ex = Exi
        Pt, Tt, Wr = self.get_Pt(Ht, Ex)
        # The parameter stroed here will act as a correction to the total performance to include wake ingestiion.
        # There is a correction on the ram drag, on the kinetic power, and on the exergy (and consequently Pt).
        output_flow = {'Ht': Ht, 'Ex': Ex, 'Pt': Pt, 'Tt': Tt, 'Wr': w * Wr, 'w': w, 'Nk': 1. / np.sqrt(Tt),
                       "F": f, "Pu": pu, "gain": {name: w * (Exi - Ex)},
                       "losses": {}}
        return output_flow

    def loose_pressure(self, loss_coefficient, input_flow, name):
        """
        Defines a component creating a total pressure loss (inlet, ducts).

        The loss coefficient is linked to the pressure loss by the relation
        :math:`\\text{loss} = -\\frac{\\gamma - 1}{\\gamma}\\ln(\\pi)`

        :param loss_coefficient: the exergy loss coefficient
        :param input_flow: the state of the input flow
        :param name: The name of the component, such as 'Inlet'
        :type loss_coefficient: float
        :type input_flow: dict
        :type name: str
        :return: the output station flow state description
        :rtype: dict
        """
        Ht = input_flow['Ht']
        # Loosing total pressure is the same as destroying exergy
        Ex = input_flow['Ex'] - loss_coefficient * self.Hs0
        w = input_flow['w']
        Pt, Tt, Wr = self.get_Pt(Ht, Ex)
        output_flow = {'Ht': Ht, 'Ex': Ex, 'Pt': Pt, 'Tt': Tt, 'Wr': w * Wr, 'w': w, 'Nk': 1. / np.sqrt(Tt),
                       "losses": {name: w * loss_coefficient * self.Hs0}}
        return output_flow

    def compress(self, loss_coefficient, input_flow, tau, name):
        """
        Defines a compressor component.

        The specific power required for the compression is :math:`H_{t2}-H_{t1} = (\\tau-1)H_{t1}`

        The variation of exergy is :math:`\\Delta \\varepsilon = H_{t2}-H_{t1}-\\text{loss}.H_{s0}.ln(\\tau)`

        :param loss_coefficient: the compressor efficiency
        :param input_flow: the state of the incoming flow
        :param tau: the total temperature ratio (defines the amount of compression)
        :param name: The name of the component, such as 'HPC'
        :type loss_coefficient: float
        :type input_flow: dict
        :type tau: float
        :type name: str
        :return: the output station flow state description
        :rtype: dict
        """
        # dH is the specific work input
        dH = input_flow['Ht'] * (tau - 1.)
        Ht = input_flow['Ht'] + dH
        # Exergy is augmented by the work input, minus the losses
        Ex = input_flow['Ex'] + dH - np.log(tau) * loss_coefficient * self.Hs0
        Pt, Tt, Wr = self.get_Pt(Ht, Ex)
        w = input_flow['w']
        output_flow = {'Ht': Ht, 'Ex': Ex, 'Pt': Pt, 'Tt': Tt, 'Wr': w * Wr, 'w': w,
                       'PR': Pt/input_flow['Pt'], 'Nk': 1. / np.sqrt(Tt),
                       "losses": {name: w * np.log(tau) * loss_coefficient * self.Hs0}}
        return output_flow

    def expand(self, loss_coefficient, input_flow, dH, name):
        """
        Defines a turbine component

        The specific power required for the compression is :math:`H_{t2}-H_{t1} = \\frac{\\Delta H}{w}`

        note: :math:`\\Delta H` is negative. :math:`\\tau < 1`

        The variation of exergy is :math:`\\Delta \\varepsilon = H_{t2}-H_{t1}+\\text{loss}.H_{s0}.\\ln(\\tau)`

        :param loss_coefficient: the turbine efficiency
        :param input_flow: the state of the incoming flow
        :param dH: the absolute work to be extracted from the flow, in Watts
        :param name: The name of the component, such as 'HPT'
        :type loss_coefficient: float
        :type input_flow: dict
        :type dH: float
        :type name: str
        :return: the output station flow state description
        :rtype: dict

        """
        w = input_flow['w']
        Ht = input_flow['Ht'] + dH / w
        tau = Ht / input_flow['Ht']
        Ex = input_flow['Ex'] + dH / w + np.log(tau) * loss_coefficient * self.Hs0
        Pt, Tt, Wr = self.get_Pt(Ht, Ex)
        output_flow = {'Ht': Ht, 'Ex': Ex, 'Pt': Pt, 'Tt': Tt, 'Wr': w * Wr, 'w': input_flow['w'],
                       'PR': Pt/input_flow['Pt'], 'Nk': 1. / np.sqrt(Tt),
                       "losses": {name: - w * np.log(tau) * loss_coefficient * self.Hs0}}
        return output_flow

    def exhaust(self, loss_coefficient, input_flow, name):
        """
        Defines a exhaust component (thrust generator).

        The engine exhausts create the thrust. Any power provided on top of the work of the thrust is then lost.

        The losses breakdown then contains:
            * The exergy loss associated to the total pressure loss in the exhaust: :math:`\\text{loss}.H_{s0}`
            * The jet loss, which is the kinetic power of the jet minus the work of the thrust: :math:`\\frac{1}{2}( F - V_0 )^2`
            * The thermal loss, which is the Exergy minus the kinetic power: :math:`\\varepsilon - \\varepsilon_0 - \\frac{1}{2}( F^2 - {V_0}^2 )`

        :param loss_coefficient: loss in the exhaust (total pressure loss)
        :param input_flow: the state of the incoming flow
        :param name: The name of the component, such as 'HPT'
        :type loss_coefficient: float
        :type input_flow: dict
        :type name: str
        :return: the output station flow state description
        :rtype: dict
        """
        Ht = input_flow['Ht']
        Ex = input_flow['Ex'] - loss_coefficient * self.Hs0
        w = input_flow['w']
        F, A, Pt, Tt = self.get_speed(Ht, Ex)
        Wr = np.sqrt(Tt / 288.15) / (Pt / 101325.)
        output_flow = {"Ht": Ht, "Ex": Ex, "Pt": Pt, "Tt": Tt, 'Wr': w * Wr, 'w': w, "F": w * F, "A": w * A,
                       "P": w * (F - self.V0) * self.V0, "Pu": 0.5 * w * F**2, 'PEx': w * Ex,
                       "losses": {name: w * loss_coefficient * self.Hs0,
                                  "Jet loss": 0.5 * w * (F - self.V0)**2,
                                  "Thermal": w * (Ex - self.Ex0) - 0.5 * w * (F**2 - self.V0**2)}}
        return output_flow

    def burner(self, loss_coefficient, input_flow, name, dH=None, Ttm=None):
        """
        Defines a burner component.

        The driving parameter may be the fuel power Dh or the exit temperature Ttm, but not both.
        If both are given, dH is used.

        The burner has two loss mechanisms:
            * One associated to the pressure loss across the component :math:`\\text{loss}.H_{s0}`
            * One associated to the temperature rise :math:`H_{s0}.\\ln(\\tau)`

        :param loss_coefficient: loss in the exhaust (total pressure loss)
        :param input_flow: the state of the incoming flow
        :param name: The name of the component, such as 'Combustor'
        :param dH: the fuel power, in Watts
        :param Ttm: the exit total temperature, before HP turbine stator cooling introduction
        :return: the output station flow state description
        :rtype: dict
        """
        w = input_flow['w']
        if dH is None:
            dHF = Cp * max(Ttm - input_flow['Tt'], 0.) * (1. - self.cooling_flow)
        else:
            dHF = dH / w
        Ht = input_flow['Ht'] + dHF
        tau = Ht / input_flow['Ht']
        Ex = input_flow['Ex'] + dHF - (np.log(tau) + loss_coefficient) * self.Hs0
        Pt, Tt, Wr = self.get_Pt(Ht, Ex)
        output_flow = {'Ht': Ht, 'Ex': Ex, 'Pt': Pt, 'Tt': Tt, 'Wr': w * Wr, 'w': input_flow['w'],
                       'Nk': 1. / np.sqrt(Tt), 'Pth': w * dHF,
                       "losses": {name: w * loss_coefficient * self.Hs0,
                                  name + " Thermal": w * np.log(tau) * self.Hs0}}
        return output_flow

    def global_performances(self, stations):
        """
        Sum up the global engine performance: thrust, efficiency, exergy destruction, etc...

        The resulting dictionary contains the following items:
            * Fnet: the total net thrust, in Newtons
            * Pprop: the work of the net thrust, in Watts
            * Pu: the kinetic power provided by the engine, in Watts
            * Pth: the power provided to the engine (usually the fuel power), in Watts
            * Anergy: the sum of all the losses (except the exhaust jet and thermal losses)
            * Jet loss: the sum of all jet losses
            * Exergy: The exergy produced by the engine
            * etathx: the ratio of produced exergy to the provided power
            * etaprx: the ratio of the work of thrust to the produced exergy
            * etapr: the ratio of the work of thrust to the produced kinetic power, a.k.a. propulsive efficiency
            * etath: the ratio of the produced kinetic power to the provided power, a.k.a. thermal efficiency
            * wfe: Pth expressed in kg/s of fuel

        :param stations: a dictionary with the state of the flow in all engine stations
        :type stations: dict
        :return: a dictionary with the global performances
        :rtype: dict
        """
        # global results:
        perf = dict()
        # net thrust is the sum of ram drag and gross thrusts
        perf['Fnet'] = sum([stations[k]["F"] for k in stations.keys() if 'F' in stations[k].keys()])
        # Work of net thrust
        perf['Pprop'] = perf['Fnet'] * self.V0
        # Usefull (a.k.a kinetic) power provided by the engine
        perf['Pu'] = sum([stations[k]["Pu"] for k in stations.keys() if 'Pu' in stations[k].keys()])
        # Power provided to the engine (usually thermal)
        perf['Pth'] = sum([stations[k]["Pth"] for k in stations.keys() if 'Pth' in stations[k].keys()])
        perf['wfe'] = perf['Pth'] / flhv
        # Sum of all losses
        perf['Anergy'] = sum([sum([stations[k]['losses'][l] for l in stations[k]['losses'].keys()
                                       if l not in ['Jet loss', 'Thermal']]) for k in stations.keys()])
        # kinetic power losses in the exhaust jets
        perf['Jet loss'] = sum([stations[k]['losses']['Jet loss'] for k in stations.keys()
                                    if 'Jet loss' in stations[k]['losses'].keys()])
        perf['Exergy'] = sum([stations[k]['PEx'] for k in stations.keys() if 'PEx' in stations[k].keys()])
        # sort-of thermal efficiency, based on exergy
        perf['etathx'] =  perf['Exergy'] / perf['Pth']
        # sort-of propulsive efficiency, based on exergy
        perf['etaprx'] = perf['Pprop'] / perf['Exergy']
        # Classical propulsive efficiency
        perf['etapr'] = perf['Pprop'] / perf['Pu']
        # Thermal efficiency
        perf['etath'] = perf['Pu'] / perf['Pth']
        return perf

    def draw_pie(self, s, ax):
        """
        Draw a pie chart of all the engine losses.
        It looks like this:

        .. image:: pie_turbojet.png

        It looks for the 'losses' key in the various stations.
        Losses named 'Jet loss' or 'Thermal' ar considered as propulsive losses and are not shown.

        :param s: the stations
        :param ax: the axes system
        :type s: dict
        :type ax: plt.axes.Axes
        :return:
        """
        sizes = []
        labels = []
        # collect the size and name of each part of the pie
        for k in sorted(s.keys()):
            name = [v for v in s[k]['losses'].keys() if v not in ['Jet loss', 'Thermal']]
            for v in s[k]['losses'].keys():
                sizes.append(s[k]['losses'][v])
                if v in ['Jet loss', 'Thermal']:
                    labels.append(name[0] + " " + v)
                else:
                    labels.append(v)
        total = sum(sizes)
        # Draw the pie chast
        ax.set_aspect('equal')
        wedges, texts = ax.pie(sizes, labels=None, autopct=None, startangle=90, counterclock=False)

        # define a dictionary of properties for the callout boxes that will contain the labels
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center")

        # for all part of the pie, add a box with the label and value, and an arrow between the box and the part
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            r = [1.2, 1.6][i%2]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate("{:s}\n{:.0f}kW\n{:.2%}".format(labels[i], sizes[i]/1000., sizes[i]/total), xy=(x, y),
                        xytext=(r*x, r*y), horizontalalignment=horizontalalignment, **kw)
        ax.set_frame_on(True)
        ax.set_title("Losses repartition")
        ax.set_xbound(-2., 2.)
        ax.set_ybound(-2., 2.)

    def cycle(self, *args, **kwargs):
        """
        Compute the performance of the engine, based on given cycle parameters

        This implementation is a simple 1-shaft turbojet.
        This method shall be overloaded for other engine architectures.

        Only two positional argument are needed:
            * the engine mass flow w, in kg/s
            * the compressor temperature ratio tau.

        Only one keyed argument is needed: dHf or Ttmax.
            * dHf is the fuel power in Watts,
            * Ttmax is the combustion chamber exit temperature in Kelvin.

        Other possible keyed arguments:
            * HPX: some customer power extraction on the shaft (if positive), in Watts.
                A negative value means you provide additional power, via some electric motor for instance.

        The computed dictionaries are:
            * the stations states: the state of the air flow at each component interfaces
            * the components state:

                * the keys are the component names
                * The content depends on the component. It includes pressure ratio, map coordinates, ...

            * the performances, see :meth:`global_performances`

        :param args: cycle parameters
        :param kwargs: optional cycle parameters
        :type args: tuple
        :type kwargs: dict
        :return: 3 dictionaries with the state of the machine
        :rtype: (dict, dict, dict)
        """
        dHf = kwargs.get('dHf')
        Ttmax = kwargs.get('Ttmax')
        w, tau = args
        # Station 0: free stream
        components = {"HPC": {"tau": tau}}
        stations = {"0": self.free_stream(w)}

        # station 2: Inlet exit, HPC entry
        stations["2"] = self.loose_pressure(self.ex_loss["inlet"], stations['0'], "Inlet")
        components["HPC"]["WR"] = stations["2"]['Wr']
        components["HPC"]["Nk"] = stations["2"]['Nk']

        # station 3: HPC exit, burner entry
        stations["3"] = self.compress(self.ex_loss["HPC"], stations["2"], tau, "HPC")
        dH_HPC = (stations["3"]['Ht'] - stations["2"]['Ht']) * stations['2']['w']
        components["HPC"]["PR"] = stations["3"]['PR']

        # station 4: Burner exit, HPT entry
        stations['4'] = self.burner(self.ex_loss["Burner"], stations['3'], "Burner", dH=dHf, Ttm=Ttmax)
        components["Burner"] = {"dHF": stations['4']['Pth']}

        # station 5: HPT exit, exhaust entry
        components["HPT"] = {'Nk': stations['4']['Nk'], 'WR': stations['4']['Wr']}
        dH_HPT = -dH_HPC - kwargs.get("HPX", 0.)
        stations["5"] = self.expand(self.ex_loss['HPT'], stations['4'], dH_HPT, "HPT")
        components['HPShaft'] = {'HPX': kwargs.get("HPX", 0.)}

        # station 9: hot flow exhaust to ambient
        stations['9'] = self.exhaust(self.ex_loss["PE"], stations['5'], "PE")

        perfs = self.global_performances(stations)
        return stations, components, perfs

    def design_guess(self, *args):
        """
        This method shall be overloaded for other engine architectures.

        This present implementation is for a simple 1-shaft turbojet.

        Two values are expected in args:
            * the required thrust
            * the required compressor pressure ratio

        The cycle parameters returned are the mass flow, estimated assuming a specific thrust of 100m/s, and the
        compression temperature, estimated using the method :meth:`from_PR_to_tau_pol`

        :param args: the performance values sought
        :type param: tuple
        :return: a guess for the cycle parameters
        :rtype: np.array
        """
        tau_h, hloss = self.from_PR_to_tau_pol(args[1], 1. - self.ex_loss['HPC'])
        return np.array([args[0] / 100., tau_h])

    def design_equations(self, x, *args):
        """
        This method shall be overloaded for other engine architectures.

        This function creates the link between the performance parameters sought (in args[0]) and the cycle results.
        The cycle parameters are in x

        This present implementation is for a simple 1-shaft turbojet. The values soughts are the total net thrust and
        the compressor pressure ratio.

        :param x: the cycle parameters
        :param args: a tuple with two items: the performance value sought and the keyed arguments for the cycle
        :type x: tuple
        :type args: tuple
        :return: a vector to be nullified
        :rtype: np.array
        """
        v = args[0]
        other_kwargs = args[1]
        s, c, p = self.cycle(*x, **other_kwargs)
        y = [p['Fnet'] / v[0] - 1., c['HPC']['PR'] / v[1] - 1.]
        return np.array(y)

    def design(self, *args, **kwargs):
        """
        Seek design cycle parameters and size the engine for a given thrust.

        The content of args and kwargs depends on your implementation of :meth:`design_equations`.
        in practice, args contains the target values to be obtained, such as the thrust and the pressure ratios, and
        kwargs contains Ttmax, the burner exit temperature, and other cycle parameters.

        You may as well provide some initial value for the cycle parameter via the keyed argument 'guess'.
        If not provided, the method :meth:`design_guess` will be used.

        The returned dictionaries are the same as the ones returned by :meth:`cycle`.

        :param args: performance parameters, dependant on the current implementation
        :param kwargs: various keyed arguments
        :type args: tuple
        :type kwargs: dict
        :return: the machine state in three dictionaries: the flow state s, the component state c, and the performances
            p.
        :rtype: (dict, dict, dict)
        """
        guess = kwargs.get('guess')
        # other keyed argument will be forwarded to the cycle method
        other_kwargs = {k: kwargs[k] for k in kwargs.keys() if k not in ['guess']}
        if guess is None:
            print(*args)
            guess = self.design_guess(*args)

        # find out the cycle parameters that provide the performance required
        output_dict = fsolve(self.design_equations, x0=guess, args=(args, other_kwargs), factor=0.1, full_output=True)

        res = output_dict[0]

        # get back the solution found
        # recompute the obtained engine
        s, c, p = self.cycle(*res, **other_kwargs)
        # keep a copy of the design state for off-design computations
        self.design_state = copy.deepcopy(s)
        self.comp_state = copy.deepcopy(c)
        # keep the design cycle solution as a seed for off-design
        self.dp = copy.deepcopy(res)
        return s, c, p

    def off_design_equations(self, x, *args):
        """
        This method shall be overloaded for other engine architectures.

        This present implementation is for a simple 1-shaft turbojet.

        This function implements the various equations to be satisfied in off-design conditions.

        for the turbojet, these are:
            * equalization of the exhaust area with the one required by the cycle
            * equalization of the compressor physical speed with the turbine physiscal speed

        The shaft speed is normally obtained via a component map. Here it has been simplified as a straight line.

        :param x: vector of values around 1. when multiplied by the design point cycle parameter, gives the tested cycle
            parameters
        :type x: np.array
        :param args: a tuple with one dictionary: the additional cycle parameters
        :return: a vector of same size as x, to be nullified
        :rtype: np.array
        """
        xx = x * self.dp
        r, c, p = self.cycle(*xx, **args[0])
        # Get the shaft speed compatible with the corrected flow, as a fraction of the design speed
        shaft_speed = {i: ((self.slope[i] * (c[i]['WR'] / self.comp_state[i]['WR'] - 1.) + 1.) *
                           self.comp_state[i]['Nk'] / c[i]['Nk']) for i in self.slope.keys()}
        # vector to be nullified:
        # [0]: hot exhaust area shall match the design one
        # [1]: HPC and HPT are linked together: same physical shaft speed ratio than design point
        y = [r['9']['A'] / self.design_state['9']['A'] - 1., shaft_speed['HPT'] / shaft_speed['HPC'] - 1.]
        # print(x, y)
        return y

    def magic_guess(self):
        """
        Provide a starting point for off-design computation which revealed to be very robust to flying conditions,
        but nothing is guaranty
        """
        guess = [0.8, 0.9]
        return guess

    def off_design(self, **kwargs):
        """
        Perform off-design computation: once the engine is sized, find the performance for other flight condition
        or throttle positions.

        All keyed arguments are optional. However, at least one of Ttmax or wfe shall be given.

        The keyed arguments are:
            * Ttmax: burner exit temperature, in Kelvin
            * wfe: fuel flow, in kg/s
            * guess: initial vector for the solver. These are factor to the cycle parameters.
                A vector of ones is the design point.
            * plus any other keyed parameters that will be forwarded to :meth:`cycle`

        :param kwargs: keyed arguments are guess, and wfe or Ttmax. guess is used as an initial guess for the solver.
            wfe is the fuel flow, Ttmax the burner exit temperature. either of them defines the throttle.
        :type kwargs: dict
        :return: the machine state in three dictionaries: the flow state s, the component state c, and the performances p
        :rtype: (dict, dict, dict)
        """
        guess = kwargs.get('guess')
        wfe = kwargs.get('wfe')
        # other keyed argument will be forwarded to the cycle method
        other_kwargs = {k: kwargs[k] for k in kwargs.keys() if k not in ['guess', 'wfe']}
        if wfe is not None:
            other_kwargs['dHf'] = wfe * flhv

        # use a list made of as many ones as there are cycle parameters to be guessed.
        if guess is None:
            x0 = np.ones_like(self.dp)
        else:
            x0 = guess

        # Solve
        output_dict = fsolve(self.off_design_equations, x0=x0, args=other_kwargs, factor=0.1, full_output=True)

        res = output_dict[0]
        ier = output_dict[2]
        msg = output_dict[3]

        # In case of successful solving, report the result
        if ier==1:
            xx = res * self.dp
            s, c, p = self.cycle(*xx, **other_kwargs)
            p['success'] = True
            p['sol'] = copy.deepcopy(res)
            return s, c, p
        else:
            print(msg)
            p = {'success': False}
            return None, None, p

    def draw_sankey(self, s, c, p, ax):
        """
        Draw the sankey diagram of the engine
        To be overloaded if the engine architecture changes.

        For the turbojet, it looks like this:

        .. image:: sankey_turbojet.png


        :param s: stations dictionary
        :param c: component dictionary
        :param p: performance dictionary
        :param ax: the matplotlib axe object where to draw the diagram
        :type s: dict
        :type c: dict
        :type p: dict
        :type ax: plt.axes.Axes
        :return: nothing
        """
        # Sets the scale of the diagram: this is the inverse of the total amount of energy involved, in kW.
        scale = 1000. / (s['0']['Ht'] * s['0']['w'] + p['Pth'])
        sankey = Sankey(ax=ax, unit="kW", scale=scale, format='%.0f')

        # now we draw each component in turn, with the output of the previous one (last)
        # connected to the input of the following one (first)
        # Intermediate arrows are other fluxes: negative if exiting the component, positive if entering the component.
        # Losses are above (orientation = 1), shaft power below (orientation = -1)
        # Inlet 0
        sankey.add(flows=[s['0']['Ex'] * s['0']['w'] / 1000.,
                          -s['2']['losses']['Inlet'] / 1000.,
                          -s['2']['Ex'] * s['2']['w'] / 1000.],
                   patchlabel='Inlet', orientations=[0, 1, 0], label='Inlet', fc='#00A000', labels=["", "Loss", None])

        # HPC 1
        dHH = s['3']['Ht'] - s['2']['Ht']
        sankey.add(flows=[s['2']['Ex'] * s['2']['w'] / 1000.,
                          -s['3']['losses']['HPC'] / 1000.,
                          dHH * s['3']['w'] / 1000.,
                          -s['3']['Ex'] * s['3']['w'] / 1000.],
                   patchlabel='HPC', orientations=[0, 1, -1, 0], prior=0, connect=(2, 0), label="HPC", fc='#0000F0',
                   labels=["", "Loss", "", None])

        # Burner 2
        dH = s['4']['Ht'] - s['3']['Ht']
        sankey.add(flows=[s['3']['Ex'] * s['3']['w'] / 1000.,
                          -s['4']['losses']['Burner'] / 1000.,
                          -s['4']['losses']['Burner Thermal'] / 1000.,
                          dH * s['4']['w'] / 1000.,
                          -s['4']['Ex'] * s['4']['w'] / 1000.],
                   patchlabel='Burner', orientations=[0, 1, 1, 1, 0], prior=1, connect=(3, 0), label='Burner',
                   fc='#F0F000', labels=["", "Loss", "Thermal", "Fuel", None])

        # HPT 3
        dHHT = s['5']['Ht'] - s['4']['Ht']
        sankey.add(flows=[s['4']['Ex'] * s['4']['w'] / 1000.,
                          -s['5']['losses']['HPT'] / 1000.,
                          dHHT * s['5']['w'] / 1000.,
                          -s['5']['Ex'] * s['5']['w'] / 1000.],
                   patchlabel='HPT', orientations=[0, 1, -1, 0], prior=2, connect=(4, 0), label='HPT', fc='#0000FF',
                   labels=["", "Loss", "", None])

        # Primary Exhaust 4
        sankey.add(flows=[s['5']['Ex'] * s['5']['w'] / 1000.,
                          -s['9']['losses']['PE'] / 1000.,
                          -s['9']['losses']['Thermal'] / 1000.,
                          -s['9']['losses']['Jet loss'] / 1000.,
                          -s['9']['P'] / 1000.,
                          -s['0']['Ex'] * s['9']['w'] / 1000.],
                   patchlabel='Pri. Exhaust', orientations=[0, 1, 1, 1, 0, 0], prior=3, connect=(3, 0),
                   label='Pri. Exhaust', fc='#FF0000',
                   labels=["", "Loss", "Thermal loss", "Prop. Loss", "Prop. Power", ""])

        # HP shaft 5
        # trunklength may need some adaptation
        sankey.add(flows=[-dHH * s['3']['w'] / 1000.,
                          -dHHT * s['5']['w'] / 1000.,
                          -c['HPShaft']['HPX']/1000.],
                   patchlabel='HP shaft', orientations=[-1, -1, 1], prior=3, connect=(2, 1), trunklength=4.05,
                   label='HP shaft', fc='#F0F0F0', labels=[None, None, "Offtake"])

        diagrams = sankey.finish()
        ax.set_title("Power flows")
        # ax.legend()

    def print_stations(self, s):
        """
        Print a simple table of all the station states.
        The table may be copy/pasted in a spreadsheet as the tab character is used as a separator.

        :param s: the dictionary with the stations flow state
        :type s: dict
        :return:
        """
        print("Station\tMass flow\tTotal Temp.\tTotal Press.\tSpec. Total enthalpy\tSpec. Exergy\tThrust\tArea\t" +
              "Prop. Pwr\tUseful Pwr")
        print("--\tkg.s-1\tK\tPa\tJ/kg\tJ/kg\tN\tm2\tW\tW")
        for st in sorted(s.keys()):
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(st, *(s[st].get(i, "")
                                                                        for i in ["w", "Tt", "Pt", "Ht", "Ex", "F",
                                                                                  "A", "P", "Pu"])))

    def print_csv(self, s):
        txt_list = []
        txt_list.append(["Station","Mass flow","Total Temperature","Total Pressure","Spec. Total enthalpy",
                         "Spec. Exergy","Thrust","Area","Propulsive Power","Useful Pwr"])
        txt_list.append(["","kg/s","K","Pa","J/kg","J/kg","N","m2","W","W"])
        for st in sorted(s.keys()):
            txt_line = [st]+[s[st].get(i, "") for i in ["w","Tt","Pt","Ht","Ex","F","A","P","Pu"]]
            txt_list.append(txt_line)
        np.savetxt("test.csv",txt_list,delimiter=";",fmt='%20s')


    def print_perfos(self, p, indent=""):
        """
        Print the performance dictionary in a pretty and useful manner.

        For the time being, the dictionary is printed as a tree.
        It can be used with any dictionary.
        Future implementation may be more specific.


        Indent is a character string that will be added at the beginning of all lines.

        :param p: the dictionary to print
        :param indent: indentation
        :type p: dict
        :type indent: str
        :return: nothing
        """
        for k in sorted(p.keys()):
            if isinstance(p[k], dict):
                print("{}{}:".format(indent, k))
                id2 = indent + "\t"
                self.print_perfos(p[k], id2)
            else:
                print("{}{}\t{}".format(indent,k, p[k]))

    def print_components(self, c):
        """
        Print the component dictionary in a pretty and useful manner.
        For the time being, the dictionary is printed as a tree.
        It can be used with any dictionary.
        Future implementation may be more specific.

        :param p: the dictionary to print
        :type p: dict
        :return: nothing
        """
        self.print_perfos(c)


class Turbofan(ExergeticEngine):
    """
    Defines a 2-shaft, 2-flows turbofan, based on the ExergeticEngine object.

    This cycle supports customer power off-takes on both LP and HP shafts, and customer bleeds on the HP compressor.
    """
    def __init__(self):
        """
        Constructor for the turbofan object, with values for the relation between corrected flow and corrected speed
        for each component.
        """
        super(Turbofan, self).__init__()
        # the only thing that needs overloading in __init__ is this dictionary:
        # dictionary with the slope of corrected flow vs. corrected speed
        self.slope = {"Fan": 1.201, "LPC": 0.729, "HPC": 0.543, "HPT": -27.4, "LPT": -27.4}

    def cycle(self, *args, **kwargs):
        """
        Compute the performance of the engine, based on given cycle parameters

        This present implementation is for a 2-shaft 2-flow turbofan.

        The positional argument are:
            * the engine core mass flow w, in kg/s
            * the by-pass ratio
            * the fan temperature ratio tau_f.
            * the low pressure compressor temperature ratio tau_l.
            * the high pressure compressor temperature ratio tau_h.

        Only one keyed argument is needed: dHf or Ttmax.
            * dHf is the fuel power in Watts,
            * Ttmax is the combustion chamber exit temperature in Kelvin.

        Other possible keyed arguments:
            * HPX: some customer power extraction on the HP shaft (if positive), in Watts.
                A negative value means you provide additional power, via some electric motor for instance.
            * LPX: some customer power extraction on the LP shaft (if positive), in Watts.
                A negative value means you provide additional power, via some electric motor for instance.
            * BleedWorkFraction: DEfines where the bleed port is located, in terms of amount of work relative the total
                HP compressor work.
                0. means at the HP compressor inlet.
                1. means at the HP compressor exit.
                The default value is 0.5.
            * wBleed: some customer bleed, in kg/s. The default value is 0.

        The computed dictionaries are:
            * the stations states: the state of the air flow at each component interfaces
            * the components state:

                * the keys are the component names
                * The content depends on the component. It includes pressure ratio, map coordinates, ...

            * the performances, see :meth:`ExergeticEngine.global_performances`

        :param args: cycle parameters
        :param kwargs: optional cycle parameters
        :type args: tuple
        :type kwargs: dict
        :return: 3 dictionaries with the state of the machine
        :rtype: (dict, dict, dict)
        """
        w, bpr, tau_f, tau_l, tau_h = args
        dHf = kwargs.get('dHf')
        Ttmax = kwargs.get('Ttmax')

        # Station 0: free stream
        components = {"Fan": {"tau": tau_f}, "LPC": {"tau": tau_l}, "HPC": {"tau": tau_h}}
        stations = {"0": self.free_stream(w * (1. + bpr))}

        # station 1: Fan face
        stations["1"] = self.loose_pressure(self.ex_loss["inlet"], stations['0'], "Inlet")
        components["Fan"]["WR"] = stations["1"]['Wr']
        components["Fan"]["Nk"] = stations["1"]['Nk']

        # station 21: in fan root = LPC face
        stations["21"] = {k: stations['1'][k] for k in ['Ht', 'Ex', 'Pt', 'Tt', 'Nk']}
        stations["21"]['losses'] = {}
        stations["21"]['w'] = w
        stations["21"]['Wr'] = w * np.sqrt(stations["21"]['Tt']/288.15)/(stations["21"]['Pt']/101325.)
        components["LPC"]["WR"] = stations["21"]['Wr']
        components["LPC"]["Nk"] = stations["21"]['Nk']

        # station 13: Fan exit
        stations["13"] = self.compress(self.ex_loss["Fan"], stations["1"], tau_f, "Fan")
        stations["13"]['w'] = w * bpr
        stations["13"]['Wr'] = stations["13"]['Wr'] * bpr / (1. + bpr)
        dH_Fan = (stations["13"]['Ht'] - stations["1"]['Ht']) * stations["13"]['w']
        components["Fan"]["PR"] = stations["13"]['PR']

        # station 19: cold flow throat exhaust
        stations["19"] = self.exhaust(self.ex_loss["SE"], stations['13'], "SE")

        # station 25: LPC exit
        stations["25"] = self.compress(self.ex_loss["LPC"], stations["21"], tau_l, "LPC")
        dH_LPC = (stations["25"]['Ht'] - stations["21"]['Ht']) * stations['21']['w']
        components["LPC"]["PR"] = stations["25"]['PR']

        # station B27: bleed port
        bwf = kwargs.get("BleedWorkFraction", 0.5)
        tau_bld = bwf * (tau_f - 1.) + 1.
        s25b = copy.deepcopy(stations["25"])
        s25b["w"] = kwargs.get("wBleed", 0.)
        stations["B27"] = self.compress(self.ex_loss["HPC"], s25b, tau_bld, "HPC_bleed")
        dH_bld = (stations["B27"]['Ht'] - stations["25"]['Ht']) * stations["B27"]['w']
        # As the bleed compression has already been computed,
        # we remove from the mass flow the bleed part, because we reuse this dictionary for the HPC computation
        s25b["w"] = stations["25"]["w"] - kwargs.get("wBleed", 0.)

        # Station 3: HPC exit.
        components["HPC"]["Nk"] = stations['25']['Nk']
        components["HPC"]["WR"] = stations['25']['Wr']
        stations["3"] = self.compress(self.ex_loss["HPC"], s25b, tau_h, "HPC")
        stations["3"]["losses"]["Bleed"] = dH_bld
        dH_HPC = (stations["3"]['Ht'] - stations["25"]['Ht']) * stations["3"]['w'] + dH_bld
        components["HPC"]["PR"] = stations["3"]['PR']

        # station 4: Burner exit
        stations['4'] = self.burner(self.ex_loss["Burner"], stations['3'], "Burner", dH=dHf, Ttm=Ttmax)
        components["Burner"] = {"dHF": stations['4']['Pth']}

        # station 43: HPT exit
        components["HPT"] = {'Nk': stations['4']['Nk'], 'WR': stations['4']['Wr']}
        dH_HPT = -dH_HPC - kwargs.get("HPX", 0.)
        stations["43"] = self.expand(self.ex_loss['HPT'], stations['4'], dH_HPT, "HPT")
        components['HPShaft'] = {'HPX': kwargs.get("HPX", 0.)}

        # station 5: LPT exit
        components["LPT"] = {'Nk': stations['43']['Nk'], 'WR': stations['43']['Wr']}
        dH_LPT = - dH_LPC - dH_Fan - kwargs.get("LPX", 0.)
        stations["5"] = self.expand(self.ex_loss['LPT'], stations['43'], dH_LPT, "LPT")
        components['LPShaft'] = {'LPX': kwargs.get("LPX", 0.)}

        # station 9: hot flow exhaust
        stations['9'] = self.exhaust(self.ex_loss["PE"], stations['5'], "PE")

        perfs = self.global_performances(stations)
        return stations, components, perfs

    def design_guess(self, *args):
        """
        This present implementation is for a 2-shaft 2-flow turbofan.

        Two values are expected in args:
            * the required thrust
            * the required compressor pressure ratio

        The cycle parameters returned are the mass flow, estimated assuming a specific thrust of 100m/s,
        the by-pass ratio, and the compressions temperatures, estimated using the method
        :meth:`ExergeticEngine.from_PR_to_tau_pol`

        :param args: the performance values sought
        :type param: tuple
        :return: a guess for the cycle parameters
        :rtype: np.array
        """
        thrust, bpr, fpr, lpr, hpr = args
        tau_f, floss = self.from_PR_to_tau_pol(fpr, 1. - self.ex_loss['Fan'])
        tau_l, lloss = self.from_PR_to_tau_pol(lpr, 1. - self.ex_loss['LPC'])
        tau_h, hloss = self.from_PR_to_tau_pol(hpr, 1. - self.ex_loss['HPC'])
        return np.array([thrust / 100., bpr, tau_f, tau_l, tau_h])

    def design_equations(self, x, *args):
        """
        This function creates the link between the performance parameters sought (in args[0]) and the cycle results.
        The cycle parameters are in x.

        This present implementation is for a 2-shaft 2-flow turbofan. The values sought are the total net thrust,
        by-pass ratio and the 3 compressors pressure ratio (fan, low pressure compressor, high pressure compressor).

        :param x: the cycle parameters
        :param args: a tuple with 2 items: the performance value sought and the keyed arguments for the cycle
        :type x: tuple
        :type args: tuple
        :return: a vector to be nullified
        :rtype: np.array
        """
        v = args[0]
        other_kwargs = args[1]
        s, c, p = self.cycle(*x, **other_kwargs)
        y = [p['Fnet'] / v[0] - 1., x[1] / v[1] - 1., c['Fan']['PR'] / v[2] - 1., c['LPC']['PR'] / v[3] - 1.,
             c['HPC']['PR'] / v[4] - 1.]
        return np.array(y)

    def off_design_equations(self, x, *args):
        """
        This present implementation is for a 2-shaft 2-flow turbofan.

        This function implements the various equations to be satisfied in off-design conditions.

        for the turbofan, these are:
            * equalization of the hot exhaust area with the one required by the cycle
            * equalization of the cold exhaust area with the one required by the cycle
            * equalization of the fan physical speed with the low pressure compressor physical speed
            * equalization of the high pressure compressor physical speed with the high pressure turbine physical speed
            * equalization of the low pressure compressor physical speed with the low pressure turbine physical speed

        The shaft speeds are normally obtained via a component map. Here it has been simplified as a straight line.

        :param x: vector of values around 1. when multiplied by the design point cycle parameter, gives the tested cycle
            parameters
        :type x: np.array
        :param args: a tuple with one dictionary: the additional cycle parameters
        :return: a vector of same size as x, to be nullified
        :rtype: np.array
        """
        xx = x * self.dp
        r, c, p = self.cycle(*xx, **args[0])
        # Get the shaft speed compatible with the corrected flow, as a fraction of the design speed
        shaft_speed = {i: ((self.slope[i] * (c[i]['WR'] / self.comp_state[i]['WR'] - 1.) + 1.) *
                           self.comp_state[i]['Nk'] / c[i]['Nk']) for i in self.slope.keys()}
        # vector to be nullified:
        # [0]: hot exhaust area shall match the design one
        # [1]: cold exhaust area shall match the design one
        # [2]: Fan and LPC are linked together
        # [3]: HPC and HPT are linked together
        # [4]: LPC and LPT are linked together
        y = [r['9']['A'] / self.design_state['9']['A'] - 1., r['19']['A'] / self.design_state['19']['A'] - 1.,
             shaft_speed['Fan'] / shaft_speed['LPC'] - 1., shaft_speed['HPT'] / shaft_speed['HPC'] - 1.,
             shaft_speed['LPT'] / shaft_speed['LPC'] - 1.]
        # print(x, y)
        return y

    def magic_guess(self):
        """
        Provide a starting point for off-design computation which revealed to be very robust to flying conditions,
        but nothing is guaranty
        """
        guess = [0.7, 0.9, 0.9, 0.9, 0.9]
        return guess

    def draw_sankey(self, s, c, p, ax):
        """
        Draw the sankey diagram of the turbofan engine.

        For the turbofan, it looks like this:

        .. image:: sankey_turbofan.png

        :param s: stations dictionary
        :param c: component dictionary
        :param p: performance dictionary
        :param ax: the matplotlib axe object where to draw the diagram
        :type s: dict
        :type c: dict
        :type p: dict
        :type ax: plt.axes.Axes
        :return: nothing
        """
        scale = 1000. / (s['0']['Ht'] * s['0']['w'] + p['Pth'])
        sankey = Sankey(ax=ax, unit="kW", scale=scale, format='%.0f')

        # Inlet 0
        sankey.add(flows=[s['0']['Ex'] * s['0']['w'] / 1000.,
                          -s['1']['losses']['Inlet'] / 1000.,
                          -s['1']['Ex'] * s['1']['w'] / 1000.],
                   patchlabel='Inlet', orientations=[0, 1, 0], label='Inlet', fc='#00A000', labels=["", "Loss", None])

        # Fan 1
        dHF = s['13']['Ht'] - s['1']['Ht']
        sankey.add(flows=[s['1']['Ex'] * s['1']['w'] / 1000.,
                          -s['13']['losses']['Fan'] / 1000.,
                          dHF * s['13']['w'] / 1000.,
                          -(s['13']['Ex']) * s['13']['w'] / 1000.,
                          -(s['21']['Ex']) * s['21']['w'] / 1000.],
                   patchlabel="Fan", orientations=[0, 1, -1, 0, 0], prior=0, connect=(2, 0), label='Fan', fc='#00F000',
                   labels=["", "Loss", "", None, None])

        # Secondary Exhaust 2
        sankey.add(flows=[s['13']['Ex'] * s['13']['w'] / 1000.,
                          -s['19']['losses']['SE'] / 1000.,
                          -s['19']['losses']['Thermal'] / 1000.,
                          -s['19']['losses']['Jet loss'] / 1000.,
                          -s['19']['P'] / 1000.,
                          -s['0']['Ex'] * s['13']['w'] / 1000.],
                   patchlabel='Sec. Exhaust', orientations=[0, 1, 1, 1, 0, 0], prior=1, connect=(3, 0),
                   label='Sec. Exhaust', fc='#00FF00',
                   labels=["", "Loss", "Thermal loss", "Prop. loss", "Prop. Power", ""])

        # LPC 3
        dHL = s['25']['Ht'] - s['21']['Ht']
        sankey.add(flows=[s['21']['Ex'] * s['21']['w'] / 1000.,
                          -s['25']['losses']['LPC'] / 1000.,
                          dHL * s['25']['w'] / 1000.,
                          -(s['25']['Ex']) * s['25']['w'] / 1000.],
                   patchlabel='LPC', orientations=[0, 1, -1, 0], prior=1, connect=(4, 0), trunklength=2., label='LPC',
                   fc='#0000FF', labels=["", "Loss", "", None])

        # HPC 4
        dHH = s['3']['Ht'] - s['25']['Ht']
        sankey.add(flows=[s['25']['Ex'] * s['25']['w'] / 1000.,
                          -s['3']['losses']['HPC'] / 1000.,
                          -s['3']['losses']['Bleed'] / 1000.,
                          dHH * s['3']['w'] / 1000.,
                          -s['3']['Ex'] * s['3']['w'] / 1000.],
                   patchlabel='HPC', orientations=[0, 1, 1, -1, 0], prior=3, connect=(3, 0), label="HPC", fc='#0000F0',
                   labels=["", "Loss", "Bleed", "", None])

        # Burner 5
        dH = s['4']['Ht'] - s['3']['Ht']
        sankey.add(flows=[s['3']['Ex'] * s['3']['w'] / 1000.,
                          -s['4']['losses']['Burner'] / 1000.,
                          -s['4']['losses']['Burner Thermal'] / 1000.,
                          dH * s['4']['w'] / 1000.,
                          -s['4']['Ex'] * s['4']['w'] / 1000.],
                   patchlabel='Burner', orientations=[0, 1, 1, 1, 0], prior=4, connect=(4, 0), label='Burner',
                   fc='#F0F000', labels=["", "Loss", "Thermal", "Fuel", None])

        # HPT 6
        dHHT = s['43']['Ht'] - s['4']['Ht']
        sankey.add(flows=[s['4']['Ex'] * s['4']['w'] / 1000.,
                          -s['43']['losses']['HPT'] / 1000.,
                          dHHT * s['43']['w'] / 1000.,
                          -s['43']['Ex'] * s['43']['w'] / 1000.],
                   patchlabel='HPT', orientations=[0, 1, -1, 0], prior=5, connect=(4, 0), label='HPT', fc='#0000FF',
                   labels=["", "Loss", "", None])

        # LPT 7
        dHHL = s['5']['Ht'] - s['43']['Ht']
        sankey.add(flows=[s['43']['Ex'] * s['43']['w'] / 1000.,
                          -s['5']['losses']['LPT'] / 1000.,
                          dHHL * s['5']['w'] / 1000.,
                          -s['5']['Ex'] * s['5']['w'] / 1000.],
                   patchlabel='LPT', orientations=[0, 1, -1, 0], prior=6, connect=(3, 0), label='LPT', fc='#00F0F0',
                   labels=["", "Loss", "", None])

        # Primary Exhaust 8
        sankey.add(flows=[s['5']['Ex'] * s['5']['w'] / 1000.,
                          -s['9']['losses']['PE'] / 1000.,
                          -s['9']['losses']['Thermal'] / 1000.,
                          -s['9']['losses']['Jet loss'] / 1000.,
                          -s['9']['P'] / 1000.,
                          -s['0']['Ex'] * s['9']['w'] / 1000.],
                   patchlabel='Pri. Exhaust', orientations=[0, 1, 1, 1, 0, 0], prior=7, connect=(3, 0),
                   label='Pri. Exhaust', fc='#FF0000',
                   labels=["", "Loss", "Thermal loss", "Prop. Loss", "Prop. Power", ""])

        # HP shaft 9
        sankey.add(flows=[-dHH * s['3']['w'] / 1000., -dHHT * s['43']['w'] / 1000., -c['HPShaft']['HPX']/1000.],
                   patchlabel='HP shaft', orientations=[-1, -1, 1], prior=6, connect=(2, 1), trunklength=3.575,
                   label='HP shaft', fc='#F0F0F0', labels=[None, None, "Offtake"])

        # LP shaft 10
        sankey.add(flows=[-dHF * s['13']['w'] / 1000., -dHL * s['25']['w'] / 1000, -dHHL * s['5']['w'] / 1000.,
                          -c['LPShaft']['LPX']/1000.],
                   patchlabel='LP shaft', orientations=[0, -1, -1, 1], prior=7, connect=(2, 2), trunklength=6.582,
                   pathlengths=[0.1, 1., 1., 0.3], label='LP shaft', fc='#A0A0A0', labels=[None, None, None, "Offtake"])
        sankey.add(flows=[-dHF * s['13']['w'] / 1000., dHF * s['13']['w'] / 1000.],
                   orientations=[-1, 0], prior=10, connect=(0, 1), pathlengths=1.05, trunklength=0.905,
                   fc='#A0A0A0', labels=[None, None])

        diagrams = sankey.finish()
        ax.set_title("Power flows")

        # ax.legend()


class Turboprop(ExergeticEngine):
    """
    Defines a 3-shaft turboprop
    """
    def __init__(self):
        super(Turboprop, self).__init__()
        # dictionary with the slope of corrected flow vs. corrected speed,
        # except PWT: corrected flow vs. expansion ratio
        self.slope = {"LPC": 0.765, "HPC": 2.22, "HPT": -25.4, "LPT": 13.92, 'PWT': -11.84}

    def cycle(self, *args, **kwargs):
        """
        Compute the performance of the engine, based on given cycle parameters

        This present implementation is for a 3-shaft turboprop.

        The positional argument are:
            * the engine core mass flow w, in kg/s
            * the specific shaft power
            * the low pressure compressor temperature ratio tau_l.
            * the high pressure compressor temperature ratio tau_h.

        Only one keyed argument is needed: dHf or Ttmax.
            * dHf is the fuel power in Watts,
            * Ttmax is the combustion chamber exit temperature in Kelvin.

        Other possible keyed arguments:
            * HPX: some customer power extraction on the HP shaft (if positive), in Watts.
                A negative value means you provide additional power, via some electric motor for instance.
            * LPX: some customer power extraction on the LP shaft (if positive), in Watts.
                A negative value means you provide additional power, via some electric motor for instance.
            * PGBX: some customer power extraction on the power turbine shaft (if positive), in Watts.
                A negative value means you provide additional power, via some electric motor for instance.

        The computed dictionaries are:
            * the stations states: the state of the air flow at each component interfaces
            * the components state:

                * the keys are the component names
                * The content depends on the component. It includes pressure ratio, map coordinates, ...

            * the performances, see :meth:`ExergeticEngine.global_performances`

        :param args: cycle parameters
        :param kwargs: optional cycle parameters
        :type args: tuple
        :type kwargs: dict
        :return: 3 dictionaries with the state of the machine
        :rtype: (dict, dict, dict)
        """
        w, shp, tau_l, tau_h = args
        dHf = kwargs.get('dHf')
        Ttmax = kwargs.get('Ttmax')

        # Station 0: free stream
        components = {"LPC": {"tau": tau_l}, "HPC": {"tau": tau_h}}
        stations = {"0": self.free_stream(w)}

        # station 1: LPC face
        stations["1"] = self.loose_pressure(self.ex_loss["inlet"], stations['0'], "Inlet")
        components["LPC"]["WR"] = stations["1"]['Wr']
        components["LPC"]["Nk"] = stations["1"]['Nk']

        # station 25: LPC exit
        stations["25"] = self.compress(self.ex_loss["LPC"], stations["1"], tau_l, "LPC")
        dH_LPC = (stations["25"]['Ht'] - stations["1"]['Ht']) * stations['1']['w']
        components["LPC"]["PR"] = stations["25"]['PR']

        # station 3: HPC exit
        components["HPC"]["Nk"] = stations['25']['Nk']
        components["HPC"]["WR"] = stations['25']['Wr']
        stations["3"] = self.compress(self.ex_loss["HPC"], stations["25"], tau_h, "HPC")
        dH_HPC = (stations["3"]['Ht'] - stations["25"]['Ht']) * stations['25']['w']
        components["HPC"]["PR"] = stations["3"]['PR']

        # station 4: Burner exit
        stations['4'] = self.burner(self.ex_loss["Burner"], stations['3'], "Burner", dH=dHf, Ttm=Ttmax)
        components["Burner"] = {"dHF": stations['4']['Pth']}

        # station 43: HPT exit
        components["HPT"] = {'Nk': stations['4']['Nk'], 'WR': stations['4']['Wr']}
        dH_HPT = -dH_HPC - kwargs.get("HPX", 0.)
        stations["43"] = self.expand(self.ex_loss['HPT'], stations['4'], dH_HPT, "HPT")
        components['HPShaft'] = {'HPX': kwargs.get("HPX", 0.)}

        # station 45: LPT exit
        components["LPT"] = {'Nk': stations['43']['Nk'], 'WR': stations['43']['Wr']}
        dH_LPT = -dH_LPC - kwargs.get("LPX", 0.)
        stations["45"] = self.expand(self.ex_loss['LPT'], stations['43'], dH_LPT, "LPT")
        components['LPShaft'] = {'LPX': kwargs.get("LPX", 0.)}

        # station 5: PWT exit
        components["PWT"] = {'Nk': stations['45']['Nk'], 'WR': stations['45']['Wr']}
        dH_PWT = - shp * w - kwargs.get("PGBX", 0.)
        stations["5"] = self.expand(self.ex_loss['PWT'], stations['45'], dH_PWT, "PWT")
        components["PWT"]['PR'] = stations['5']['PR']
        components['PropShaft'] = {'PGBX': kwargs.get("PGBX", 0.)}

        # station 9: hot flow exhaust
        stations['9'] = self.exhaust(self.ex_loss["PE"], stations['5'], "PE")

        # station 10: propeller
        stations['10'] = copy.deepcopy(stations['0'])
        stations['10']['w'] = 0.
        stations['10'].pop('A')
        stations['10'].pop('Wr')
        stations['10']["Pu"] = w * shp * (1. - self.ex_loss['prop'])*(1 - self.ex_loss.get("PGB", 0.))
        stations['10']['PEx'] = stations['10']["Pu"]
        stations['10']['F'] = stations['10']["Pu"] / stations['0']['V0']
        stations['10']["losses"] = {"Propeller": w * shp * (1. - self.ex_loss.get("PGB", 0.)) * self.ex_loss['prop'],
                                    "PGB": w * shp * self.ex_loss.get("PGB", 0.)}

        perfs = self.global_performances(stations)
        return stations, components, perfs

    def design_guess(self, *args):
        """
        This present implementation is for a 2-shaft 2-flow turbofan.

        Two values are expected in args:
            * the required thrust
            * the required compressors pressure ratio

        The cycle parameters returned are the mass flow, the specific shaft power, and the compression temperature
        ratios, estimated using the method :meth:`ExergeticEngine.from_PR_to_tau_pol`

        :param args: the performance values sought
        :type param: tuple
        :return: a guess for the cycle parameters
        :rtype: np.array
        """
        thrust, lpr, hpr = args
        tau_l, lloss = self.from_PR_to_tau_pol(lpr, 1. - self.ex_loss['LPC'])
        tau_h, hloss = self.from_PR_to_tau_pol(hpr, 1. - self.ex_loss['HPC'])
        shp = thrust * self.V0 / (1. - self.ex_loss["prop"])
        return np.array([shp / self.V0**2, self.V0**2, tau_l, tau_h])

    def design_equations(self, x, *args):
        """
        This function creates the link between the performance parameters sought (in args[0]) and the cycle results.
        The cycle parameters are in x.

        This present implementation is for a 3-shaft turboprop. The values sought are the total net thrust,
        the 2 compressors pressure ratio (low pressure compressor, high pressure compressor), and the exhaust pressure
        ratio.

        :param x: the cycle parameters
        :param args: a tuple with 2 items: the performance value sought and the keyed arguments for the cycle
        :type x: tuple
        :type args: tuple
        :return: a vector to be nullified
        :rtype: np.array
        """
        thrust, lpr, hpr, exhpr = args[0]
        other_kwargs = args[1]
        s, c, p = self.cycle(*x, **other_kwargs)
        y = [p['Fnet'] / thrust - 1., c['LPC']['PR'] / lpr - 1., c['HPC']['PR'] / hpr - 1., s['9']['Pt']/self.Ps0 - exhpr]
        return np.array(y)

    def off_design_equations(self, x, *args):
        """
        This present implementation is for a 2-shaft 2-flow turbofan.

        This function implements the various equations to be satisfied in off-design conditions.

        for the turbofan, these are:
            * equalization of the hot exhaust area with the one required by the cycle
            * equalization of the low pressure compressor physical speed with the low pressure turbine physical speed
            * equalization of the high pressure compressor physical speed with the high pressure turbine physical speed
            * equalization of the propeller physical speed with the power turbine physical speed


        The shaft speeds are normally obtained via a component map. Here it has been simplified as a straight line.

        :param x: vector of values around 1. when multiplied by the design point cycle parameter, gives the tested cycle
            parameters
        :type x: np.array
        :param args: a tuple with one dictionary: the additional cycle parameters
        :return: a vector of same size as x, to be nullified
        :rtype: np.array
        """
        xx = x * self.dp
        r, c, p = self.cycle(*xx, **args[0])
        # Get the shaft speed compatible with the corrected flow, as a fraction of the design speed
        # Get the shaft speed compatible with the corrected flow, as a fraction of the design speed
        shaft_speed = {i: ((self.slope[i] * (c[i]['WR'] / self.comp_state[i]['WR'] - 1.) + 1.) *
                           self.comp_state[i]['Nk'] / c[i]['Nk']) for i in self.slope.keys()}
        # the power turbine is a particular case: the shaft speed is fixed by the propeller
        shaft_speed['PWT'] = (self.slope['PWT'] * (c['PWT']['WR'] / self.comp_state['PWT']['WR'] - 1.) + 1.) * \
                             (self.comp_state['PWT']['PR'] - 1.) / (c['PWT']['PR'] - 1.)
        # vector to be nullified:
        # [0]: hot exhaust area shall match the design one
        # [1]: HPC and HPT are linked together
        # [2]: LPC and LPT are linked together
        # [3]: propeller and PWT are linked together (fixed speed)
        y = [r['9']['A'] / self.design_state['9']['A'] - 1.,
             shaft_speed['HPT'] / shaft_speed['HPC'] - 1.,
             shaft_speed['LPT'] / shaft_speed['LPC'] - 1.,
             shaft_speed['PWT'] - 1.]
        return y

    def draw_sankey(self, s, c, p, ax):
        """
        Draw the sankey diagram of the turboprop engine.

        For the turboprop, it looks like this:

        .. image:: sankey_turboprop.png

        :param s: stations dictionary
        :param c: component dictionary
        :param p: performance dictionary
        :param ax: the matplotlib axe object where to draw the diagram
        :type s: dict
        :type c: dict
        :type p: dict
        :type ax: plt.axes.Axes
        :return: nothing
        """
        scale = 1000. / (s['0']['Ht'] * s['0']['w'] + p['Pth'])
        sankey = Sankey(ax=ax, unit="kW", scale=scale, format='%.0f')

        # Inlet 0
        sankey.add(flows=[s['0']['Ex'] * s['0']['w'] / 1000.,
                          -s['1']['losses']['Inlet'] / 1000.,
                          -s['1']['Ex'] * s['1']['w'] / 1000.],
                   patchlabel='Inlet', orientations=[0, 1, 0], label='Inlet', fc='#00A000', labels=["", "Loss", None])

        # LPC 1
        dHL = s['25']['Ht'] - s['1']['Ht']
        sankey.add(flows=[s['1']['Ex'] * s['1']['w'] / 1000.,
                          -s['25']['losses']['LPC'] / 1000.,
                          dHL * s['25']['w'] / 1000.,
                          -(s['25']['Ex']) * s['25']['w'] / 1000.],
                   patchlabel='LPC', orientations=[0, 1, -1, 0], prior=0, connect=(2, 0), label='LPC',
                   fc='#0000FF', labels=["", "Loss", "", None])

        # HPC 2
        dHH = s['3']['Ht'] - s['25']['Ht']
        sankey.add(flows=[s['25']['Ex'] * s['25']['w'] / 1000.,
                          -s['3']['losses']['HPC'] / 1000.,
                          dHH * s['3']['w'] / 1000.,
                          -s['3']['Ex'] * s['3']['w'] / 1000.],
                   patchlabel='HPC', orientations=[0, 1, -1, 0], prior=1, connect=(3, 0), label="HPC", fc='#0000F0',
                   labels=["", "Loss", "", None])

        # Burner 3
        dH = s['4']['Ht'] - s['3']['Ht']
        sankey.add(flows=[s['3']['Ex'] * s['3']['w'] / 1000.,
                          -s['4']['losses']['Burner'] / 1000.,
                          -s['4']['losses']['Burner Thermal'] / 1000.,
                          dH * s['4']['w'] / 1000.,
                          -s['4']['Ex'] * s['4']['w'] / 1000.],
                   patchlabel='Burner', orientations=[0, 1, 1, 1, 0], prior=2, connect=(3, 0), label='Burner',
                   fc='#F0F000', labels=["", "Loss", "Thermal", "Fuel", None])

        # HPT 4
        dHHT = s['43']['Ht'] - s['4']['Ht']
        sankey.add(flows=[s['4']['Ex'] * s['4']['w'] / 1000.,
                          -s['43']['losses']['HPT'] / 1000.,
                          dHHT * s['43']['w'] / 1000.,
                          -s['43']['Ex'] * s['43']['w'] / 1000.],
                   patchlabel='HPT', orientations=[0, 1, -1, 0], prior=3, connect=(4, 0), label='HPT', fc='#0000FF',
                   labels=["", "Loss", "", None])

        # LPT 5
        dHHL = s['45']['Ht'] - s['43']['Ht']
        sankey.add(flows=[s['43']['Ex'] * s['43']['w'] / 1000.,
                          -s['45']['losses']['LPT'] / 1000.,
                          dHHL * s['45']['w'] / 1000.,
                          -s['45']['Ex'] * s['45']['w'] / 1000.],
                   patchlabel='LPT', orientations=[0, 1, -1, 0], prior=4, connect=(3, 0), label='LPT', fc='#00F0F0',
                   labels=["", "Loss", "", None])

        # PWT 6
        dHP = s['5']['Ht'] - s['45']['Ht']
        sankey.add(flows=[s['45']['Ex'] * s['45']['w'] / 1000.,
                          -s['5']['losses']['PWT'] / 1000.,
                          dHP * s['5']['w'] / 1000.,
                          -s['5']['Ex'] * s['5']['w'] / 1000.],
                   patchlabel='PWT', orientations=[0, 1, -1, 0], prior=5, connect=(3, 0), label='PWT', fc='#00F0F0',
                   labels=["", "Loss", "", None])

        # Primary Exhaust 7
        sankey.add(flows=[s['5']['Ex'] * s['5']['w'] / 1000.,
                          -s['9']['losses']['PE'] / 1000.,
                          -s['9']['losses']['Thermal'] / 1000.,
                          -s['9']['losses']['Jet loss'] / 1000.,
                          -s['9']['P'] / 1000.,
                          -s['0']['Ex'] * s['9']['w'] / 1000.],
                   patchlabel='Pri. Exhaust', orientations=[0, 1, 1, 1, 0, 0], prior=6, connect=(3, 0),
                   label='Pri. Exhaust', fc='#FF0000',
                   labels=["", "Loss", "Thermal loss", "Prop. Loss", "Prop. Power", ""])

        # HP shaft 8
        sankey.add(flows=[-dHH * s['3']['w'] / 1000, -dHHT * s['4']['w'] / 1000., c['HPShaft']['HPX']/1000.],
                   patchlabel='HP shaft', orientations=[-1, -1, 1], prior=4, connect=(2, 1), trunklength=4.018,
                   label='HP shaft', fc='#F0F0F0', labels=[None, None, "Offtake"])

        # LP shaft 9
        sankey.add(flows=[-dHL * s['25']['w'] / 1000, -dHHL * s['43']['w'] / 1000., c['LPShaft']['LPX']/1000.],
                   patchlabel='LP shaft', orientations=[-1, -1, 1], prior=5, connect=(2, 1), trunklength=6.187,
                   pathlengths=[1., 1., 0.3], label='LP shaft', fc='#A0A0A0', labels=[None, None, "Offtake"])

        # Prop shaft 10
        sankey.add(flows=[-dHP * s['5']['w'] / 1000., dHP * s['5']['w'] / 1000., c['PropShaft']['PGBX']/1000.],
                   orientations=[0, 1, -1], prior=6, connect=(2, 0), patchlabel='Prop shaft',
                   fc='#A0A0A0', labels=[None, None, "Offtake"])

        # Propeller 11
        sankey.add(flows=[-dHP * s['5']['w'] / 1000., -s['10']['Pu']/1000., -s['10']['losses']['Propeller']/1000.,
                          -s['10']['losses']['PGB']/1000.],
                   orientations=[0, 0, -1, -1], prior=10, connect=(1, 0),
                   fc='#A0A0A0', labels=["", "Prop. Power", "Prop. Loss", "PB Loss"])

        diagrams = sankey.finish()
        ax.set_title("Power flows")
        # ax.legend()


class ElectricFan(ExergeticEngine):
    """
    Defines an electric fan, based on the ExergeticEngine object.

    This cycle support wake ingestion.
    """
    def __init__(self):
        """
        Constructor for the ElectricFan object, with value for the relation between fan corrected flow and
        fan corrected speed.
        """
        super(ElectricFan, self).__init__()
        # dictionary with the slope of corrected flow vs. corrected speed
        self.slope = {"Fan": 1.201}

    def cycle(self, *args, **kwargs):
        """
        Compute the performance of the engine, based on given cycle parameters

        This present implementation is for an electric fan.

        The positional argument are:
            * the engine core mass flow w, in kg/s
            * the fan temperature ratio tau_f.

        Other possible keyed arguments:
            * d_bli: drag of the body in front of the engine, in Newtons.

        The computed dictionaries are:
            * the stations states: the state of the air flow at each component interfaces
            * the components state:

                * the keys are the component names
                * The content depends on the component. It includes pressure ratio, map coordinates, ...

            * the performances, see :meth:`ExergeticEngine.global_performances`

        :param args: cycle parameters
        :param kwargs: optional cycle parameters
        :type args: tuple
        :type kwargs: dict
        :return: 3 dictionaries with the state of the machine
        :rtype: (dict, dict, dict)
        """
        w, tau_f = args
        d_bli = kwargs.get('d_bli')

        # Station 0: free stream
        components = {"Fan": {"tau": tau_f}}
        stations = {"0": self.free_stream(w)}

        # wake ingestion
        stations["1"] = self.wake_ingestion(d_bli, stations["0"], 'BLI')

        # station 2: Engine face
        stations["2"] = self.loose_pressure(self.ex_loss["inlet"], stations['1'], "Inlet")
        components["Fan"]["WR"] = stations["2"]['Wr']
        components["Fan"]["Nk"] = stations["2"]['Nk']

        # station 3: Fan exit
        stations["3"] = self.compress(self.ex_loss["Fan"], stations["2"], tau_f, "Fan")
        stations['3']['Pth'] = (stations["3"]['Ht'] - stations["2"]['Ht']) * stations['2']['w']
        components["Fan"]["PR"] = stations["3"]['PR']

        # station 9: hot flow exhaust
        stations['9'] = self.exhaust(self.ex_loss["PE"], stations['3'], "PE")

        perfs = self.global_performances(stations)

        return stations, components, perfs

    def design_guess(self, *args):
        """
        This present implementation is for an electric fan.

        Two values are expected in args:
            * the required thrust
            * the required fan pressure ratio

        The cycle parameters returned are the mass flow, estimated assuming a specific thrust of 100m/s,
        and the fan temperature ratio, estimated using the method :meth:`ExergeticEngine.from_PR_to_tau_pol`

        :param args: the performance values sought
        :type param: tuple
        :return: a guess for the cycle parameters
        :rtype: np.array
        """
        thrust, fpr = args
        tau_f, floss = self.from_PR_to_tau_pol(fpr, 1. - self.ex_loss['Fan'])
        return np.array([thrust / 100., tau_f])

    def design_equations(self, x, *args):
        """
        This function creates the link between the performance parameters sought (in args[0]) and the cycle results.
        The cycle parameters are in x.

        This present implementation is for an electric fan. The values sought are the total net thrust,
        and the fan pressure ratio.

        :param x: the cycle parameters
        :param args: a tuple with 2 items: the performance value sought and the keyed arguments for the cycle
        :type x: tuple
        :type args: tuple
        :return: a vector to be nullified
        :rtype: np.array
        """
        thrust, fpr = args[0]
        other_kwargs = args[1]
        s, c, p = self.cycle(*x, **other_kwargs)
        y = [p['Fnet'] / thrust - 1., c['Fan']['PR'] / fpr - 1.]
        return np.array(y)

    def off_design_equations(self, x, *args):
        """
        This present implementation is for an electric fan.

        This function implements the various equations to be satisfied in off-design conditions.

        for the turbofan, these are:
            * equalization of the exhaust area with the one required by the cycle
            * equalization of the fan physical speed with prescribed one

        The shaft speed is normally obtained via a component map. Here it has been simplified as a straight line.

        :param x: vector of values around 1. when multiplied by the design point cycle parameter, gives the tested cycle
            parameters
        :type x: np.array
        :param args: a tuple with one dictionary: the additional cycle parameters
        :return: a vector of same size as x, to be nullified
        :rtype: np.array
        """
        xx = x * self.dp
        throttle = args[0]['throttle']
        r, c, p = self.cycle(*xx, **args[0])
        # Get the shaft speed compatible with the corrected flow, as a fraction of the design speed
        shaft_speed = {i: ((self.slope[i] * (c[i]['WR'] / self.comp_state[i]['WR'] - 1.) + 1.) *
                           self.comp_state[i]['Nk'] / c[i]['Nk']) for i in self.slope.keys()}
        # vector to be nullified:
        # [0]: exhaust area shall match the design one
        # [1]: fan shaft physical speed shall match the prescribed one
        y = [r['9']['A'] / self.design_state['9']['A'] - 1., shaft_speed['Fan'] / throttle - 1.]
        # print(x, y)
        return y

    def draw_sankey(self, s, c, p, ax):
        """
        Draw the sankey diagram of the electric fan engine.

        For the electric fan , it looks like this:

        .. image:: sankey_efan.png

        :param s: stations dictionary
        :param c: component dictionary
        :param p: performance dictionary
        :param ax: the matplotlib axe object where to draw the diagram
        :type s: dict
        :type c: dict
        :type p: dict
        :type ax: plt.axes.Axes
        :return: nothing
        """
        scale = 1000. / (s['0']['Ht'] * s['0']['w'] + p['Pth'])
        sankey = Sankey(ax=ax, unit="kW", scale=scale, format='%.0f')

        # BLI 0
        sankey.add(flows=[s['0']['Ex'] * s['0']['w'] / 1000.,
                          -s['1']['gain']['BLI'] / 1000.,
                          -s['1']['Ex'] * s['1']['w'] / 1000.],
                   patchlabel='BLI', orientations=[0, 1, 0], label='BLI', fc='#00A000', labels=["", "Loss", None])

        # Inlet 1
        sankey.add(flows=[s['1']['Ex'] * s['1']['w'] / 1000.,
                          -s['2']['losses']['Inlet'] / 1000.,
                          -s['2']['Ex'] * s['2']['w'] / 1000.],
                   patchlabel='Inlet', orientations=[0, 1, 0], prior=0, connect=[2, 0], label='Inlet', fc='#00A000',
                   labels=["", "Loss", None])

        # Fan 2
        dH = s['3']['Ht'] - s['2']['Ht']
        sankey.add(flows=[s['2']['Ex'] * s['2']['w'] / 1000.,
                          -s['3']['losses']['Fan'] / 1000.,
                          dH * s['3']['w'] / 1000.,
                          -s['3']['Ex'] * s['3']['w'] / 1000.],
                   patchlabel='Fan', orientations=[0, 1, -1, 0], prior=1, connect=(2, 0), label="Fan", fc='#0000F0',
                   labels=["", "Loss", "", None])

        # Primary Exhaust 2
        sankey.add(flows=[s['3']['Ex'] * s['3']['w'] / 1000.,
                          -s['9']['losses']['PE'] / 1000.,
                          -s['9']['losses']['Thermal'] / 1000.,
                          -s['9']['losses']['Jet loss'] / 1000.,
                          -s['9']['P'] / 1000.,
                          -s['0']['Ex'] * s['9']['w'] / 1000.],
                   patchlabel='Pri. Exhaust', orientations=[0, 1, 1, 1, 0, 0], prior=2, connect=(3, 0), label='Pri. Exhaust',
                   fc='#FF0000', labels=["", "Loss", "Thermal loss", "Prop. Loss", "Prop. Power", ""])

        # HP shaft 9
        sankey.add(flows=[dH * s['3']['w'] / 1000,
                          -dH * s['3']['w'] / 1000.],
                   patchlabel='Shaft', orientations=[0, -1], prior=2, connect=(2, 1), trunklength=1.,
                   label='Shaft', fc='#F0F0F0', labels=[None, None])

        diagrams = sankey.finish()
        ax.set_title("Power flows")
        # ax.legend()


if __name__ == '__main__':
    # Instantiate the engine object
    TF = ExergeticEngine()
    # Set the flight conditions to 35000ft, ISA, Mn 0.78 as static temperature, static pressure and Mach number
    TF.set_flight(218.808, 23842.272, 0.78)

    TF.ex_loss["inlet"] = TF.from_PR_loss_to_Ex_loss(0.997)
    tau_h, TF.ex_loss["HPC"] = TF.from_PR_to_tau_pol(20., 0.85)
    TF.ex_loss["Burner"] = TF.from_PR_loss_to_Ex_loss(0.95)
    aut_h, TF.ex_loss["HPT"] = TF.from_PR_to_tau_pol(0.289451726, 0.90)
    TF.ex_loss["PE"] = TF.from_PR_loss_to_Ex_loss(0.99)
    print(TF.ex_loss)
    x0 = np.array([24., tau_h])

    # Design for a given thrust (Newton), BPR, FPR, LPC PR, HPC PR, T41 (Kelvin)
    Ttm = 1750.
    rd, c, p = TF.cycle(24., tau_h, Ttmax=Ttm)
    print(p['Fnet'])
    rd, c, p = TF.design(21400., 20., Ttmax=Ttm, guess=x0)
    # show the results
    TF.print_stations(rd)
    TF.print_perfos(p)
    TF.print_components(c)

    # recompute the same, but in off-design mode
    rd, c, p = TF.off_design(wfe=p['wfe'])

    fig = plt.figure(facecolor="w", tight_layout=True)
    ax1 = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    TF.draw_sankey(rd, c, p, ax1)
    plt.show()
    fig = plt.figure(facecolor="w", tight_layout=True)
    ax2 = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    TF.draw_pie(rd, ax2)
    plt.show()

    guess = p["sol"]
    # now for a cruise loop
    Tm = [1.1, 1.08, 1.06, 1.04, 1.02, 1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76,
          0.74, 0.72, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60]
    # Tm = [k*Ttm for k in kwf]
    Fn = []
    sfc = []
    for T in Tm:
        r, c, p = TF.off_design(guess=guess, Ttmax=T*Ttm)
        if p['success']:
            guess = p['sol']
            Fn.append(p['Fnet'] / 10.)
            sfc.append(p['wfe'] * 36000. / p['Fnet'])

    print("\t".join(["{}".format(T*Ttm) for T in Tm]))
    print("\t".join(["{}".format(F) for F in Fn]))
    print("\t".join(["{}".format(s) for s in sfc]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Fn, sfc, '-o')
    ax.grid()
    plt.show()
    # now for a take-off point
    TF.set_flight(288.15, 101325., 0.25)
    x0 = np.array([10., 1.])
    rd = TF.off_design(Ttmax=Ttm, guess=x0)

    # show_dict(rd)
