#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.optimize import fsolve
import unit

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import rc
font = {'size':15}
rc('font',**font)




class SmallPlane(object):

    def __init__(self, npax=4, alt=unit.m_ft(3000), dist=unit.m_km(500), tas=unit.mps_kmph(180), mode="classic"):
        # Earth and atmosphere
        self.g = 9.80665
        self.disa = 0.

        # Top level requirements
        self.alt = alt
        self.vtas = tas
        self.distance = dist
        self.n_pax = npax
        self.mode = mode  # "classic" or "electric"

        # Additionnal informations
        self.m_pax = 90
        self.diversion_time = unit.s_min(30)
        self.lod = 14  # Aerodynamic efficiency

        # Propulsion
        self.prop_eff = 0.80
        self.motor_eff = 0.95  # MAGNIX

        # Energy storage
        self.psfc = unit.convert_from("lb/shp/h", 0.6)
        self.fuel_hv = unit.J_MJ(43)  # gasoline
        self.piston_eng_pw_density = unit.W_kW(1)  # W/kg
        self.elec_motor_pw_density = unit.W_kW(4.5)  # W/kg   MAGNIX
        self.power_elec_pw_density = unit.W_kW(10)  # W/kg
        self.battery_enrg_density = unit.J_Wh(200)  # Wh/kg

        # design results
        self.design = None

    def __str__(self):
        """Print main figures
        """
        s = ["\nAirplane : %s" %self.mode,
             "Npax     %d" % self.n_pax,
             "Distance %d km" % unit.km_m(self.distance),
             "TAS      %d km/h" % unit.kmph_mps(self.vtas),
             "Altitude %d ft" % unit.ft_m(self.alt),
             "---------------------------------"]

        if self.design==None:
            s.append(">>  NO DESIGN  <<")
            return "\n".join(s) # print only recquirements

        s.append("Max power = %.0f kW" %unit.kW_W(self.design["pw_max"]))
        if self.design["airplane_mode"]=="classic":
            s.append("Mission fuel = %.0f kg" %self.design["mission_fuel"])
            s.append("Reserve fuel = %.0f kg" %self.design["reserve_fuel"])
            s.append("Total fuel = %.0f kg" %self.design["total_fuel"])
        elif self.design["airplane_mode"]=="electric":
            s.append("Cruise power = %.0f kW" %unit.kW_W(self.design["pw_cruise"]))
            s.append("Battery mass = %.0f kg" %self.design["battery_mass"])
        else:
            raise Exception("Aircraft mode is unknown")
        s.append("Mission energy = %.0f kWh" %unit.kWh_J(self.design["mission_energy"]))
        s.append("Reserve energy = %.0f kWh" %unit.kWh_J(self.design["reserve_energy"]))
        s.append("Total energy = %.0f kWh" %unit.kWh_J(self.design["total_energy"]))
        s.append("MTOW = %.0f kg"%self.design["mtow"])
        s.append("OWE = %.0f kg"%self.design["owe"])
        s.append("Payload = %.0f kg"%self.design["payload"])
        s.append("")
        s.append("PK / MTOW minimum = %.2f pk/km" %self.design["pk_o_m_min"])
        s.append("PK / MTOW = %.2f pk/kg" %self.design["pk_o_m"])
        s.append("PK / Energy = %.2f pk/kWh"%self.design["pk_o_e"])
        return "\n|\t".join(s)

    def breguet(self, tow):
        """Used for classical airplane burning gasoline
        """
        fuel_mission = tow*(1.-np.exp(-(self.psfc*self.g*self.distance)/(self.prop_eff*self.lod)))   # piston engine
        fuel_reserve = (tow-fuel_mission)*(1.-np.exp(-(self.psfc*self.g*self.vtas*self.diversion_time)/(self.prop_eff*self.lod)))
        return fuel_mission, fuel_reserve

    def classic_design(self, mtow):
        """Aggregate all discipline to compute design point characteristics of a classicla airplane burning gasoline
        """
        pw_max = self.max_power(mtow)
        owe = self.basic_owe(mtow)
        fuel_mission, fuel_reserve = self.breguet(mtow)
        fuel_total = fuel_mission + fuel_reserve
        payload = self.n_pax * self.m_pax
        total_energy = fuel_total * self.fuel_hv
        return {"airplane_mode":self.mode,
                "pw_max":pw_max,
                "mission_fuel":fuel_mission,
                "reserve_fuel":fuel_reserve,
                "total_fuel":fuel_total,
                "mission_energy":fuel_mission * self.fuel_hv,
                "reserve_energy":fuel_reserve*self.fuel_hv,
                "total_energy":total_energy,
                "mtow":mtow,
                "owe":owe,
                "payload":payload,
                "pk_o_m_min":unit.km_m(self.distance)/670,
                "pk_o_m":self.n_pax*unit.km_m(self.distance)/mtow,
                "pk_o_e":self.n_pax*unit.km_m(self.distance)/unit.kWh_J(total_energy)}

    def aerodynamic(self, mass):
        """Compute required thrust at cruise point
        """
        fn = mass * self.g / self.lod
        return fn

    def propulsion(self, pw_max, fn):
        """Copute required power at cruise point and engine mass
        """
        pw = fn * self.vtas / (self.prop_eff*self.motor_eff)
        m_engine = pw_max / self.elec_motor_pw_density
        return pw, m_engine

    def energy_system(self, pw_max, pw, mass):
        """Compute added system and battery masses
        """
        m_system = pw_max / self.power_elec_pw_density
        m_battery = (  pw * (self.distance/self.vtas)
                     + mass*self.g*self.alt/(self.motor_eff*self.prop_eff)
                    ) / self.battery_enrg_density
        return m_system, m_battery

    def regulation(self, pw):
        """Compute additional battery mass for reserve
        """
        m_reserve = (pw * self.diversion_time) / self.battery_enrg_density
        return m_reserve

    def max_power(self, mtow):
        """Estimate max installed power
        """
        pw_max = (0.0197*mtow + 100.6)*mtow
        return pw_max

    def basic_owe(self, mtow):
        """Estimate classical airplane empty weight
        """
        # owe_basic = (-9.6325e-07 * mtow + 6.1041e-01) * mtow
        owe_basic = 0.606 * mtow
        return owe_basic

    def full_elec_design(self, mtow):
        """Aggregate all discipline outputs to compute design point characteristics of a full electric airplane
        """
        pw_max = self.max_power(mtow)
        owe_basic = self.basic_owe(mtow)
        fn = self.aerodynamic(mtow)
        pw, m_engine = self.propulsion(pw_max,fn)
        m_system, m_battery = self.energy_system(pw_max, pw, mtow)
        m_reserve = self.regulation(pw)
        owe =  owe_basic \
             - pw_max / self.piston_eng_pw_density \
             + m_engine + m_system + m_battery + m_reserve
        payload = self.n_pax * self.m_pax
        total_energy = (m_battery + m_reserve) * self.battery_enrg_density
        return {"airplane_mode":self.mode,
                "pw_max":pw_max,
                "pw_cruise":pw,
                "fn_cruise":fn,
                "battery_mass":m_battery+m_reserve,
                "mission_energy":m_battery*self.battery_enrg_density,
                "reserve_energy":m_reserve*self.battery_enrg_density,
                "total_energy":total_energy,
                "mtow":mtow,
                "owe":owe,
                "payload":payload,
                "pk_o_m_min":unit.km_m(self.distance)/670,
                "pk_o_m":self.n_pax*unit.km_m(self.distance)/mtow,
                "pk_o_e":self.n_pax*unit.km_m(self.distance)/unit.kWh_J(total_energy)}

    def old_design_solver(self):
        """Compute the design point
        """
        def fct(mtow):
            if self.mode=="classic":
                dict = self.classic_design(mtow)
                return mtow - dict["owe"] - dict["payload"] - dict["total_fuel"]
            elif self.mode=="electric":
                dict = self.full_elec_design(mtow)
                return mtow - dict["owe"] - dict["payload"]
            else:
                raise Exception("Aircraft mode is unknown")

        mtow_ini = 5000
        output_dict = fsolve(fct, x0=mtow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        mtow = output_dict[0][0]

        if self.mode=="classic":
            self.design = self.classic_design(mtow)
        elif self.mode=="electric":
            self.design = self.full_elec_design(mtow)

    def design_solver(self, err=1e-4, maxiter=200):
        """ Uses a fixed point iteration procedure to solve the design. It recquires more iterations than a gradient based method,
        but convergences towards the attractive solution point (positive MTOW).

        The procedure starts with an initial guess for mtow.
        The first iteration computes the airplane design from this initial guess. This results in a new mtow.
        Then we iteratively solve f(mtow)=mtow, with f() the MTOW computed by the aircraft design.

        * err : the relative precision criterion for convergence. Default 1e-4.
        """

        if self.mode=="classic":
            def fct(mtow):
                design = self.classic_design(mtow)
                return design["owe"] + design["payload"] + design["total_fuel"]
        elif self.mode=="electric":
            def fct(mtow):
                design = self.full_elec_design(mtow)
                return design["owe"] + design["payload"]
        else:
            raise Exception("Aircraft mode is unknown")

        mtow_old =0
        mtow=2000 # initial guess
        i = 0
        while (mtow-mtow_old)/mtow>err and i<maxiter:
            mtow_old = mtow
            mtow = fct(mtow)
            i+=1

        if i == maxiter:
            raise RuntimeWarning("Convergence problem, exceed max number of iterations.")

        if self.mode == "classic":
            self.design = self.classic_design(mtow)
        elif self.mode == "electric":
            self.design = self.full_elec_design(mtow)

    def max_distance(self):
        """Compute the design that brings the minimum value for the PK/M criterion
        """
        def fct(dist):
            self.distance = dist
            dict = self.design_solver()
            return dict["pk_o_m"] - dict["pk_o_m_min"]

        dist_ini = self.distance * 2.
        output_dict = fsolve(fct, x0=dist_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.distance = output_dict[0][0]

        return self.design_solver()

    def compute_PKoM_on_grid(self, X, Y, **kwargs):
        """
        s = ["\nAirplane : %s" %self.mode,
             "Npax     %.1f" % self.n_pax,
             "Distance %d km" % unit.km_m(self.distance),
             "TAS      %d km/h" % unit.kmph_mps(self.vtas),
             "Altitude %d ft" % unit.ft_m(self.alt),
             "---------------------------------"]

        if self.design==None:
            s.append(">>  NO DESIGN  <<")
            return "\n".join(s) # print only recquirements

        s.append("Max power = %.0f kW" %unit.kW_W(self.design["pw_max"]))
        if self.design["airplane_mode"]=="classic":
            s.append("Mission fuel = %.0f kg" %self.design["mission_fuel"])
            s.append("Reserve fuel = %.0f kg" %self.design["reserve_fuel"])
            s.append("Total fuel = %.0f kg" %self.design["total_fuel"])
        elif self.design["airplane_mode"]=="electric":
            s.append("Cruise power = %.0f kW" %unit.kW_W(self.design["pw_cruise"]))
            s.append("Battery mass = %.0f kg" %self.design["battery_mass"])

        Compute the ratio between the Passenger.distance/MTOW (PKoM) and the minimum acceptable PKoM for a set of design range an number of passenger.
        :param X: 2D array of distance
        :param Y: 2D array of Npax
        :param kwargs: SmallPlane attribut that must be changed
        :return: 2D array of computed values
        """
        pkm=[]
        for key,val in kwargs.items(): # iterate over the kwargs list
            if key not in self.__dict__:
                raise KeyError('%s is not a SmallPlane attribut' %key)
            setattr(self,key,val) # change the attribut value. Raises a KeyError if invalid key is entered

        for x,y in zip(X.flatten(),Y.flatten()):
            self.distance = x
            self.n_pax = y
            self.design_solver()
            pkm.append(self.design["pk_o_m"]/self.design["pk_o_m_min"])
        # reshape pkm to 2D array
        pkm = np.array(pkm)
        return pkm.reshape(np.shape(X))

    def plot_PKoM(self,X, Y, PKM, cax=None):
        """Plot Passenger*distance/MTOW ratio versus Npax and Distance"""
        CS = ax.contourf(X / 1000, Y, PKM, levels=10)
        C = ax.contour(X / 1000, Y, PKM, levels=[1], colors=['red'], linewidths=2)
        ax.clabel(C, inline=True, fmt="%d")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("N passenger")
        ax.grid(True)
        if cax is None:
            return plt.colorbar(CS, label=r"PK/M")  # create colorbar
        else:
            plt.colorbar(CS, cax=CB.ax, label=r"PK/M")  # redraw colorbar on existing axes


if __name__ == '__main__':

    #-------------------------------------------------------------------------------------------------------------------
    # Validation examples

    spc = SmallPlane(npax=4.5, dist=unit.m_km(1300), tas=unit.mps_kmph(280),mode="classic")     # TB20
    spc.design_solver()
    print(spc)

    # spc.max_distance(mode="classic")
    # print("")
    # print("Max distance vs PK/M = ", "%.0f"%unit.km_m(spc.distance), " km")


    spe = SmallPlane(npax=2, dist=unit.m_km(130), tas=unit.mps_kmph(130),mode="electric")       # H55
    spe.battery_enrg_density = unit.J_Wh(200)
    spe.design_solver()
    print(spe)

    # spe.max_distance(mode="electric")
    # print("")
    # print("Max distance vs PK/M = ", "%.0f"%unit.km_m(spe.distance), " km")


    #-------------------------------------------------------------------------------------------------------------------
    # MAP display : minimal working example

    sp = SmallPlane(alt=unit.m_ft(3000),
                    tas=unit.mps_kmph(130),
                    mode="electric")

    sp.battery_enrg_density = unit.J_Wh(500)


    distances = np.linspace(50e3, 1200e3, 30)
    npaxs = np.arange(1, 20)
    X, Y = np.meshgrid(distances, npaxs)

    pkm = []
    for x,y in zip(X.flatten(),Y.flatten()):
        sp.distance = x
        sp.n_pax = y
        sp.new_design_solver()
        pkm.append(sp.design["pk_o_m"]/sp.design["pk_o_m_min"])

    # convert to numpy array with good shape
    pkm = np.array(pkm)
    pkm = pkm.reshape(np.shape(X))

    print("")
    # Plot contour
    cs = plt.contourf(X / 1000, Y, pkm, levels=20)

    c2c = plt.contour(X / 1000, Y, Y/X*1e3, levels=[0.0146], colors =['lightgrey'], linewidths=2)
    c2h = plt.contourf(X / 1000, Y, Y/X*1e3, levels=[0.0146,1], linewidths=2, colors='none', hatches=['\\'])
    for c in c2h.collections:
        c.set_edgecolor('lightgrey')

    # # plt.clabel(c1, inline=True, fmt="%d",fontsize=15)

    c1c = plt.contour(X / 1000, Y, pkm, levels=[1], colors =['yellow'], linewidths=2)
    c1h = plt.contourf(X / 1000, Y, pkm, levels=[0,1], colors='none', linewidths=2, hatches=['\\'])
    for c in c1h.collections:
        c.set_edgecolor('yellow')

    # plt.plot(X / 1000, Y, '+k')
    # plt.plot([0, distances[-1]*1e-3], [0, (19./1300.)*distances[-1]*1e-3], linestyle='solid', color='blue')

    plt.colorbar(cs, label=r"P.K/M")
    plt.grid(True)

    plt.suptitle("PK/M Field")
    plt.xlabel("Distance (km)")
    plt.ylabel("N passenger")

    plt.show()


    print("")
    print("Criteria = ", "%.3f" % (spe.design["pk_o_m"] / spc.design["pk_o_m"]))

    #----------------------------------------------- PLOT NPAX vs DISTANCE

    # Set the grid distance and npax
    distances = np.linspace(50e3, 500e3, 10)
    npaxs = np.arange(1, 20, 2)
    X, Y = np.meshgrid(distances, npaxs)

    # Initialize Plot
    fig,ax = plt.subplots(figsize=(10,7))
    sp = SmallPlane()
    pkm = sp.compute_PKoM_on_grid(X,Y)
    CB = sp.plot_PKoM(X,Y,pkm)
    plt.subplots_adjust(left=0.1,bottom=0.5,right=1.,top=0.98) # adjust position

    # ------------------------------------------------------- SLIDERS
    left = 0.25 # left starting point of the slider
    width = 0.4 # width of the slider
    space = 0.04 # vertical space between sliders
    height = 0.02 # height of a slider
    tas_ax = plt.axes([left, 8*space, width, height])    # Airspeed (km/h)
    tas_slider = Slider(
        ax=tas_ax,
        label="Speed",
        valmin=100,
        valmax=500,
        valfmt='%d km/h',
        valinit=150,
    )
    alt_ax = plt.axes([left, 7*space, width, height])    # Altitude (m)
    alt_slider = Slider(
        ax=alt_ax,
        label="Altitude",
        valmin=0,
        valmax=12e3,
        valfmt='%d m',
        valinit=unit.m_ft(3000),
    )
    lod_ax = plt.axes([left, 6*space, width, height])  # LIFT / DRAG
    lod_slider = Slider(
        ax=lod_ax,
        label="Lift/Drag",
        valmin=10,
        valmax=30,
        valfmt='%0.1f',
        valinit=14,
    )
    mef_ax = plt.axes([left, 5*space, width, height])  # Motor efficiency
    mef_slider = Slider(
        ax=mef_ax,
        label=r"$\eta_{motor}$",
        valmin=0,
        valmax=1.,
        valfmt='%0.2f',
        valinit=0.95,
    )
    pef_ax = plt.axes([left, 4*space, width, height])  # propeller efficiency
    pef_slider = Slider(
        ax=pef_ax,
        label=r"$\eta_{prop}$",
        valmin=0,
        valmax=1,
        valfmt='%0.2f',
        valinit=0.8,
    )
    mpd_ax = plt.axes([left, 3*space, width, height])  # Motor Power density (kW/kg)
    mpd_slider = Slider(
        ax=mpd_ax,
        label="Motor Power Dens.",
        valmin=0,
        valmax=10,
        valfmt='%0.1f kW/kg',
        valinit=4.5,
    )
    epd_ax = plt.axes([left, 2*space, width, height])  # Electric systems Power density (kW/kg)
    epd_slider = Slider(
        ax=epd_ax,
        label="Elec Power dens.",
        valmin=0,
        valmax=20,
        valfmt='%0.1f kW/kg',
        valinit=10,
    )
    bat_ax = plt.axes([left, space, width, height])  # Battery energetic density (Wh/kg)
    bat_slider = Slider(
        ax=bat_ax,
        label="Bat Enrg Dens.",
        valmin=0,
        valmax=800,
        valfmt='%d Wh/kg',
        valinit=200,
    )
    # ------------------------------------------------------- MODE SELECTOR
    mode_ax = plt.axes([left+width+0.15, 8*space+height-0.12, 0.12, 0.12])
    mode_rbutton = RadioButtons(
        ax = mode_ax,
        labels = ('classic','electric'),
        active=0 # default value is 'classic'
    )
    # ------------------------------------------------------- RESET BUTTON
    # Reset Button
    reset_ax = plt.axes([0.9, 0, 0.1, 0.05])
    reset_button = Button(reset_ax, 'Reset')

    # ------------------------------------------------------- CONNECT AND UPDATE (SLIDERS + BUTTONS)
    def update(val):
        """The function to be called anytime a slider's value changes"""
        # remove all previous contours
        ax.clear()
        CB.ax.clear()
        # Read sliders values
        tas = unit.mps_kmph(tas_slider.val) # m/s
        alt = alt_slider.val # m
        lod = lod_slider.val
        mef = mef_slider.val
        pef = pef_slider.val
        mpd = unit.W_kW(mpd_slider.val) # W/kg
        epd = unit.W_kW(epd_slider.val) # W/kg
        bat = unit.J_Wh(bat_slider.val) # J/kg
        mod = mode_rbutton.value_selected
        # Recompute and plot data
        pkm = sp.compute_PKoM_on_grid(X, Y, vtas=tas, alt=alt, lod=lod, motor_eff=mef, prop_eff=pef,
                      elec_motor_pw_density=mpd, power_elec_pw_density=epd, battery_enrg_density=bat, mode=mod)
        sp.plot_PKoM(X,Y,pkm,cax=CB.ax)


    # Connect Sliders and radio button
    sliders = [tas_slider, alt_slider,lod_slider,mef_slider,pef_slider,mpd_slider,epd_slider,bat_slider]
    for s in sliders:
        s.on_changed(update)
    mode_rbutton.on_clicked(update)

    # Connect reset button
    def reset(event):
        for s in sliders:
            s.reset()
        mode_rbutton.set_active(0)

    reset_button.on_clicked(reset)

    plt.show() # THE END











