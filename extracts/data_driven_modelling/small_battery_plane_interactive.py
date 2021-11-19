#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Department, ENAC
"""

import numpy as np
import unit
from physical_data import PhysicalData

from small_battery_plane import SmallPlane

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib import rc
font = {'size':12}
rc('font',**font)



def plot_PKoM(X, Y, PKM, cax=None):
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


#----------------------------------------------- PLOT NPAX vs DISTANCE
phd = PhysicalData()

# Set the grid distance and npax
distances = np.linspace(50e3, 500e3, 10)
npaxs = np.arange(2, 20, 2)
X, Y = np.meshgrid(distances, npaxs)

# Initialize Plot
fig,ax = plt.subplots(figsize=(10,7))
sp = SmallPlane(phd)
pkm = sp.compute_PKoM_on_grid(X,Y)
CB = plot_PKoM(X,Y,pkm)
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

# ------------------------------------------------------- SPAN SELECTOR
xmax_ax = plt.axes([0.8, 0.4, 0.1, 0.05])
xmax_box = TextBox(xmax_ax,'', initial=str(int(distances[-1]/1000)))
ymax_ax = plt.axes([0.01, 0.95, 0.05, 0.05])
ymax_box = TextBox(ymax_ax,'', initial=str(npaxs[-1]))

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
    mpd = unit.W_kW(mpd_slider.val) # ->W/kg
    epd = unit.W_kW(epd_slider.val) # ->W/kg
    bat = unit.J_Wh(bat_slider.val) # ->J/kg
    mod = mode_rbutton.value_selected
    # Recompute and plot data
    xmax = int(xmax_box.text) # max distance (km)
    ymax = int(ymax_box.text) # max number of passenger
    X,Y = np.meshgrid(np.linspace(1e3,xmax*1000,10),np.arange(2,ymax+2,2))
    pkm = sp.compute_PKoM_on_grid(X, Y, vtas=tas, altp=alt, lod=lod, elec_motor_efficiency=mef, prop_efficiency=pef,
                  elec_motor_pw_density=mpd, power_elec_pw_density=epd, battery_enrg_density=bat, mode=mod)
    plot_PKoM(X,Y,pkm,cax=CB.ax)


# Connect Sliders and radio button
sliders = [tas_slider, alt_slider,lod_slider,mef_slider,pef_slider,mpd_slider,epd_slider,bat_slider]
for s in sliders:
    s.on_changed(update)
mode_rbutton.on_clicked(update)
xmax_box.on_submit(update)
ymax_box.on_submit(update)

# Connect reset button
def reset(event):
    for s in sliders:
        s.reset()
    mode_rbutton.set_active(0)

reset_button.on_clicked(reset)

plt.show() # THE END



