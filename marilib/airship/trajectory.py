#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 20 20:20:20 2020
@author: Thierry DRUOT
"""

import numpy as np
import numpy.linalg as lin
from scipy.integrate import solve_ivp

from tabulate import tabulate
import matplotlib.pyplot as plt



def s_min(min): return min*60.  # Translate minutes into seconds
def min_s(s): return s/60.      # Translate seconds into minutes

def s_h(h): return h*3600.   # Translate hours into seconds
def h_s(s): return s/3600.   # Translate seconds into hours

def rad_deg(deg): return deg*np.pi/180.   # Translate degrees into radians
def deg_rad(rad): return rad*180./np.pi   # Translate radians into degrees


def atmosphere(altp, disa=0.):
    """Ambiant data from pressure altitude from ground to 50 km according to Standard Atmosphere
    """
    g = 9.80665
    r = 287.053
    gam = 1.4

    Z = np.array([0., 11000., 20000., 32000., 47000., 50000.])
    dtodz = np.array([-0.0065, 0., 0.0010, 0.0028, 0.])
    P = np.array([101325., 0., 0., 0., 0., 0.])
    T = np.array([288.15, 0., 0., 0., 0., 0.])

    if (Z[-1] < altp):
        raise Exception("atmosphere, altitude cannot exceed 50km")

    j = 0
    while (Z[1+j] <= altp):
        T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
        if (0. < np.abs(dtodz[j])):
            P[j+1] = P[j]*(1. + (dtodz[j]/T[j]) * (Z[j+1]-Z[j]))**(-g/(r*dtodz[j]))
        else:
            P[j+1] = P[j]*np.exp(-(g/r)*((Z[j+1]-Z[j])/T[j]))
        j = j + 1

    if (0. < np.abs(dtodz[j])):
        pamb = P[j]*(1 + (dtodz[j]/T[j])*(altp-Z[j]))**(-g/(r*dtodz[j]))
    else:
        pamb = P[j]*np.exp(-(g/r)*((altp-Z[j])/T[j]))
    tamb = T[j] + dtodz[j]*(altp-Z[j]) + disa
    rho = pamb / ( r * tamb )
    return pamb, tamb, rho


def air(pos):
    """Simulation of atmospheric data, use interpolation into ERA5 in place
    """
    lng, lat, alt = pos
    pamb,tamb,rho = atmosphere(alt)     # WARNING : go around as no pressure and temperature are available
    wx,wy,wz = [10., 0., 0.]            # WARNING : Wind is fixed here : 10 m/s Eastward
    return pamb,tamb,rho,wx,wy,wz



class AirShip(object):

    def __init__(self, mass, width):

        self.mass = mass
        self.area_ref = 0.25*np.pi*width**2     # Width is taken as the mean diameter of the airship
        self.cx = 0.5                           # TODO : check this value

    def drag(self,rho,vgnd,wind):
        """Compute drag force on the airship
        """
        vair = vgnd - wind
        return -0.5*rho*self.area_ref*self.cx*lin.norm(vair)*vair

    def state_dot(self, t, state):
        """Compute the derivative of the state vector
        Note that all speed coordinates are in the local frame X : Eastward, Y : Northward, Z : Upward
        WARNING : in its present form, vertical acceleration is not correct as buoyancy is not considered
        """
        earth_radius = 6371229.            # From ERA5 doc relative to GRIB2 TODO : Check if GRIB2 are used

        lng,lat,alt,vx,vy,vz = state
        pamb,tamb,rho,wx,wy,wz = air([lng,lat,alt])
        vgnd = np.array([vx,vy,vz])
        wind = np.array([wx,wy,wz])
        drag = self.drag(rho,vgnd,wind)
        state_d = np.array([vx/earth_radius,
                            vy/earth_radius,
                            vz/earth_radius,
                            drag[0]/self.mass,
                            drag[1]/self.mass,
                            drag[2]/self.mass])
        return state_d

    def trajectory(self,to,t1,dt,pos,spd):
        """Compute the trajectory from given position and initial speed and over a given time frame
        Note that dt is only used to define a time step for trajectory description in the output
        """
        lng,lat,alt = pos   # WARNING lng and lat are in radians, alt is in m above earth radius
        vx, vy, vz = spd    # WARNING vx, vy and vz represent ground speed and are in m/s

        t_eval = np.linspace(t0,t1,int((t1-t0)/dt))
        state0 = np.array([lng,lat,alt,vx,vy,vz])

        sol = solve_ivp(self.state_dot, [t0,t1], state0, t_eval=t_eval, method="RK45")

        time = sol.t
        long = sol.y[0]
        latt = sol.y[1]
        altp = sol.y[2]

        return [time,long,latt,altp]



if __name__ == "__main__":

    mass = 100000.  # 100 tons
    width = 150.    # m

    # Instantiate the Airship
    ship = AirShip(mass, width)

    # Initial position
    pos = [rad_deg(43.),
           rad_deg(2.),
           3000.]

    # Initial speed
    spd = [0.,10.,0.]

    # Time frame and step
    t0 = 0.
    t1 = s_h(24.)
    dt = s_min(10.)

    # Compute the trajectory
    time,long,latt,altp = ship.trajectory(t0,t1,dt,pos,spd)

    # Format and print the numerical data
    table = []
    for i,t in enumerate(time):
        table.append([t,long[i],latt[i],altp[i]])
    print(tabulate(table))

    # Plot trajectory
    window_title = "Airship"
    plot_title = "Trajectory"

    fig,axes = plt.subplots(1,1)
    fig.canvas.set_window_title(window_title)
    fig.suptitle(plot_title, fontsize=14)

    plt.plot(deg_rad(long),deg_rad(latt),linewidth=2,color="blue")

    plt.grid(True)

    plt.ylabel('Lattitude (deg)')
    plt.xlabel('Longitude (deg)')

    plt.show()
