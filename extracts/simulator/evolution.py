#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""


import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import time

from marilib.utils import unit
from controler import MyJoystick


#===================================================================================================================
# Vector product of U by V
def cross_product(U, V):
    res = np.array([U[1]*V[2]-U[2]*V[1] ,
                    U[2]*V[0]-U[0]*V[2] ,
                    U[0]*V[1]-U[1]*V[0]])
    return res

def vector_product(U, V):
    res = U[0]*V[0] + U[1]*V[1] + U[2]*V[2]
    return res

def norm(V):
    n = np.sqrt(V[0]**2 + V[1]**2 + V[2]**2)
    return n

def renorm_vector(V):
    n = np.sqrt(V[0]**2 + V[1]**2 + V[2]**2)
    if (n>0.):
        N = V/n
    else:
        N = np.array([0. ,0. , 0.])
    return N

def renorm_frame(frame):
    frame[:,0] = renorm_vector(frame[:,0])
    frame[:,2] = cross_product(frame[:,0],frame[:,1])
    frame[:,1] = cross_product(frame[:,2],frame[:,0])

def Rx(phi):
    return np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])


def get_arrow(p,vec):
    return p[0], p[1], p[2], vec[0], vec[1], vec[2]


def get_controls():
    global ctrl
    dx,dl,dm,dn = ctrl.get_axis()
    return dx,dl,dm,dn


def get_state(time,dx,dl,dm,dn):
    global time_step, ux, uy, uz
    global g, m, loc, spd              # state vector

    lft = m*g * (1 - dm)        # lift force
    phi = -0.75 * np.pi * dl    # lift bank angle

    loc_d,spd_d = state_dot(loc,spd,lft,phi)

    loc = loc + loc_d * time_step   # Euler scheme on position
    spd = spd + spd_d * time_step   # Euler scheme on attitude

    x0 = renorm_vector(spd)          # X vector of mouving frame
    y0 = renorm_vector(cross_product(uz,x0))
    z0 = cross_product(x0,y0)
    Rzy = np.column_stack([x0,y0,z0])

    att = np.matmul(Rzy, Rx(phi))

    return loc,att,lft


def state_dot(loc,spd,lft,phi):
    global uz, g, m

    x0 = renorm_vector(spd)
    y0 = renorm_vector(cross_product(uz,x0))
    z0 = cross_product(x0,y0)

    lift = lft * (np.sin(phi)*y0 - np.cos(phi)*z0)
    weight = np.array([0,0,m*g])

    loc_d = spd
    spd_d = (lift+weight) / m

    return loc_d,spd_d


def update_frame(time):
    global quiver_x, quiver_y, quiver_z, lift_vec, mass_vec
    global g, m, loc, att
    global vec_length

    dx,dl,dm,dn = get_controls()

    loc,att,lft = get_state(time,dx,dl,dm,dn)

    quiver_x.remove()
    quiver_y.remove()
    quiver_z.remove()
    lift_vec.remove()
    mass_vec.remove()

    quiver_x = ax.quiver(*get_arrow(loc,att[:,0]*vec_length))
    quiver_y = ax.quiver(*get_arrow(loc,att[:,1]*vec_length))
    quiver_z = ax.quiver(*get_arrow(loc,att[:,2]*vec_length))
    lift_vec = ax.quiver(*get_arrow(loc,-att[:,2]*lft))
    mass_vec = ax.quiver(*get_arrow(loc,uz*m*g))


# Set the joystick
#---------------------------------------------------------------------------
ctrl = MyJoystick()


# Set simulation
#---------------------------------------------------------------------------
time_step = 0.010

# Set the scene
#---------------------------------------------------------------------------
fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

ax.set_xlim(-250, 250)
ax.set_ylim(-250, 250)
ax.set_zlim(0, 500)

ax.invert_yaxis()
ax.invert_zaxis()

vec_length = 50

ux = np.array([1,0,0])
uy = np.array([0,1,0])
uz = np.array([0,0,1])

g = 9.80665
m = 15
loc = np.array([-200,0,250])   # Position
spd = np.array([100,0,0])     # Speed

psi = unit.rad_deg(0)
theta = unit.rad_deg(0)
phi = unit.rad_deg(0)

att = np.matmul(np.matmul(Rz(psi), Ry(theta)), Rx(phi))

quiver_x = ax.quiver(*get_arrow(loc,att[:,0]*vec_length))
quiver_y = ax.quiver(*get_arrow(loc,att[:,1]*vec_length))
quiver_z = ax.quiver(*get_arrow(loc,att[:,2]*vec_length))
lift_vec = ax.quiver(*get_arrow(loc,-att[:,2]*m*g))
mass_vec = ax.quiver(*get_arrow(loc,uz*m*g))


# anim = FuncAnimation(fig, update_frame, frames=200, interval=20, repeat=True, blit=False)
anim = FuncAnimation(fig, update_frame, interval=time_step*1e3, repeat=True, blit=False)



plt.show()


