
from ivy.std_api import IvySendMsg, IvyBindMsg, IvyMainLoop, IvyInit, IvyStart

import numpy
import argparse
import platform
import sys

import data as geo

from tools import rad_deg
from geometry import geometry
from contour import contour
from force import force


DEFAULTBUS = {'Darwin': "224.255.255.255:2010",
              'Linux': "127.255.255.255:2010"}

g = 9.80665

xcg = 0.2555
mass = 70000

vair = 150

psi = rad_deg(0)
theta = rad_deg(0)
phi = rad_deg(0)

alpha = rad_deg(0)
betha = rad_deg(0)

dl = rad_deg(0)
dm = rad_deg(0)
dn = rad_deg(0)
dx = 0

p = rad_deg(0)
q = rad_deg(0)
r = rad_deg(0)

trim = rad_deg(0)

# a0 = rad_deg(3.031)
a0 = rad_deg(0)
f0 = mass*g/5

# NEW 2021
# Apply scaling factor of 1.50e-5 on all forces and 1.50e-6 on moments (MtotalXg)


geometry()

contour(a0,trim,dl,dm,dn)

#------------------REGEX---------------------------------

STARTGETTINGSHAPES = "^StartGettingShapes mass=(\S+) xcg=(\S+) vair=(\S+) psi=(\S+) theta=(\S+) phi=(\S+) alpha=(\S+) betha=(\S+) dl=(\S+) dm=(\S+) dn=(\S+)"
STARTCOMPUTING = "^StartComputation mass=(\S+) xcg=(\S+) vair=(\S+) psi=(\S+) theta=(\S+) phi=(\S+) alpha=(\S+) betha=(\S+) dl=(\S+) dm=(\S+) dn=(\S+)"



#------------------TOOLS---------------------------------


#------------------CALLBACK------------------------------

def sender(name, array):
    for point in array:
        IvySendMsg("ShapePoint name={0} ptX={1} ptY={2} ptZ={3}".format(name, *point))

def onStartGettingShape(client, *args):
    mass,xcg,vair,psi,theta,phi,alpha,betha,dl,dm,dn = [float(arg) for arg in args]
    IvySendMsg("ShapeStart name=fuselage")
    IvySendMsg("ShapeStart name=vtp")
    IvySendMsg("ShapeStart name=ruder")
    IvySendMsg("ShapeStart name=htpr")
    IvySendMsg("ShapeStart name=htpl")
    IvySendMsg("ShapeStart name=elevatorr")
    IvySendMsg("ShapeStart name=elevatorl")
    IvySendMsg("ShapeStart name=wingr")
    IvySendMsg("ShapeStart name=wingl")
    IvySendMsg("ShapeStart name=aileronr")
    IvySendMsg("ShapeStart name=aileronl")
    IvySendMsg("ShapeStart name=naceller")
    IvySendMsg("ShapeStart name=nacellel")
    geometry()

    contour(a0,trim,dl,dm,dn)
    
#// fuselage
    
    sender("fuselage", geo.FusXYZ)
#// vtp
    
    sender("vtp", geo.VtpXYZ)
# ruder
    
    sender("ruder", geo.RudXYZ)
#/ htp right and left
#right
    
    sender("htpr", geo.rHtpXYZ)
#left
    
    sender("htpl", geo.lHtpXYZ)
#// elevator right and left
#right
    
    sender("elevatorr", geo.rElevXYZ)
#left
    
    sender("elevatorl", geo.lElevXYZ)
#// wing left and right
#right
    
    sender("wingr", geo.rWingXYZ)
#left
    
    sender("wingl", geo.lWingXYZ)
# aileron left and right
#right
    
    sender("aileronr", geo.rAilXYZ)
#left
    		
    sender("aileronl", geo.lAilXYZ)
#// nacelle left and right
#right
    
    sender("naceller", geo.rNacXYZ)
    
#left
    		
    sender("nacellel", geo.lNacXYZ)
    
    
#No more shape, asks to draw
    IvySendMsg("Draw ffs")    
    
def onStartComputing(agent, *args):

    mass,xcg,vair,psi,theta,phi,alpha,betha,dl,dm,dn = [float(arg) for arg in args]
    kscale = 1.5e-5
    force(a0,f0,mass,xcg,vair,psi,theta,phi,alpha,betha,trim,dl,dm,dn,dx,p,q,r)
    # sending rWaf
    IvySendMsg("Force name=rWaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.WingRapp, *geo.rWafVec, "yellow"))
    # sending lWaf
    IvySendMsg("Force name=lWaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.WingLapp, *geo.lWafVec, "yellow"))  
    # sending rAaf
    IvySendMsg("Force name=rAaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.AilRapp, *geo.rAafVec, "red"))
    # sending lAaf
    IvySendMsg("Force name=lAaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.AilLapp, *geo.lAafVec, "red"))

    # sending rHaf
    IvySendMsg("Force name=rHaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.HtpRapp, *geo.rHafVec, "yellow"))
    # sending rEaf
    IvySendMsg("Force name=rEaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.HtpRapp, *geo.rEafVec, "red"))
    # sending lHaf
    IvySendMsg("Force name=lHaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.HtpLapp, *geo.lHafVec, "yellow"))
    # sending lEaf
    IvySendMsg("Force name=lEaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.HtpLapp, *geo.lEafVec, "red"))

    # sending Vaf
    IvySendMsg("Force name=Vaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.FusApp, *geo.VafVec, "yellow"))
    # sending Raf
    IvySendMsg("Force name=Raf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.VtpApp, *geo.RafVec, "red"))
    # sending Total Moment
    IvySendMsg("Moment name=MTot normeX={0} normeY={1} normeZ={2}".format(*geo.MtotalXg))

    # # TO BE ADDED
    # # sending rRaf
    # IvySendMsg("Force name=rEaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.RotRapp, *geo.rRafVec, "violet"))
    # # sending lRaf
    # IvySendMsg("Force name=lEaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.RotLapp, *geo.lRafVec, "violet"))
    # # sending rRaf
    # IvySendMsg("Force name=rEaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.HtpRapp, *geo.rHtpRafVec, "violet"))
    # # sending lRaf
    # IvySendMsg("Force name=lEaf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.HtpLapp, *geo.lHtpRafVec, "violet"))
    # # sending Raf
    # IvySendMsg("Force name=Raf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.VtpApp, *geo.rotVafVec, "violet"))
    #
    # # TO BE ADDED
    # # sending mg
    # IvySendMsg("Force name=Raf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.NacRapp, *geo.rNtfVec, "grey"))
    # # sending lift
    # IvySendMsg("Force name=Raf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.NacLapp, *geo.lNtfVec, "grey"))
    #
    # # TO BE ADDED
    # # sending mg
    # IvySendMsg("Force name=Raf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.mgApp, *geo.mg, "brown"))
    # # sending lift
    # IvySendMsg("Force name=Raf applicationX={0} applicationY={1} applicationZ={2} normeX={3} normeY={4} normeZ={5} color={6}".format(*geo.LiftApp, *geo.LiftTotal, "blue"))


def on_cnx(a,b):
    print("Initializing the bus....\n")
    
def on_die(a,b):
    print("Exiting the bus....\n")

def args(argv):
    """analyse les arguments en ligne de commande"""
    defaultbus = DEFAULTBUS.get(platform.system(), "127.255.255.255:2010")
    parser = argparse.ArgumentParser(description='ivy message viewer')
    parser.add_argument('-b', type=str, help='Ivy bus domain')
    args = parser.parse_args()
    bus = args.b if args.b else defaultbus
    return bus


if __name__=="__main__":

    my_bus = args(sys.argv)

    agent_name = "pythonApp"
    ready_msg = "pythonApp ready!"
    
    IvyInit(agent_name, ready_msg, on_cnx, on_die)

    IvyStart(my_bus)

    IvyBindMsg(onStartComputing, STARTCOMPUTING)
    IvyBindMsg(onStartGettingShape, STARTGETTINGSHAPES)
    IvyMainLoop()
    
    








