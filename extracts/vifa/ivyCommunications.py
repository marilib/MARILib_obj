from ivy.std_api import IvySendMsg, IvyBindMsg, IvyMainLoop, IvyInit, IvyStart

import numpy

import data as geo

from tools import rad_deg
from geometry import geometry
from contour import contour
from force import force


xcg = 0.20
mass = 80000

vair = 150

psi = 0
theta = 0
phi = 0

alpha = rad_deg(4)
betha = rad_deg(6)

dl = rad_deg(-20)
dm = rad_deg(20)
dn = rad_deg(30)

trim = rad_deg(0)

a0 = -rad_deg(0)





geometry(a0,trim)

contour(dl,dm,dn)

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
    geometry(a0,trim)

    contour(dl,dm,dn)
    
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
    force(mass,xcg,vair,psi,theta,phi,alpha,betha,dl,dm,dn)
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
    IvySendMsg("Moment name=MTot normeX={0} normeY={1} normeZ={2}".format(*geo.MtotalXg))

    
    
def on_cnx(a,b):
    print("Initializing the bus....\n")
    
def on_die(a,b):
    print("Exiting the bus....\n")
 
    
if __name__=="__main__":


    agent_name = "pythonApp"
    ready_msg = "pythonApp ready!"
    
    IvyInit(agent_name, ready_msg, on_cnx, on_die)
    IvyStart()
    IvyBindMsg(onStartComputing, STARTCOMPUTING)
    IvyBindMsg(onStartGettingShape, STARTGETTINGSHAPES)
    IvyMainLoop()
    
    








