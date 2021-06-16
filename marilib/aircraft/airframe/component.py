#!/usr/bin/env python3
"""
Authors: *DRUOT Thierry, Nicolas Monrolin*

This module describes all the components that can be included in an airframe.
The :class:`Component` defines the common features of all components.

.. note:: All physical parameters are given in SI units.
"""

import numpy as np

from marilib.utils import earth, unit



class Component(object):
    """Define common features for all airplane components.
    Every component will override the methods defined in this abstract class.
    A component is meant to be contained in an instance of :class:`marilib.aircraft.airframe.airframe_root.Airframe`.

    **Attributs**
        * aircraft : the aircraft to which the component belongs. Needed for some pre-design methods (call to requirements) or multi-components interaction.
        * frame_origin : [x,y,z] origin of the *local* coordinate system inside the global aircraft coordinate system.
        * frame_angles : [psi,theta,phi] Euler angles to describe rotation of the local coordinate system.
        * mass : the net mass of the component
        * cg : [xg,yg,zg] the position of center of mass in **local** coordinates.
        * inertia_tensor : the inertia tensor of the component
        * gross_wet_area : wetted area of the component alone
        * net_wet_area : wetted area of the component in the assembly (without footprints)
        * aero_length : characteristic length of the component in the direction of the flow. Used for Reynolds number.
        * form_factor : factor on skin friction to account for lift independent pressure drag

    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.frame_origin = np.full(3,None)

        self.mass = None
        self.cg = np.full(3,None)
        self.inertia_tensor = np.full((3,3),None)

        self.gross_wet_area = 0.    # wetted area of the component alone
        self.net_wet_area = 0.      # wetted area of the component in the assembly (without footprints)
        self.aero_length = 1.       # characteristic length of the component in the direction of the flow
        self.form_factor = 0.       # factor on skin friction to account for lift independent pressure drag

    def get_component_type(self):
        raise NotImplementedError

    def eval_geometry(self):
        """Estimates the geometry of the component from the aircraft requirements and statistical design laws.
        """
        raise NotImplementedError

    def eval_mass(self):
        """Estimates the geometry of the component from the aircraft requirements and statistical design laws.
        """
        raise NotImplementedError

    def get_mass_mwe(self):
        """Returns the *Manufacturer's Weight Empty*, the mass (kg) of the component"""
        return self.mass

    def get_mass_owe(self):
        """Returns the *Operating Weight Empty*, the mass (kg) of the component. Can differ from MWE for some components"""
        return self.mass

    def get_cg_mwe(self):
        """Returns the position of the center of gravity of the manufacturer empty aircraft"""
        return self.cg

    def get_cg_owe(self):
        """Returns the position of the center of gravity of the operational empty aircraft"""
        return self.cg

    def get_inertia_tensor(self):
        """returns the inertia matrix of the component, if implemented."""
        return self.inertia_tensor

    def get_net_wet_area(self):
        """The net wet area of the component in contact with outer airflow, 0 if the component has no aerodynamic surface."""
        return self.net_wet_area

    def get_aero_length(self):
        """The characteristic aerodynamic length of the component (used for Reynolds number estimations for example)."""
        return self.aero_length

    def get_form_factor(self):
        """Form factor used to estimate form drag."""
        return self.form_factor

    def sketch_3view(self):
        return None

    def get_this_shape(self, name): # TODO: is the docstring up to date ?
        """Contour curves for 3 view drawing
        nose1 : modern nose (A220, A350, 787)
        nose3 : classical Airbus nose
        nose2 : symetrical nose
        cone1 : classical tail cone
        cone2 : symetrical cone
        section : circle
        """
        curve = {
        "nose1":np.array([[ 0.0000 , 0.3339 , 0.3339 , 0.0000 ,  0.0000 ] ,
                          [ 0.0050 , 0.3848 , 0.3084 , 0.0335 , -0.0335 ] ,
                          [ 0.0150 , 0.4253 , 0.2881 , 0.0652 , -0.0652 ] ,
                          [ 0.0500 , 0.5033 , 0.2490 , 0.1101 , -0.1101 ] ,
                          [ 0.1000 , 0.5811 , 0.2100 , 0.1585 , -0.1585 ] ,
                          [ 0.1800 , 0.6808 , 0.1600 , 0.2215 , -0.2215 ] ,
                          [ 0.2773 , 0.7704 , 0.1151 , 0.2859 , -0.2859 ] ,
                          [ 0.4191 , 0.8562 , 0.0721 , 0.3624 , -0.3624 ] ,
                          [ 0.5610 , 0.9198 , 0.0402 , 0.4211 , -0.4211 ] ,
                          [ 0.7738 , 0.9816 , 0.0092 , 0.4761 , -0.4761 ] ,
                          [ 0.9156 , 0.9962 , 0.0019 , 0.4976 , -0.4976 ] ,
                          [ 1.0000 , 1.0000 , 0.0000 , 0.5000 , -0.5000 ]]),

        "nose2":np.array([[ 0.0000 , 0.5000 ,  0.5000 , 0.0000 ,  0.0000 ] ,
                          [ 0.0050 , 0.5335 ,  0.4665 , 0.0335 , -0.0335 ] ,
                          [ 0.0191 , 0.5646 ,  0.4354 , 0.0646 , -0.0646 ] ,
                          [ 0.0624 , 0.6196 ,  0.3804 , 0.1196 , -0.1196 ] ,
                          [ 0.1355 , 0.6878 ,  0.3122 , 0.1878 , -0.1878 ] ,
                          [ 0.1922 , 0.7297 ,  0.2703 , 0.2297 , -0.2297 ] ,
                          [ 0.2773 , 0.7859 ,  0.2141 , 0.2859 , -0.2859 ] ,
                          [ 0.4191 , 0.8624 ,  0.1376 , 0.3624 , -0.3624 ] ,
                          [ 0.5610 , 0.9211 ,  0.0789 , 0.4211 , -0.4211 ] ,
                          [ 0.7738 , 0.9761 ,  0.0239 , 0.4761 , -0.4761 ] ,
                          [ 0.9156 , 0.9976 ,  0.0024 , 0.4976 , -0.4976 ] ,
                          [ 1.0000 , 1.0000 ,  0.0000 , 0.5000 , -0.5000 ]]),

        "nose3":np.array([[ 0.0000 , 0.4453 , 0.4453 , 0.0000 ,  0.0000 ] ,
                          [ 0.0050 , 0.4733 , 0.4112 , 0.0335 , -0.0335 ] ,
                          [ 0.0191 , 0.5098 , 0.3833 , 0.0646 , -0.0646 ] ,
                          [ 0.0624 , 0.5718 , 0.3188 , 0.1196 , -0.1196 ] ,
                          [ 0.1355 , 0.6278 , 0.2531 , 0.1878 , -0.1878 ] ,
                          [ 0.1922 , 0.7263 , 0.2142 , 0.2297 , -0.2297 ] ,
                          [ 0.2773 , 0.8127 , 0.1631 , 0.2859 , -0.2859 ] ,
                          [ 0.4191 , 0.8906 , 0.0962 , 0.3624 , -0.3624 ] ,
                          [ 0.5610 , 0.9392 , 0.0536 , 0.4211 , -0.4211 ] ,
                          [ 0.7738 , 0.9818 , 0.0122 , 0.4761 , -0.4761 ] ,
                          [ 0.9156 , 0.9976 , 0.0025 , 0.4976 , -0.4976 ] ,
                          [ 1.0000 , 1.0000 , 0.0000 , 0.5000 , -0.5000 ]]),

        "nose4":np.array([[ 0.0000, 0.3700, 0.3700, 0.0000,  0.0000 ] ,
                          [ 0.0050, 0.4200, 0.3200, 0.0335, -0.0335 ] ,
                          [ 0.0150, 0.4500, 0.2900, 0.0652, -0.0652 ] ,
                          [ 0.0500, 0.5000, 0.2400, 0.1200, -0.1200 ] ,
                          [ 0.1000, 0.5500, 0.1900, 0.1750, -0.1750 ] ,
                          [ 0.1800, 0.6000, 0.1400, 0.2400, -0.2400 ] ,
                          [ 0.2773, 0.6450, 0.1000, 0.3000, -0.3000 ] ,
                          [ 0.4191, 0.7000, 0.0500, 0.3700, -0.3700 ] ,
                          [ 0.5610, 0.7550, 0.0250, 0.4300, -0.4300 ] ,
                          [ 0.6200, 0.7800, 0.0180, 0.4500, -0.4500 ] ,
                          [ 0.7700, 0.9700, 0.0090, 0.4850, -0.4850 ] ,
                          [ 0.8100, 0.9900, 0.0060, 0.4900, -0.4900 ] ,
                          [ 0.9100, 0.9970, 0.0030, 0.4976, -0.4976 ] ,
                          [ 1.0000, 1.0000, 0.0000, 0.5000, -0.5000 ]]),

        "cone1":np.array([[ 0.0000 , 1.0000 , 0.0000 , 0.5000 , -0.5000 ] ,
                          [ 0.0213 , 1.0000 , 0.0082 , 0.5000 , -0.5000 ] ,
                          [ 0.0638 , 1.0000 , 0.0230 , 0.4956 , -0.4956 ] ,
                          [ 0.1064 , 1.0000 , 0.0393 , 0.4875 , -0.4875 ] ,
                          [ 0.1489 , 1.0000 , 0.0556 , 0.4794 , -0.4794 ] ,
                          [ 0.1915 , 1.0000 , 0.0786 , 0.4720 , -0.4720 ] ,
                          [ 0.2766 , 1.0000 , 0.1334 , 0.4566 , -0.4566 ] ,
                          [ 0.3617 , 1.0000 , 0.1964 , 0.4330 , -0.4330 ] ,
                          [ 0.4894 , 1.0000 , 0.3024 , 0.3822 , -0.3822 ] ,
                          [ 0.6170 , 1.0000 , 0.4159 , 0.3240 , -0.3240 ] ,
                          [ 0.7447 , 1.0000 , 0.5374 , 0.2577 , -0.2577 ] ,
                          [ 0.8723 , 1.0000 , 0.6627 , 0.1834 , -0.1834 ] ,
                          [ 0.8936 , 0.9963 , 0.6901 , 0.1679 , -0.1679 ] ,
                          [ 0.9149 , 0.9881 , 0.7139 , 0.1524 , -0.1524 ] ,
                          [ 0.9362 , 0.9800 , 0.7413 , 0.1333 , -0.1333 ] ,
                          [ 0.9574 , 0.9652 , 0.7687 , 0.1097 , -0.1097 ] ,
                          [ 0.9787 , 0.9533 , 0.8043 , 0.0788 , -0.0788 ] ,
                          [ 0.9894 , 0.9377 , 0.8280 , 0.0589 , -0.0589 ] ,
                          [ 1.0000 , 0.9103 , 0.8784 , 0.0162 , -0.0162 ]]),

        "cone2":np.array([[ 0.0000 , 1.0000 , 0.0000 , 0.5000 , -0.5000 ] ,
                          [ 0.0213 , 1.0000 , 0.0000 , 0.5000 , -0.5000 ] ,
                          [ 0.0638 , 0.9956 , 0.0044 , 0.4956 , -0.4956 ] ,
                          [ 0.1064 , 0.9875 , 0.0125 , 0.4875 , -0.4875 ] ,
                          [ 0.1489 , 0.9794 , 0.0206 , 0.4794 , -0.4794 ] ,
                          [ 0.1915 , 0.9720 , 0.0280 , 0.4720 , -0.4720 ] ,
                          [ 0.2766 , 0.9566 , 0.0434 , 0.4566 , -0.4566 ] ,
                          [ 0.3617 , 0.9330 , 0.0670 , 0.4330 , -0.4330 ] ,
                          [ 0.4894 , 0.8822 , 0.1178 , 0.3822 , -0.3822 ] ,
                          [ 0.6170 , 0.8240 , 0.1760 , 0.3240 , -0.3240 ] ,
                          [ 0.7447 , 0.7577 , 0.2423 , 0.2577 , -0.2577 ] ,
                          [ 0.8723 , 0.6834 , 0.3166 , 0.1834 , -0.1834 ] ,
                          [ 0.8936 , 0.6679 , 0.3321 , 0.1679 , -0.1679 ] ,
                          [ 0.9149 , 0.6524 , 0.3476 , 0.1524 , -0.1524 ] ,
                          [ 0.9362 , 0.6333 , 0.3667 , 0.1333 , -0.1333 ] ,
                          [ 0.9574 , 0.6097 , 0.3903 , 0.1097 , -0.1097 ] ,
                          [ 0.9787 , 0.5788 , 0.4212 , 0.0788 , -0.0788 ] ,
                          [ 0.9894 , 0.5589 , 0.4411 , 0.0589 , -0.0589 ] ,
                          [ 1.0000 , 0.5162 , 0.4838 , 0.0162 , -0.0162 ]]),

        "sec1":np.array([[  0.5000000 , 0.0000000 ,  0.0000000 ] ,
                         [  0.4903926 , 0.0975452 , -0.0975452 ] ,
                         [  0.4619398 , 0.1913417 , -0.1913417 ] ,
                         [  0.4157348 , 0.2777851 , -0.2777851 ] ,
                         [  0.3535534 , 0.3535534 , -0.3535534 ] ,
                         [  0.2777851 , 0.4157348 , -0.4157348 ] ,
                         [  0.1913417 , 0.4619398 , -0.4619398 ] ,
                         [  0.0975452 , 0.4903926 , -0.4903926 ] ,
                         [  0.0000000 , 0.5000000 , -0.5000000 ] ,
                         [- 0.0975452 , 0.4903926 , -0.4903926 ] ,
                         [- 0.1913417 , 0.4619398 , -0.4619398 ] ,
                         [- 0.2777851 , 0.4157348 , -0.4157348 ] ,
                         [- 0.3535534 , 0.3535534 , -0.3535534 ] ,
                         [- 0.4157348 , 0.2777851 , -0.2777851 ] ,
                         [- 0.4619398 , 0.1913417 , -0.1913417 ] ,
                         [- 0.4903926 , 0.0975452 , -0.0975452 ] ,
                         [- 0.5000000 , 0.0000000 ,  0.0000000 ]]),

        "sec2":np.array([[ 0.5000000 , 0.0000000 ,  0.0000000] ,
                         [ 0.4951963 , 0.0975452 , -0.0975452] ,
                         [ 0.4809699 , 0.1913417 , -0.1913417] ,
                         [ 0.4578674 , 0.2777851 , -0.2777851] ,
                         [ 0.4267767 , 0.3535534 , -0.3535534] ,
                         [ 0.3888926 , 0.4157348 , -0.4157348] ,
                         [ 0.3456709 , 0.4619398 , -0.4619398] ,
                         [ 0.2987726 , 0.4903926 , -0.4903926] ,
                         [ 0.2500000 , 0.5000000 , -0.5000000] ,
                         [ 0.0000000 , 0.5000000 , -0.5000000] ,
                         [-0.2500000 , 0.5000000 , -0.5000000] ,
                         [-0.2987726 , 0.4903926 , -0.4903926] ,
                         [-0.3456709 , 0.4619398 , -0.4619398] ,
                         [-0.3888926 , 0.4157348 , -0.4157348] ,
                         [-0.4267767 , 0.3535534 , -0.3535534] ,
                         [-0.4578674 , 0.2777851 , -0.2777851] ,
                         [-0.4809699 , 0.1913417 , -0.1913417] ,
                         [-0.4951963 , 0.0975452 , -0.0975452] ,
                         [-0.5000000 , 0.0000000 ,  0.0000000]])}

        return [curve[n] for n in name]


class Nacelle(Component):

    def __init__(self, aircraft):
        super(Nacelle, self).__init__(aircraft)

    def sketch_3view(self):
        nac_length = self.length
        nac_height = self.width
        nac_width =  self.width
        nac_x = self.frame_origin[0]
        nac_y = self.frame_origin[1]
        nac_z = self.frame_origin[2]

        section, = self.get_this_shape(["sec1"])

        nac_xz = np.array([[nac_x                , nac_z+0.4*nac_height ] ,
                           [nac_x+0.1*nac_length , nac_z+0.5*nac_height ] ,
                           [nac_x+0.5*nac_length , nac_z+0.5*nac_height ] ,
                           [nac_x+nac_length     , nac_z+0.3*nac_height ] ,
                           [nac_x+nac_length     , nac_z-0.3*nac_height ] ,
                           [nac_x+0.5*nac_length , nac_z-0.5*nac_height ] ,
                           [nac_x+0.1*nac_length , nac_z-0.5*nac_height ] ,
                           [nac_x                , nac_z-0.4*nac_height ] ,
                           [nac_x                , nac_z+0.4*nac_height ]])

        nac_xy = np.array([[nac_x                , nac_y+0.4*nac_width ] ,
                           [nac_x+0.1*nac_length , nac_y+0.5*nac_width ] ,
                           [nac_x+0.5*nac_length , nac_y+0.5*nac_width ] ,
                           [nac_x+nac_length     , nac_y+0.3*nac_width ] ,
                           [nac_x+nac_length     , nac_y-0.3*nac_width ] ,
                           [nac_x+0.5*nac_length , nac_y-0.5*nac_width ] ,
                           [nac_x+0.1*nac_length , nac_y-0.5*nac_width ] ,
                           [nac_x                , nac_y-0.4*nac_width ] ,
                           [nac_x                , nac_y+0.4*nac_width ]])

        d_nac_yz = np.stack([section[0:,0]*nac_width , section[0:,1]*nac_height , section[0:,2]*nac_height], axis=1)

        d_fan_yz = np.stack([section[0:,0]*0.80*nac_width , section[0:,1]*0.80*nac_height , section[0:,2]*0.80*nac_height], axis=1)

        nac_yz = np.vstack([np.stack([nac_y+d_nac_yz[0:,0] , nac_z+d_nac_yz[0:,1]],axis=1) ,
                            np.stack([nac_y+d_nac_yz[::-1,0] , nac_z+d_nac_yz[::-1,2]],axis=1)])

        nac_le = np.stack([[nac_x]*len(nac_yz) , nac_yz[:,0] , nac_yz[:,1]], axis=1)

        nac_te = np.stack([[nac_x+nac_length]*len(nac_yz) , nac_yz[:,0] , nac_yz[:,1]], axis=1)

        nac_toc = 0.15

        if hasattr(self, "propeller_width"):
            prop_width = self.propeller_width
            d_prop_yz = np.stack([section[0:,0]*prop_width , section[0:,1]*prop_width , section[0:,2]*prop_width], axis=1)
            disk_yz = np.vstack([np.stack([nac_y+d_prop_yz[0:,0] , nac_z+d_prop_yz[0:,1]],axis=1) ,
                                 np.stack([nac_y+d_prop_yz[::-1,0] , nac_z+d_prop_yz[::-1,2]],axis=1)])
        else:
            disk_yz = np.vstack([np.stack([nac_y+d_fan_yz[0:,0] , nac_z+d_fan_yz[0:,1]],axis=1) ,
                                 np.stack([nac_y+d_fan_yz[::-1,0] , nac_z+d_fan_yz[::-1,2]],axis=1)])

        return {"xy":nac_xy , "yz":nac_yz, "xz":nac_xz, "disk":disk_yz, "le":nac_le, "te":nac_te, "toc":nac_toc}

class Tank(Component):

    def __init__(self, aircraft):
        super(Tank, self).__init__(aircraft)
        super(Tank, self).__init__(aircraft)

    def get_mfw(self):
        return self.mfw_volume_limited

    def performance_ratio(self, aircraft):
        if aircraft.arrangement.fuel_type=="liquid_h2": return unit.Pam3pkg_barLpkg(250.)
        elif aircraft.arrangement.fuel_type=="compressed_h2": return unit.Pam3pkg_barLpkg(700.)
        else: return unit.Pam3pkg_barLpkg(250.)

    def fuel_over_pressure(self, aircraft):
        if aircraft.arrangement.fuel_type=="liquid_h2": return unit.Pa_bar(10.)
        elif aircraft.arrangement.fuel_type=="compressed_h2": return unit.Pa_bar(700.)
        else: return 0.

class Pod(Component):

    def __init__(self, aircraft):
        super(Pod, self).__init__(aircraft)

        class_name = "Pod"

        self.structure_shell_thickness = aircraft.get_init(class_name,"structure_shell_thickness")
        self.structure_shell_surface_mass = aircraft.get_init(class_name,"structure_shell_surface_mass")

        self.gravimetric_index = aircraft.get_init(class_name,"gravimetric_index")
        self.volumetric_index = aircraft.get_init(class_name,"volumetric_index")

        self.dry_bay_length = None

        self.external_pod_area = None
        self.external_pod_volume = None

        self.structure_internal_volume = None
        self.structure_shell_volume = None
        self.structure_shell_mass = None

        self.shell_specific_volume = None
        self.fuel_volume = None

        self.shell_specific_mass = None
        self.fuel_mass = None

    def size_fuel_tank(self,location):
        # Tank is supposed to be composed of a cylindrical part ended with two emisphers
        # An unusable length equal to one diameter is taken for tapered ends
        self.external_pod_area = np.pi*self.width**2 + (self.length-2.*self.width-self.dry_bay_length) * (np.pi*self.width)
        self.external_pod_volume = (1./6.)*np.pi*self.width**3 + (self.length-2.*self.width-self.dry_bay_length) * (0.25*np.pi*self.width**2)

        if location=="external":
            self.structure_internal_volume = (1./6.)*np.pi*(self.width-2.*self.structure_shell_thickness)**3 + (self.length-2.*self.width-self.dry_bay_length) * (0.25*np.pi*(self.width-2.*self.structure_shell_thickness)**2)
            self.structure_shell_volume = self.external_pod_volume - self.structure_internal_volume
        elif location=="internal":
            self.structure_shell_surface_mass = 0.  # kg/m2
            self.structure_shell_thickness = 0.     # m
            self.structure_internal_volume = self.external_pod_volume
            self.structure_shell_volume = 0.
        else:
            raise Exception("Tank location is unknown")

        self.shell_specific_volume = self.structure_internal_volume*(1.-self.volumetric_index)
        self.fuel_volume = self.structure_internal_volume*self.volumetric_index

        return

    def mass_fuel_tank(self,location):
        if location=="external":
            self.structure_shell_mass = self.external_pod_area * self.structure_shell_surface_mass
        elif location=="internal":
            self.structure_shell_mass = 0.
        else:
            raise Exception("Tank location is unknown")

        # Tank equiped structural mass
        if self.aircraft.arrangement.fuel_type in ["liquid_h2","compressed_h2"]:
            self.shell_specific_mass = self.structure_internal_volume*(1./self.gravimetric_index-1.)*self.volumetric_index*self.fuel_density
        else:
            self.shell_specific_mass = 0.

        self.fuel_mass = self.fuel_density * self.fuel_volume

        return

    def get_mfw(self):
        return self.mfw_volume_limited

    def performance_ratio(self, aircraft):
        if aircraft.arrangement.fuel_type=="liquid_h2": return unit.Pam3pkg_barLpkg(250.)
        elif aircraft.arrangement.fuel_type=="compressed_h2": return unit.Pam3pkg_barLpkg(700.)
        else: return unit.Pam3pkg_barLpkg(250.)

    def fuel_over_pressure(self, aircraft):
        if aircraft.arrangement.fuel_type=="liquid_h2": return unit.Pa_bar(10.)
        elif aircraft.arrangement.fuel_type=="compressed_h2": return unit.Pa_bar(700.)
        else: return 0.

    def sketch_3view(self):
        pod_width = self.width
        pod_length = self.length
        pod_x_axe = self.frame_origin[0]
        pod_y_axe = self.frame_origin[1]
        pod_z_axe = self.frame_origin[2]
        wing_x_body = self.wing_axe_x
        wing_z_body = self.wing_axe_z
        wing_c_body = self.wing_axe_c

        nose2,cone2,section = self.get_this_shape(["nose2","cone2","sec1"])

        r_nose = 0.15       # Fuselage length ratio of nose evolutive part
        r_cone = 0.35       # Fuselage length ratio of tail cone evolutive part

        pod_cyl_yz = np.stack([pod_y_axe + section[0:,0]*pod_width , pod_z_axe + section[0:,1]*pod_width , pod_z_axe + section[0:,2]*pod_width], axis=1)

        pod_front = np.vstack([np.stack([pod_cyl_yz[0:,0] , pod_cyl_yz[0:,1]],axis=1) , np.stack([pod_cyl_yz[::-1,0] , pod_cyl_yz[::-1,2]],axis=1)])

        pod_nose_xz = np.stack([pod_x_axe + nose2[0:,0]*pod_length*r_nose , pod_z_axe - 0.5*pod_width + nose2[0:,1]*pod_width , pod_z_axe - 0.5*pod_width + nose2[0:,2]*pod_width], axis=1)
        pod_cone_xz = np.stack([pod_x_axe + (1-r_cone)*pod_length + cone2[0:,0]*pod_length*r_cone , pod_z_axe - 0.5*pod_width + cone2[0:,1]*pod_width , pod_z_axe - 0.5*pod_width + cone2[0:,2]*pod_width], axis=1)
        pod_xz = np.vstack([pod_nose_xz , pod_cone_xz])

        pod_side = np.vstack([np.stack([pod_xz[0:-2,0] , pod_xz[0:-2,1]],axis=1) , np.stack([pod_xz[:0:-1,0] , pod_xz[:0:-1,2]],axis=1)])

        pod_nose_xy = np.stack([pod_x_axe + nose2[0:,0]*pod_length*r_nose , pod_y_axe + nose2[0:,3]*pod_width , pod_y_axe + nose2[0:,4]*pod_width], axis=1)
        pod_cone_xy = np.stack([pod_x_axe + (1-r_cone)*pod_length + cone2[0:,0]*pod_length*r_cone , pod_y_axe + cone2[0:,3]*pod_width , pod_y_axe + cone2[0:,4]*pod_width], axis=1)
        pod_xy = np.vstack([pod_nose_xy , pod_cone_xy])

        pod_top = np.vstack([np.stack([pod_xy[1:-2,0]  , pod_xy[1:-2,1]],axis=1) , np.stack([pod_xy[:0:-1,0] , pod_xy[:0:-1,2]],axis=1)])

        return {"xy":pod_top , "yz":pod_front, "xz":pod_side, "body_xz":pod_xz, "body_xy":pod_xy}


class Cabin(Component):
    """The Cabin includes the passenger seats, the crew space and furnishings."""

    def __init__(self, aircraft):
        super(Cabin, self).__init__(aircraft)

        self.n_pax_ref = self.aircraft.requirement.n_pax_ref
        self.n_pax_front = aircraft.get_init(self,"n_pax_front", val=self.__n_pax_front())
        self.n_aisle = aircraft.get_init(self,"n_aisle", val=self.__n_aisle())
        self.seat_width = aircraft.get_init(self,"seat_width")
        self.seat_pitch = aircraft.get_init(self,"seat_pitch")
        self.aisle_width = aircraft.get_init(self,"aisle_width")

        self.width = None
        self.length = None
        self.co2_metric_area = None

        self.m_pax_nominal = aircraft.get_init(self,"m_pax_nominal", val=self.__m_pax_nominal())
        self.m_pax_max = aircraft.get_init(self,"m_pax_max", val=self.__m_pax_max())
        self.m_pax_cabin = aircraft.get_init(self,"m_pax_cabin")
        self.m_furnishing = None
        self.m_op_item = None
        self.nominal_payload = None
        self.maximum_payload = None

        self.cg_furnishing = np.full(3,None)
        self.cg_op_item = np.full(3,None)

        self.pax_max_fwd_cg = np.full(3,None)
        self.pax_max_fwd_mass = None

        self.pax_max_bwd_cg = np.full(3,None)
        self.pax_max_bwd_mass = None

    def __n_pax_front(self):
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        if  (n_pax_ref<=12):   n_pax_front = 2
        elif(n_pax_ref<=24):  n_pax_front = 3
        elif(n_pax_ref<=70):  n_pax_front = 4
        elif(n_pax_ref<=120): n_pax_front = 5
        elif(n_pax_ref<=225): n_pax_front = 6
        elif(n_pax_ref<=300): n_pax_front = 8
        elif(n_pax_ref<=375): n_pax_front = 9
        else:                      n_pax_front = 10
        return n_pax_front

    def __n_aisle(self):
        if(self.n_pax_front <= 6): n_aisle = 1
        else:                      n_aisle = 2
        return n_aisle

    def __m_pax_nominal(self):
        design_range = self.aircraft.requirement.design_range
        if  (design_range <= unit.m_NM(500.)): m_pax_nominal = 105.
        elif(design_range <= unit.m_NM(1500.)): m_pax_nominal = 105.
        elif(design_range <= unit.m_NM(3500.)): m_pax_nominal = 105.
        elif(design_range <= unit.m_NM(5500.)): m_pax_nominal = 105.
        else: m_pax_nominal = 110.
        return m_pax_nominal

    def __m_pax_max(self):
        if(self.aircraft.requirement.design_range <= unit.m_NM(500.)): m_pax_max = 105.
        elif(self.aircraft.requirement.design_range <= unit.m_NM(1500.)): m_pax_max = 110.
        elif(self.aircraft.requirement.design_range <= unit.m_NM(3500.)): m_pax_max = 120.
        elif(self.aircraft.requirement.design_range <= unit.m_NM(5500.)): m_pax_max = 135.
        else: m_pax_max = 150.
        return m_pax_max

    def get_component_type(self):
        return "cabin"

    def eval_geometry(self):
        design_range = self.aircraft.requirement.design_range
        self.width =  self.seat_width*self.n_pax_front + self.aisle_width*self.n_aisle + 0.10   # Statistical regression
        self.length =  2.0*self.width + self.seat_pitch*self.n_pax_ref/self.n_pax_front \
                     + 2.0e-9*self.n_pax_ref*design_range    # Statistical regression
        self.projected_area = 0.95*self.length*self.width       # Factor 0.95 accounts for tapered parts

    def eval_mass(self):
        design_range = self.aircraft.requirement.design_range
        cabin_frame_origin = self.aircraft.airframe.cabin.frame_origin

        self.m_furnishing = (0.063*self.n_pax_ref**2 + 9.76*self.n_pax_ref)       # Furnishings mass
        self.m_op_item = max(160., 5.2*(self.n_pax_ref*design_range*1e-6))        # Operator items mass

        self.nominal_payload = self.n_pax_ref * self.m_pax_nominal
        self.maximum_payload = self.n_pax_ref * self.m_pax_max

        fwd_cabin_vec = np.array([self.aircraft.airframe.wing.mac_loc[0], 0., 0.]) + np.array([0.25*self.aircraft.airframe.wing.mac, 0., 0.]) - cabin_frame_origin
        bwd_cabin_vec = cabin_frame_origin + np.array([self.length, 0., 0.]) - fwd_cabin_vec

        self.pax_max_fwd_cg = cabin_frame_origin + 0.50*fwd_cabin_vec                   # Payload max forward CG
        self.pax_max_fwd_mass = self.n_pax_ref*self.m_pax_cabin * fwd_cabin_vec[0]/self.length   # Payload mass for max forward CG

        self.pax_max_bwd_cg = cabin_frame_origin + fwd_cabin_vec + 0.50*bwd_cabin_vec   # Payload max backward CG
        self.pax_max_bwd_mass = self.n_pax_ref*self.m_pax_cabin * bwd_cabin_vec[0]/self.length   # Payload mass for max backward CG

        x_cg_furnishing = self.frame_origin[0] + 0.55*self.length      # Rear cabin is heavier because of higher density
        x_cg_op_item = x_cg_furnishing    # Operator items cg

        self.cg_furnishing = np.array([x_cg_furnishing, 0., 0.])
        self.cg_op_item = np.array([x_cg_op_item, 0., 0.])

        self.mass = self.m_furnishing + self.m_op_item
        self.cg = (self.cg_furnishing*self.m_furnishing + self.cg_op_item*self.m_op_item) / self.mass

    def get_mass_mwe(self):
        return self.m_furnishing

    def get_cg_mwe(self):
        return self.cg_furnishing

class Cargo(Component):
    """The Cargo defines the space where luggage and other payload can be stored."""

    def __init__(self, aircraft):
        super(Cargo, self).__init__(aircraft)

        self.container_pallet_mass = None

        self.freight_max_fwd_cg = np.full(3,None)
        self.freight_max_fwd_mass = None

        self.freight_max_bwd_cg = np.full(3,None)
        self.freight_max_bwd_mass = None

    def get_component_type(self):
        return "cargo_hold"

    def eval_geometry(self):
        self.frame_origin = self.aircraft.airframe.cabin.frame_origin

    def eval_mass(self):
        n_pax_ref = self.aircraft.airframe.cabin.n_pax_ref
        n_pax_front = self.aircraft.airframe.cabin.n_pax_front
        m_pax_max = self.aircraft.airframe.cabin.m_pax_max
        m_pax_cabin = self.aircraft.airframe.cabin.m_pax_cabin
        body_width = self.aircraft.airframe.body.width
        body_length = self.aircraft.airframe.body.length
        wing_root_loc = self.aircraft.airframe.wing.root_loc
        wing_root_c = self.aircraft.airframe.wing.root_c
        cabin_length = self.aircraft.airframe.cabin.length

        cargo_frame_origin = self.aircraft.airframe.cargo.frame_origin

        if (n_pax_front>=6):
            forward_hold_length = wing_root_loc[0] - self.frame_origin[0]
            backward_hold_length = self.frame_origin[0]+cabin_length - (wing_root_loc[0]+wing_root_c)
            hold_length = forward_hold_length + backward_hold_length

            self.container_pallet_mass = 4.36 * body_width * body_length        # Container and pallet mass
            self.mass = self.container_pallet_mass

            fwd_hold_vec = np.array([self.aircraft.airframe.wing.mac_loc[0], 0., 0.]) + np.array([0.25*self.aircraft.airframe.wing.mac, 0., 0.]) - cargo_frame_origin
            bwd_hold_vec = cargo_frame_origin + np.array([hold_length, 0., 0.]) - fwd_hold_vec

            self.freight_max_fwd_cg = cargo_frame_origin + 0.50*fwd_hold_vec                   # Payload max forward CG
            self.freight_max_fwd_mass = n_pax_ref*(m_pax_max-m_pax_cabin) * fwd_hold_vec[0]/hold_length   # Payload mass for max forward CG

            self.freight_max_bwd_cg = cargo_frame_origin + fwd_hold_vec + 0.50*bwd_hold_vec   # Payload max backward CG
            self.freight_max_bwd_mass = n_pax_ref*(m_pax_max-m_pax_cabin) * bwd_hold_vec[0]/hold_length   # Payload mass for max backward CG

            self.cg =   (self.freight_max_fwd_cg*self.freight_max_fwd_mass + self.freight_max_bwd_cg*self.freight_max_bwd_mass) \
                      / (self.freight_max_fwd_mass + self.freight_max_bwd_mass)

        else:
            self.freight_max_fwd_cg = np.array([0., 0., 0.])
            self.freight_max_fwd_mass = 0.

            self.freight_max_bwd_cg = np.array([0., 0., 0.])
            self.freight_max_bwd_mass = 0.

            self.mass = 0.
            self.cg = np.array([0., 0., 0.])

    def get_mass_mwe(self):
        return 0.

    def get_cg_mwe(self):
        return np.array([0., 0., 0.])

class Fuselage(Component):
    """The skin of the aircraft body (tube and wing configuration)
    """
    def __init__(self, aircraft):
        super(Fuselage, self).__init__(aircraft)

        self.forward_ratio = aircraft.get_init(self,"forward_ratio")
        self.wall_thickness = aircraft.get_init(self,"wall_thickness")
        self.rear_bulkhead_ratio = aircraft.get_init(self,"rear_bulkhead_ratio")
        self.nose_cone_ratio = aircraft.get_init(self,"nose_cone_ratio")
        self.tail_cone_ratio = aircraft.get_init(self,"tail_cone_ratio")
        self.section_type = aircraft.get_init(self,"section_type")

        self.width = None
        self.height = None
        self.length = None
        self.nose_cone_length = None
        self.tail_cone_length = None

        self.mass_correction_factor = aircraft.get_init(self,"mass_correction_factor")

    def get_component_type(self):
        return "body"

    def eval_geometry(self):
        self.frame_origin = [0., 0., 0.]

        cabin_width = self.aircraft.airframe.cabin.width
        cabin_length = self.aircraft.airframe.cabin.length

        self.width = cabin_width + self.wall_thickness      # fuselage walls are supposed 0.2m thick
        self.height = 1.05 * self.width

        self.aircraft.airframe.cabin.frame_origin = [self.forward_ratio*self.width, 0., 0.]     # cabin position inside the fuselage
        self.aircraft.airframe.cargo.frame_origin = [self.forward_ratio*self.width, 0., 0.]     # cabin position inside the fuselage

        if self.aircraft.arrangement.tank_architecture=="rear":
            self.length = cabin_length + self.aircraft.airframe.tank.length + (self.forward_ratio + self.rear_bulkhead_ratio)*self.width
        else:
            self.length = cabin_length + (self.forward_ratio + self.rear_bulkhead_ratio)*self.width

        self.nose_cone_length = self.nose_cone_ratio*self.width
        self.tail_cone_length = self.tail_cone_ratio*self.width

        self.gross_wet_area = 2.70*self.length*np.sqrt(self.width*self.height)
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.length
        self.form_factor = 1.05

    def eval_mass(self):
        kfus = np.pi*self.length*np.sqrt(self.width*self.height)
        self.mass = self.mass_correction_factor * 5.40*kfus**1.19      # Statistical regression versus fuselage built surface
        self.cg = np.array([0.50*self.length, 0., 0.40*self.height])     # Middle of the fuselage

    def sketch_3view(self):
        body_width = self.width
        body_height = self.height
        body_length = self.length

        if self.section_type=="ellipse":
            nose,cone,section = self.get_this_shape(["nose1","cone1","sec1"])
        elif self.section_type=="square":
            nose,cone,section = self.get_this_shape(["nose4","cone1","sec2"])
        else:
            raise Exception("Section type is unknown")

        r_nose = self.nose_cone_length / self.length    # Fuselage length ratio of nose evolutive part
        r_cone = self.tail_cone_length / self.length    # Fuselage length ratio of tail cone evolutive part

        cyl_yz = np.stack([section[0:,0]*body_width , section[0:,1]*body_height+0.5*body_height , section[0:,2]*body_height+0.5*body_height], axis=1)

        body_front = np.vstack([np.stack([cyl_yz[0:,0] , cyl_yz[0:,1]],axis=1) , np.stack([cyl_yz[::-1,0] , cyl_yz[::-1,2]],axis=1)])

        nose_xz = np.stack([nose[0:,0]*body_length*r_nose , nose[0:,1]*body_height , nose[0:,2]*body_height], axis=1)
        cone_xz = np.stack([(1-r_cone)*body_length + cone[0:,0]*body_length*r_cone , cone[0:,1]*body_height , cone[0:,2]*body_height], axis=1)
        body_xz = np.vstack([nose_xz , cone_xz])

        body_side = np.vstack([np.stack([body_xz[0:-2,0] , body_xz[0:-2,1]],axis=1) , np.stack([body_xz[:0:-1,0] , body_xz[:0:-1,2]],axis=1)])

        nose_xy = np.stack([nose[0:,0]*body_length*r_nose , nose[0:,3]*body_width , nose[0:,4]*body_width], axis=1)
        cone_xy = np.stack([(1-r_cone)*body_length + cone[0:,0]*body_length*r_cone , cone[0:,3]*body_width , cone[0:,4]*body_width], axis=1)
        body_xy = np.vstack([nose_xy , cone_xy])

        body_top = np.vstack([np.stack([body_xy[1:-2,0]  , body_xy[1:-2,1]],axis=1) , np.stack([body_xy[:0:-1,0] , body_xy[:0:-1,2]],axis=1)])

        return {"xy":body_top , "yz":body_front, "xz":body_side, "body_xz":body_xz, "body_xy":body_xy}


class Wing(Component):
    """"""
    def __init__(self, aircraft):
        super(Wing, self).__init__(aircraft)

        design_range = self.aircraft.requirement.design_range
        n_pax_ref = self.aircraft.requirement.n_pax_ref

        n_pax_front = self.aircraft.airframe.cabin.n_pax_front
        n_aisle = self.aircraft.airframe.cabin.n_aisle

        self.wing_morphing = aircraft.get_init(self,"wing_morphing")   # "aspect_ratio_driven" or "span_driven"
        self.area = 60. + 88.*n_pax_ref*design_range*1.e-9
        self.span = None
        self.aspect_ratio = aircraft.get_init(self,"aspect_ratio", val=self.aspect_ratio())
        self.taper_ratio = aircraft.get_init(self,"taper_ratio", val=self.taper_ratio())
        self.dihedral = aircraft.get_init(self,"dihedral")
        self.sweep0 = None
        self.sweep25 = aircraft.get_init(self,"sweep25", val=self.sweep25())
        self.sweep100 = None
        self.setting = None
        self.hld_type = aircraft.get_init(self,"hld_type", val=self.high_lift_type())
        self.front_spar_ratio = aircraft.get_init(self,"front_spar_ratio")
        self.rear_spar_ratio = aircraft.get_init(self,"rear_spar_ratio")

        self.x_rout = None      # Design variable for hq_optim

        self.root_loc = np.full(3,None)     # Position of root chord leading edge
        self.root_toc = None                # thickness over chord ratio of root chord
        self.root_c = None                  # root chord length

        self.kink_loc =  np.full(3,None)    # Position of kink chord leading edge
        self.kink_toc = None                # thickness over chord ratio of kink chord
        self.kink_c = None                  # kink chord length

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_toc = None                 # thickness over chord ratio of tip chord
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None
        
        self.mass_correction_factor = aircraft.get_init(self,"mass_correction_factor")

    def get_component_type(self):
        return "wing"

    def aspect_ratio(self):
        if (self.aircraft.arrangement.power_architecture in ["tp","ep"]):
            ar = 10
        else:
            ar = 9
        return ar

    def taper_ratio(self):
        taper_ratio = 0.20
        return taper_ratio

    def sweep25(self):
        sweep25 = 1.6*max(0.,(self.aircraft.requirement.cruise_mach - 0.5))     # Empirical law
        return sweep25

    def high_lift_type(self):
        if (self.aircraft.arrangement.power_architecture in ["tp","ep"]):
            hld_type = 2
        else:
            hld_type = 9
        return hld_type

    def eval_geometry(self, hq_optim=False):
        wing_attachment = self.aircraft.arrangement.wing_attachment
        cruise_mach = self.aircraft.requirement.cruise_mach
        body_width = self.aircraft.airframe.body.width
        body_length = self.aircraft.airframe.body.length
        body_height = self.aircraft.airframe.body.height

        self.tip_toc = 0.10
        self.kink_toc = self.tip_toc + 0.01
        self.root_toc = self.kink_toc + 0.03

        if(self.wing_morphing=="aspect_ratio_driven"):   # Aspect ratio is driving parameter
            self.span = np.sqrt(self.aspect_ratio*self.area)
        elif(self.wing_morphing=="span_driven"): # Span is driving parameter
            self.aspect_ratio = self.span**2/self.area
        else:
            print("geometry_predesign_, wing_wing_morphing index is unkown")

        y_root = 0.5*body_width
        y_kink = 1.5*body_width
        y_tip = 0.5*self.span

        if(15< unit.deg_rad(self.sweep25)):  # With kink
          Phi100intTE = max(0., 2. * (self.sweep25 - unit.rad_deg(32.)))
          tan_phi100 = np.tan(Phi100intTE)
          A = ((1-0.25*self.taper_ratio)*y_kink+0.25*self.taper_ratio*y_root-y_tip) / (0.75*y_kink+0.25*y_root-y_tip)
          B = (np.tan(self.sweep25)-tan_phi100) * ((y_tip-y_kink)*(y_kink-y_root)) / (0.25*y_root+0.75*y_kink-y_tip)
          self.root_c = (self.area-B*(y_tip-y_root)) / (y_root+y_kink+A*(y_tip-y_root)+self.taper_ratio*(y_tip-y_kink))
          self.kink_c = A*self.root_c + B
          self.tip_c = self.taper_ratio*self.root_c

        else:   # Without kink
          self.root_c = 2.*self.area / (2.*y_root*(1.-self.taper_ratio) + (1.+self.taper_ratio)*np.sqrt(self.aspect_ratio*self.area))
          self.tip_c = self.taper_ratio*self.root_c
          self.kink_c = ((y_tip-y_kink)*self.root_c + (y_kink-y_root)*self.tip_c) / (y_tip-y_root)


        tan_phi0 = 0.25*(self.kink_c-self.tip_c)/(y_tip-y_kink) + np.tan(self.sweep25)

        self.mac = 2.*( 3.*y_root*self.root_c**2 \
                       +(y_kink-y_root)*(self.root_c**2+self.kink_c**2+self.root_c*self.kink_c) \
                       +(y_tip-y_kink)*(self.kink_c**2+self.tip_c**2+self.kink_c*self.tip_c) \
                      )/(3*self.area)

        y_mac = (  3.*self.root_c*y_root**2 \
                 +(y_kink-y_root)*(self.kink_c*(y_root+y_kink*2.)+self.root_c*(y_kink+y_root*2.)) \
                 +(y_tip-y_kink)*(self.tip_c*(y_kink+y_tip*2.)+self.kink_c*(y_tip+y_kink*2.)) \
                )/(3.*self.area)

        x_mac_local = ( (y_kink-y_root)*tan_phi0*((y_kink-y_root)*(self.kink_c*2.+self.root_c) \
                       +(y_tip-y_kink)*(self.kink_c*2.+self.tip_c))+(y_tip-y_root)*tan_phi0*(y_tip-y_kink)*(self.tip_c*2.+self.kink_c) \
                      )/(3*self.area)

        if (not hq_optim):
             self.x_root = 0.33*body_length**1.1 - (x_mac_local + 0.25*self.mac)
        x_root = self.x_root
        x_kink = x_root + (y_kink-y_root)*tan_phi0
        x_tip = x_root + (y_tip-y_root)*tan_phi0

        x_mac = x_root+( (x_kink-x_root)*((y_kink-y_root)*(self.kink_c*2.+self.root_c) \
                            +(y_tip-y_kink)*(self.kink_c*2.+self.tip_c))+(x_tip-x_root)*(y_tip-y_kink)*(self.tip_c*2.+self.kink_c) \
                           )/(self.area*3.)
        if (wing_attachment=="low"):
            z_root = 0.
        else:
            z_root = body_height - 0.5*self.root_toc*self.root_c

        z_kink = z_root+(y_kink-y_root)*np.tan(self.dihedral)
        z_tip = z_root+(y_tip-y_root)*np.tan(self.dihedral)

        self.root_loc = np.array([x_root, y_root, z_root])
        self.kink_loc = np.array([x_kink, y_kink, z_kink])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, None])

        self.frame_origin = [x_root, 0., z_root]

        self.gross_wet_area = 2.00*(self.area - self.root_c*body_width)
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

        # Wing setting
        #-----------------------------------------------------------------------------------------------------------
        g = earth.gravity()
        r,gam,Cp,Cv = earth.gas_data()

        disa = 0.
        rca = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach
        mass = 0.95*self.aircraft.weight_cg.mtow

        pamb,tamb,tstd,dtodz = earth.atmosphere(rca, disa)

        hld_conf = 0.
        cza_wo_htp, xlc_wo_htp, ki_wing = self.eval_aero_data(hld_conf, mach)
        cza_wing = cza_wo_htp

        # AoA = 2.5° at cruise start
        self.setting = (0.97*mass*g) / (0.5*gam*pamb*mach**2*self.area*cza_wing) - unit.rad_deg(2.5)

    def downwash_angle(self, ki_wing, cz):
        """Estimate downwash angle due to the wing
        """
        return cz * ki_wing

    def eval_aero_data(self, hld_conf, mach):
        """Estimate wing aerodynamic characteristics
        """
        body_width = self.aircraft.airframe.body.width
        wing_span = self.aircraft.airframe.wing.span
        wing_ar = self.aircraft.airframe.wing.aspect_ratio
        sweep25 = self.aircraft.airframe.wing.sweep25
        wing_c_mac = self.aircraft.airframe.wing.mac
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc[0]

        # Polhamus formula, lift gradiant without HTP
        cza_wo_htp =  (np.pi*wing_ar*(1.07*(1+body_width/wing_span)**2)*(1.-body_width/wing_span)) \
                    / (1+np.sqrt(1.+0.25*wing_ar**2*(1+np.tan(sweep25)**2-mach**2)))

        xlc_wo_htp = wing_mac_loc + (0.25+0.10*hld_conf)*wing_c_mac # Neutral point

        ki_wing = (1.05 + (body_width / self.span)**2)  / (np.pi * self.aspect_ratio)

        return cza_wo_htp, xlc_wo_htp, ki_wing

    def high_lift(self, hld_conf):
        """Retrieves max lift and zero aoa lift of a given (flap/slat) deflection (from 0 to 1).
            * 0 =< hld_type =< 10 : type of high lift device
            * 0 =< hld_conf =< 1  : (slat) flap deflection

        Typically:
            * hld_conf = 1 gives the :math:`C_{z,max}` for landind (`czmax_ld`)
            * hld_conf = 0.1 to 0.5 gives the :math:`C_{z,max}` for take-off(`czmax_to`)

        .. todo:: check this documentation
        """

        # Maximum lift coefficients of different airfoils, DUBS 1987
        czmax_ld = {0 : 1.45 ,  # Clean
                    1 : 2.25 ,  # Flap only, Rotation without slot
                    2 : 2.60 ,  # Flap only, Rotation single slot      (ATR)
                    3 : 2.80 ,  # Flap only, Rotation double slot
                    4 : 2.80 ,  # Fowler Flap
                    5 : 2.00 ,  # Slat only
                    6 : 2.45 ,  # Slat + Flap rotation without slot
                    7 : 2.70 ,  # Slat + Flap rotation single slot
                    8 : 2.90 ,  # Slat + Flap rotation double slot
                    9 : 2.90 ,  # Slat + Fowler                      (A320)
                    10 : 3.20,  # Slat + Fowler + Fowler double slot (A321)
                    }.get(self.hld_type, "Erreur - high_lift_, HLDtype out of range")    # 9 is default if x not found

        if (self.hld_type<5):
            czmax_base = 1.45      # Flap only
        else:
            if (hld_conf==0): czmax_base = 1.45 # Clean
            else: czmax_base = 2.00             # Slat + Flap

        czmax_2d = (1.-hld_conf)*czmax_base + hld_conf*czmax_ld     # Setting effect

        if (hld_conf==0):
            cz0_2d = 0. # Clean
        else:
            cz0_2d = czmax_2d - czmax_base  # Assumed the Lift vs AoA is just translated upward and Cz0 clean equal to zero

        # Source : http://aerodesign.stanford.edu/aircraftdesign/highlift/clmaxest.html
        czmax = czmax_2d * (1.-0.08*np.cos(self.sweep25)**2) * np.cos(self.sweep25)**0.75
        cz0 = cz0_2d

        return czmax, cz0

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        mzfw = self.aircraft.weight_cg.mzfw
        hld_conf_ld = self.aircraft.aerodynamics.hld_conf_ld

        (cz_max_ld,cz0) = self.high_lift(hld_conf_ld)

        A = 32*self.area**1.1
        B = 4.*self.span**2 * np.sqrt(mtow*mzfw)
        C = 1.1e-6*(1.+2.*self.aspect_ratio)/(1.+self.aspect_ratio)
        D = (0.6*self.root_toc+0.3*self.kink_toc+0.1*self.tip_toc) * (self.area/self.span)
        E = np.cos(self.sweep25)**2
        F = 1200.*max(0., cz_max_ld - 1.8)**1.5

        self.mass = self.mass_correction_factor * (A + (B*C)/(D*E) + F)   # Shevell formula + high lift device regression

        self.cg =  0.25*(self.root_loc + 0.40*np.array([self.root_c, 0., 0.])) \
                 + 0.55*(self.kink_loc + 0.40*np.array([self.kink_c, 0., 0.])) \
                 + 0.20*(self.tip_loc + 0.40*np.array([self.tip_c, 0., 0.]))

    def sketch_3view(self):
        wing_x_root = self.root_loc[0]
        wing_y_root = self.root_loc[1]
        wing_z_root = self.root_loc[2]
        wing_c_root = self.root_c
        wing_toc_r = self.root_toc
        wing_x_kink = self.kink_loc[0]
        wing_y_kink = self.kink_loc[1]
        wing_z_kink = self.kink_loc[2]
        wing_c_kink = self.kink_c
        wing_toc_k = self.kink_toc
        wing_x_tip = self.tip_loc[0]
        wing_y_tip = self.tip_loc[1]
        wing_z_tip = self.tip_loc[2]
        wing_c_tip = self.tip_c
        wing_toc_t = self.tip_toc

        le = np.array([[wing_x_tip              , -wing_y_tip  , wing_z_tip ],
                       [wing_x_kink             , -wing_y_kink , wing_z_kink],
                       [wing_x_root             , -wing_y_root , wing_z_root],
                       [wing_x_root             ,  wing_y_root , wing_z_root],
                       [wing_x_kink             ,  wing_y_kink , wing_z_kink],
                       [wing_x_tip              ,  wing_y_tip  , wing_z_tip]])

        te = np.array([[wing_x_tip+wing_c_tip   , -wing_y_tip  , wing_z_tip ],
                       [wing_x_kink+wing_c_kink , -wing_y_kink , wing_z_kink],
                       [wing_x_root+wing_c_root , -wing_y_root , wing_z_root],
                       [wing_x_root+wing_c_root ,  wing_y_root , wing_z_root],
                       [wing_x_kink+wing_c_kink ,  wing_y_kink , wing_z_kink],
                       [wing_x_tip+wing_c_tip   ,  wing_y_tip  , wing_z_tip]])

        toc = np.array([wing_toc_t, wing_toc_k, wing_toc_r, wing_toc_r, wing_toc_k, wing_toc_t])

        wing_xy = np.array([[wing_x_root             ,  wing_y_root ],
                            [wing_x_tip              ,  wing_y_tip  ],
                            [wing_x_tip+wing_c_tip   ,  wing_y_tip  ],
                            [wing_x_kink+wing_c_kink ,  wing_y_kink ],
                            [wing_x_root+wing_c_root ,  wing_y_root ],
                            [wing_x_root+wing_c_root , -wing_y_root ],
                            [wing_x_kink+wing_c_kink , -wing_y_kink ],
                            [wing_x_tip+wing_c_tip   , -wing_y_tip  ],
                            [wing_x_tip              , -wing_y_tip  ],
                            [wing_x_root             , -wing_y_root ],
                            [wing_x_root             ,  wing_y_root ]])

        wing_yz = np.array([[ wing_y_root  , wing_z_root                        ],
                            [ wing_y_kink  , wing_z_kink                        ],
                            [ wing_y_tip   , wing_z_tip                         ],
                            [ wing_y_tip   , wing_z_tip+wing_toc_t*wing_c_tip   ],
                            [ wing_y_kink  , wing_z_kink+wing_toc_k*wing_c_kink ],
                            [ wing_y_root  , wing_z_root+wing_toc_r*wing_c_root ],
                            [-wing_y_root  , wing_z_root+wing_toc_r*wing_c_root ],
                            [-wing_y_kink  , wing_z_kink+wing_toc_k*wing_c_kink ],
                            [-wing_y_tip   , wing_z_tip+wing_toc_t*wing_c_tip   ],
                            [-wing_y_tip   , wing_z_tip                         ],
                            [-wing_y_kink  , wing_z_kink                        ],
                            [-wing_y_root  , wing_z_root                        ],
                            [ wing_y_root  , wing_z_root                        ]])

        wing_xz = np.array([[wing_x_tip                  , wing_z_tip+wing_toc_t*wing_c_tip                           ],
                            [wing_x_tip+0.1*wing_c_tip   , wing_z_tip+wing_toc_t*wing_c_tip-0.5*wing_toc_t*wing_c_tip ],
                            [wing_x_tip+0.7*wing_c_tip   , wing_z_tip+wing_toc_t*wing_c_tip-0.5*wing_toc_t*wing_c_tip ],
                            [wing_x_tip+wing_c_tip       , wing_z_tip+wing_toc_t*wing_c_tip                           ],
                            [wing_x_tip+0.7*wing_c_tip   , wing_z_tip+wing_toc_t*wing_c_tip+0.5*wing_toc_t*wing_c_tip ],
                            [wing_x_tip+0.1*wing_c_tip   , wing_z_tip+wing_toc_t*wing_c_tip+0.5*wing_toc_t*wing_c_tip ],
                            [wing_x_tip                  , wing_z_tip+wing_toc_t*wing_c_tip                           ],
                            [wing_x_kink                 , wing_z_kink+0.5*wing_toc_k*wing_c_kink                     ],
                            [wing_x_root                 , wing_z_root+0.5*wing_toc_r*wing_c_root                     ],
                            [wing_x_root+0.1*wing_c_root , wing_z_root                                                ],
                            [wing_x_root+0.7*wing_c_root , wing_z_root                                                ],
                            [wing_x_root+wing_c_root     , wing_z_root+0.5*wing_toc_r*wing_c_root                     ],
                            [wing_x_kink+wing_c_kink     , wing_z_kink+0.5*wing_toc_k*wing_c_kink                     ],
                            [wing_x_tip+wing_c_tip       , wing_z_tip+wing_toc_t*wing_c_tip                           ]])

        if (self.aircraft.arrangement.tank_architecture=="pods"):
            wing_x_body = self.aircraft.airframe.tank.wing_axe_x
            wing_z_body = self.aircraft.airframe.tank.wing_axe_z
            wing_c_body = self.aircraft.airframe.tank.wing_axe_c
            tip_wing_xz = np.array([[wing_x_tip                  , wing_z_tip+wing_toc_t*wing_c_tip                           ],
                                    [wing_x_tip+0.1*wing_c_tip   , wing_z_tip+wing_toc_t*wing_c_tip-0.5*wing_toc_t*wing_c_tip ],
                                    [wing_x_tip+0.7*wing_c_tip   , wing_z_tip+wing_toc_t*wing_c_tip-0.5*wing_toc_t*wing_c_tip ],
                                    [wing_x_tip+wing_c_tip       , wing_z_tip+wing_toc_t*wing_c_tip                           ],
                                    [wing_x_tip+0.7*wing_c_tip   , wing_z_tip+wing_toc_t*wing_c_tip+0.5*wing_toc_t*wing_c_tip ],
                                    [wing_x_tip+0.1*wing_c_tip   , wing_z_tip+wing_toc_t*wing_c_tip+0.5*wing_toc_t*wing_c_tip ],
                                    [wing_x_tip                  , wing_z_tip+wing_toc_t*wing_c_tip                           ],
                                    [wing_x_body                 , wing_z_body+0.5*wing_toc_k*wing_c_kink                     ],
                                    [wing_x_body+wing_c_body     , wing_z_body+0.5*wing_toc_k*wing_c_kink                     ],
                                    [wing_x_tip+wing_c_tip       , wing_z_tip+wing_toc_t*wing_c_tip                           ]])
        else:
            tip_wing_xz = None

        return {"xy":wing_xy, "yz":wing_yz, "xz":wing_xz, "xz_tip":tip_wing_xz, "le":le, "te":te, "toc":toc}


class Vstab(Component):

    def __init__(self, aircraft):
        super(Vstab, self).__init__(aircraft)

    def get_component_type(self):
        return "vtp"

    def sketch_3view(self):
        vtp_t_o_c = self.toc
        vtp_x_root = self.root_loc[0]
        vtp_y_root = self.root_loc[1]
        vtp_z_root = self.root_loc[2]
        vtp_c_root = self.root_c
        vtp_x_tip = self.tip_loc[0]
        vtp_y_tip = self.tip_loc[1]
        vtp_z_tip = self.tip_loc[2]
        vtp_c_tip = self.tip_c

        vtp_le = np.array([[vtp_x_root            , vtp_y_root , vtp_z_root ],
                           [vtp_x_tip             , vtp_y_tip  , vtp_z_tip  ]])

        vtp_te = np.array([[vtp_x_root+vtp_c_root , vtp_y_root , vtp_z_root ],
                           [vtp_x_tip+vtp_c_tip   , vtp_y_tip  , vtp_z_tip  ]])

        vtp_toc = np.array([vtp_t_o_c, vtp_t_o_c])

        vtp_xz = np.array([[vtp_x_root            , vtp_z_root ],
                           [vtp_x_tip             , vtp_z_tip  ],
                           [vtp_x_tip+vtp_c_tip   , vtp_z_tip  ],
                           [vtp_x_root+vtp_c_root , vtp_z_root ],
                           [vtp_x_root            , vtp_z_root ]])

        vtp_xy = np.array([[vtp_x_root                , vtp_y_root                            ],
                           [vtp_x_root+0.1*vtp_c_root , vtp_y_root + 0.5*vtp_t_o_c*vtp_c_root ],
                           [vtp_x_root+0.7*vtp_c_root , vtp_y_root + 0.5*vtp_t_o_c*vtp_c_root ],
                           [vtp_x_root+vtp_c_root     , vtp_y_root                            ],
                           [vtp_x_root+0.7*vtp_c_root , vtp_y_root - 0.5*vtp_t_o_c*vtp_c_root ],
                           [vtp_x_root+0.1*vtp_c_root , vtp_y_root - 0.5*vtp_t_o_c*vtp_c_root ],
                           [vtp_x_root                , vtp_y_root                            ],
                           [vtp_x_tip                 , vtp_y_tip                             ],
                           [vtp_x_tip+0.1*vtp_c_tip   , vtp_y_tip + 0.5*vtp_t_o_c*vtp_c_tip   ],
                           [vtp_x_tip+0.7*vtp_c_tip   , vtp_y_tip + 0.5*vtp_t_o_c*vtp_c_tip   ],
                           [vtp_x_tip+vtp_c_tip       , vtp_y_tip                             ],
                           [vtp_x_tip+0.7*vtp_c_tip   , vtp_y_tip - 0.5*vtp_t_o_c*vtp_c_tip   ],
                           [vtp_x_tip+0.1*vtp_c_tip   , vtp_y_tip - 0.5*vtp_t_o_c*vtp_c_tip   ],
                           [vtp_x_tip                 , vtp_y_tip                             ]])

        vtp_yz = np.array([[vtp_y_root + 0.5*vtp_t_o_c*vtp_c_root , vtp_z_root ],
                           [vtp_y_tip + 0.5*vtp_t_o_c*vtp_c_tip   , vtp_z_tip  ],
                           [vtp_y_tip - 0.5*vtp_t_o_c*vtp_c_tip   , vtp_z_tip  ],
                           [vtp_y_root - 0.5*vtp_t_o_c*vtp_c_root , vtp_z_root ],
                           [vtp_y_root + 0.5*vtp_t_o_c*vtp_c_root , vtp_z_root ]])

        return {"xy":vtp_xy , "yz":vtp_yz, "xz":vtp_xz, "le":vtp_le, "te":vtp_te, "toc":vtp_toc}

class VtpClassic(Vstab):

    def __init__(self, aircraft):
        super(VtpClassic, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.20*wing_area  # Design variable for hq_optim
        self.height = None
        self.aspect_ratio = aircraft.get_init(self,"aspect_ratio")
        self.taper_ratio = aircraft.get_init(self,"taper_ratio")
        self.toc = aircraft.get_init(self,"toc")
        self.sweep25 = aircraft.get_init(self,"sweep25", val=self.__sweep25())
        self.thrust_volume_factor = aircraft.get_init(self,"thrust_volume_factor")
        self.wing_volume_factor = aircraft.get_init(self,"wing_volume_factor")
        self.anchor_ratio = aircraft.get_init(self,"anchor_ratio")
        self.lever_arm = None

        self.root_loc = np.full(3,None)     # Position of root chord leading edge
        self.root_c = None                  # root chord length

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None
        
        self.mass_correction_factor = aircraft.get_init(self,"mass_correction_factor")

    def __sweep25(self):
        sweep25 = max(unit.rad_deg(25.), self.aircraft.airframe.wing.sweep25 + unit.rad_deg(10.)) # Empirical law
        return sweep25

    def eval_geometry(self):
        body_length = self.aircraft.airframe.body.length
        body_height = self.aircraft.airframe.body.height
        tail_cone_length = self.aircraft.airframe.body.tail_cone_length
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_mac = self.aircraft.airframe.wing.mac

        self.height = np.sqrt(self.aspect_ratio*self.area)
        self.root_c = 2*self.area/(self.height*(1+self.taper_ratio))
        self.tip_c = self.taper_ratio*self.root_c

        x_root = body_length*(1-tail_cone_length/body_length*(1-self.anchor_ratio)) - self.root_c
        x_tip = x_root + 0.25*(self.root_c-self.tip_c) + self.height*np.tan(self.sweep25)

        y_root = 0.
        y_tip = 0.

        z_root = body_height
        z_tip = z_root + self.height

        self.mac = self.height*(self.root_c**2+self.tip_c**2+self.root_c*self.tip_c)/(3*self.area)
        x_mac = x_root+(x_tip-x_root)*self.height*(2*self.tip_c+self.root_c)/(6*self.area)
        y_mac = 0.
        z_mac = z_tip**2*(2*self.tip_c+self.root_c)/(6*self.area)

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_mac_loc[0] + 0.25*wing_mac)

        self.root_loc = np.array([x_root, y_root, z_root])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = [x_root, 0., z_root]

        self.gross_wet_area = 2.01*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_aero_data(self):
        """WARNING : output values are in Wing reference area
        """
        wing_area = self.aircraft.airframe.wing.area
        cyb_vtp =  (np.pi*self.aspect_ratio)/(1+np.sqrt(1+(self.aspect_ratio/2)**2))*(self.area/wing_area)   # Helmbold formula
        xlc_vtp = self.mac_loc[0] + 0.25*self.mac   # Position of VTP center of lift
        aoa_max_vtp = unit.rad_deg(30.)             # Maximum angle of attack allowed for VTP
        ki_vtp = 1.3/(np.pi*self.aspect_ratio)      # VTP induced drag coefficient
        return cyb_vtp, xlc_vtp, aoa_max_vtp, ki_vtp

    def eval_mass(self):
        self.mass = self.mass_correction_factor * 25. * self.area
        self.cg = self.mac_loc + 0.20*np.array([self.mac, 0., 0.])

    def eval_area(self):
        reference_thrust = self.aircraft.power_system.get_reference_thrust()
        nacelle_loc_ext = self.aircraft.airframe.nacelle.frame_origin
        wing_area = self.aircraft.airframe.wing.area
        wing_span = self.aircraft.airframe.wing.span
        area_1 = self.thrust_volume_factor*(1.e-3*reference_thrust*nacelle_loc_ext[1])/self.lever_arm
        area_2 = self.wing_volume_factor*(wing_area*wing_span)/self.lever_arm
        self.area = max(area_1,area_2)

class VtpTtail(Vstab):

    def __init__(self, aircraft):
        super(VtpTtail, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.20*wing_area  # Design variable for hq_optim
        self.height = None
        self.aspect_ratio = aircraft.get_init(self,"aspect_ratio")
        self.taper_ratio = aircraft.get_init(self,"taper_ratio")
        self.toc = aircraft.get_init(self,"toc")
        self.sweep25 = aircraft.get_init(self,"sweep25", val=self.__sweep25())
        self.thrust_volume_factor = aircraft.get_init(self,"thrust_volume_factor")
        self.wing_volume_factor = aircraft.get_init(self,"wing_volume_factor")
        self.anchor_ratio = aircraft.get_init(self,"anchor_ratio")
        self.lever_arm = None

        self.root_loc = np.full(3,None)     # Position of root chord leading edge
        self.root_c = None                  # root chord length

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None
        
        self.mass_correction_factor = aircraft.get_init(self,"mass_correction_factor")

    def __sweep25(self):
        sweep25 = max(unit.rad_deg(25.), self.aircraft.airframe.wing.sweep25 + unit.rad_deg(10.)) # Empirical law
        return sweep25

    def eval_geometry(self):
        body_length = self.aircraft.airframe.body.length
        body_height = self.aircraft.airframe.body.height
        tail_cone_length = self.aircraft.airframe.body.tail_cone_length
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_mac = self.aircraft.airframe.wing.mac

        self.height = np.sqrt(self.aspect_ratio*self.area)
        self.root_c = 2*self.area/(self.height*(1+self.taper_ratio))
        self.tip_c = self.taper_ratio*self.root_c

        x_root = body_length*(1-tail_cone_length/body_length*(1-self.anchor_ratio)) - self.root_c
        x_tip = x_root + 0.25*(self.root_c-self.tip_c) + self.height*np.tan(self.sweep25)

        y_root = 0.
        y_tip = 0.

        z_root = body_height
        z_tip = z_root + self.height

        self.mac = self.height*(self.root_c**2+self.tip_c**2+self.root_c*self.tip_c)/(3*self.area)
        x_mac = x_root+(x_tip-x_root)*self.height*(2*self.tip_c+self.root_c)/(6*self.area)
        y_mac = 0.
        z_mac = z_tip**2*(2*self.tip_c+self.root_c)/(6*self.area)

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_mac_loc[0] + 0.25*wing_mac)

        self.root_loc = np.array([x_root, y_root, z_root])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = [x_root, 0., z_root]

        self.gross_wet_area = 2.01*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_aero_data(self):
        """WARNING : output values are in Wing reference area
        """
        wing_area = self.aircraft.airframe.wing.area
        # Helmbold formula corrected with endplate effect
        cyb_vtp =  1.2*(np.pi*self.aspect_ratio)/(1+np.sqrt(1+(self.aspect_ratio/2)**2))*(self.area/wing_area)
        xlc_vtp = self.mac_loc[0] + 0.25*self.mac   # Position of VTP center of lift
        aoa_max_vtp = unit.rad_deg(35.)             # Maximum angle of attack allowed for VTP
        ki_vtp = 1.1/(np.pi*self.aspect_ratio)      # VTP induced drag coefficient
        return cyb_vtp, xlc_vtp, aoa_max_vtp, ki_vtp

    def eval_mass(self):
        self.mass = self.mass_correction_factor * 28. * self.area
        self.cg = self.mac_loc + 0.20*np.array([self.mac, 0., 0.])

    def eval_area(self):
        reference_thrust = self.aircraft.power_system.get_reference_thrust()
        nacelle_loc_ext = self.aircraft.airframe.nacelle.frame_origin
        wing_area = self.aircraft.airframe.wing.area
        wing_span = self.aircraft.airframe.wing.span
        area_1 = self.thrust_volume_factor*(1.e-3*reference_thrust*nacelle_loc_ext[1])/self.lever_arm
        area_2 = self.wing_volume_factor*(wing_area*wing_span)/self.lever_arm
        self.area = max(area_1,area_2)

class VtpHtail(Vstab):

    def __init__(self, aircraft, side):
        super(VtpHtail, self).__init__(aircraft)

        self.airplane_side = side

        wing_area = aircraft.airframe.wing.area

        self.area = 0.10*wing_area  # Design variable for hq_optim
        self.height = None
        self.aspect_ratio = aircraft.get_init(self,"aspect_ratio")
        self.taper_ratio = aircraft.get_init(self,"taper_ratio")
        self.toc = aircraft.get_init(self,"toc")
        self.sweep25 = aircraft.get_init(self,"sweep25", val=self.__sweep25())
        self.thrust_volume_factor = aircraft.get_init(self,"thrust_volume_factor")
        self.wing_volume_factor = aircraft.get_init(self,"wing_volume_factor")
        self.lever_arm = None

        self.root_loc = np.full(3,None)     # Position of root chord leading edge
        self.root_c = None                  # root chord length

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None
        
        self.mass_correction_factor = aircraft.get_init(self,"mass_correction_factor")

    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

    def __sweep25(self):
        sweep25 = max(unit.rad_deg(25.), self.aircraft.airframe.wing.sweep25 + unit.rad_deg(10.)) # Empirical law
        return sweep25

    def eval_geometry(self):
        htp_tip_loc = self.aircraft.airframe.horizontal_stab.tip_loc
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_mac = self.aircraft.airframe.wing.mac

        self.height = np.sqrt(self.aspect_ratio*(self.area))
        self.root_c = 2*(self.area)/(self.height*(1+self.taper_ratio))
        self.tip_c = self.taper_ratio*self.root_c

        x_root = htp_tip_loc[0]
        x_tip = x_root + 0.25*(self.root_c-self.tip_c) + self.height*np.tan(self.sweep25)

        y_root = htp_tip_loc[1]
        y_tip = htp_tip_loc[1]

        z_root = htp_tip_loc[2]
        z_tip = z_root + self.height

        self.mac = self.height*(self.root_c**2+self.tip_c**2+self.root_c*self.tip_c)/(3*(self.area))
        x_mac = x_root+(x_tip-x_root)*self.height*(2*self.tip_c+self.root_c)/(6*(self.area))
        y_mac = y_tip
        z_mac = z_tip**2*(2*self.tip_c+self.root_c)/(6*self.area)

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_mac_loc[0] + 0.25*wing_mac)

        self.root_loc = np.array([x_root, y_root*self.get_side(), z_root])
        self.tip_loc = np.array([x_tip, y_tip*self.get_side(), z_tip])
        self.mac_loc = np.array([x_mac, y_mac*self.get_side(), z_mac])

        self.frame_origin = [x_root, y_root, z_root]

        self.gross_wet_area = 2.01*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_aero_data(self):
        """WARNING : output values are in Wing reference area
        """
        wing_area = self.aircraft.airframe.wing.area
        cyb_vtp =  (np.pi*self.aspect_ratio)/(1+np.sqrt(1+(self.aspect_ratio/2)**2))*(self.area/wing_area)   # Helmbold formula
        xlc_vtp = self.mac_loc[0] + 0.25*self.mac   # Position of VTP center of lift
        aoa_max_vtp = unit.rad_deg(35.)             # Maximum angle of attack allowed for VTP
        ki_vtp = 1.3/(np.pi*self.aspect_ratio)      # VTP induced drag coefficient
        return cyb_vtp, xlc_vtp, aoa_max_vtp, ki_vtp

    def eval_mass(self):
        self.mass = self.mass_correction_factor * 25. * self.area
        self.cg = self.mac_loc + 0.20*np.array([self.mac, 0., 0.])

    def eval_area(self):
        reference_thrust = self.aircraft.power_system.get_reference_thrust()
        nacelle_loc_ext = self.aircraft.airframe.nacelle.frame_origin
        wing_area = self.aircraft.airframe.wing.area
        wing_span = self.aircraft.airframe.wing.span
        area_1 = self.thrust_volume_factor*(1.e-3*reference_thrust*nacelle_loc_ext[1])/self.lever_arm
        area_2 = self.wing_volume_factor*(wing_area*wing_span)/self.lever_arm
        self.area = 0.5 * max(area_1,area_2)


class Hstab(Component):

    def __init__(self, aircraft):
        super(Hstab, self).__init__(aircraft)

    def get_component_type(self):
        return "htp"

    def sketch_3view(self):
        htp_span = self.span
        htp_dihedral = self.dihedral
        htp_t_o_c = self.toc
        htp_x_axe = self.axe_loc[0]
        htp_z_axe = self.axe_loc[2]
        htp_c_axe = self.axe_c
        htp_x_tip = self.tip_loc[0]
        htp_z_tip = self.tip_loc[2]
        htp_c_tip = self.tip_c

        htp_le = np.array([[htp_x_tip           , -0.5*htp_span , htp_z_tip],
                           [htp_x_axe           ,  0            , htp_z_axe],
                           [htp_x_tip           ,  0.5*htp_span , htp_z_tip]])

        htp_te = np.array([[htp_x_tip+htp_c_tip , -0.5*htp_span , htp_z_tip],
                           [htp_x_axe+htp_c_axe ,  0            , htp_z_axe],
                           [htp_x_tip+htp_c_tip ,  0.5*htp_span , htp_z_tip]])

        htp_toc = np.array([htp_t_o_c, htp_t_o_c, htp_t_o_c])

        htp_xy = np.array([[htp_x_axe           ,  0            ],
                           [htp_x_tip           ,  0.5*htp_span ],
                           [htp_x_tip+htp_c_tip ,  0.5*htp_span ],
                           [htp_x_axe+htp_c_axe ,  0            ],
                           [htp_x_tip+htp_c_tip , -0.5*htp_span ],
                           [htp_x_tip           , -0.5*htp_span ],
                           [htp_x_axe           ,  0            ]])

        htp_xz = np.array([[htp_x_tip              , htp_z_tip                          ],
                           [htp_x_tip+0.1*htp_c_tip , htp_z_tip+0.5*htp_t_o_c*htp_c_tip ],
                           [htp_x_tip+0.7*htp_c_tip , htp_z_tip+0.5*htp_t_o_c*htp_c_tip ],
                           [htp_x_tip+htp_c_tip     , htp_z_tip                         ],
                           [htp_x_tip+0.7*htp_c_tip , htp_z_tip-0.5*htp_t_o_c*htp_c_tip ],
                           [htp_x_tip+0.1*htp_c_tip , htp_z_tip-0.5*htp_t_o_c*htp_c_tip ],
                           [htp_x_tip               , htp_z_tip                         ],
                           [htp_x_axe               , htp_z_axe                         ],
                           [htp_x_axe+0.1*htp_c_axe , htp_z_axe-0.5*htp_t_o_c*htp_c_axe ],
                           [htp_x_axe+0.7*htp_c_axe , htp_z_axe-0.5*htp_t_o_c*htp_c_axe ],
                           [htp_x_axe+htp_c_axe     , htp_z_axe                         ],
                           [htp_x_tip+htp_c_tip     , htp_z_tip                         ],
                           [htp_x_tip+0.7*htp_c_tip , htp_z_tip-0.5*htp_t_o_c*htp_c_tip ],
                           [htp_x_tip+0.1*htp_c_tip , htp_z_tip-0.5*htp_t_o_c*htp_c_tip ],
                           [htp_x_tip               , htp_z_tip                         ]])

        htp_yz = np.array([[ 0           , htp_z_axe                                                        ],
                           [ 0.5*htp_span , htp_z_axe+0.5*htp_span*np.tan(htp_dihedral)                     ],
                           [ 0.5*htp_span , htp_z_axe+0.5*htp_span*np.tan(htp_dihedral)-htp_t_o_c*htp_c_tip ],
                           [ 0            , htp_z_axe-htp_t_o_c*htp_c_axe                                   ],
                           [-0.5*htp_span , htp_z_axe+0.5*htp_span*np.tan(htp_dihedral)-htp_t_o_c*htp_c_tip ],
                           [-0.5*htp_span , htp_z_axe+0.5*htp_span*np.tan(htp_dihedral)                     ],
                           [ 0            , htp_z_axe                                                       ]])

        return {"xy":htp_xy, "yz":htp_yz, "xz":htp_xz, "le":htp_le, "te":htp_te, "toc":htp_toc}

class HtpClassic(Hstab):

    def __init__(self, aircraft):
        super(HtpClassic, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.33*wing_area  # Design variable for hq_optim
        self.span = None
        self.aspect_ratio = aircraft.get_init(self,"aspect_ratio")
        self.taper_ratio = aircraft.get_init(self,"taper_ratio")
        self.toc = aircraft.get_init(self,"toc")
        self.sweep25 = aircraft.get_init(self,"sweep25", val=self.__sweep25())
        self.dihedral = aircraft.get_init(self,"dihedral")
        self.volume_factor = aircraft.get_init(self,"volume_factor")
        self.lever_arm = None

        self.axe_loc = np.full(3,None)     # Position of the virtual central chord
        self.axe_c = None                  # Length of the virtual central chord

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None
        
        self.mass_correction_factor = aircraft.get_init(self,"mass_correction_factor")

    def __sweep25(self):
        sweep25 = abs(self.aircraft.airframe.wing.sweep25) + unit.rad_deg(5)     # Design rule
        return sweep25

    def eval_geometry(self):
        body_height = self.aircraft.airframe.body.height
        vtp_root_loc = self.aircraft.airframe.vertical_stab.root_loc
        vtp_root_c = self.aircraft.airframe.vertical_stab.root_c
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_mac = self.aircraft.airframe.wing.mac

        self.span = np.sqrt(self.aspect_ratio*self.area)
        y_axe = 0.
        y_tip = 0.5*self.span

        htp_z_wise_anchor = 0.80       # Locate HTP versus end body height
        z_axe = htp_z_wise_anchor*body_height
        z_tip = z_axe + y_tip*np.tan(self.dihedral)

        self.axe_c = 2.*self.area/(self.span*(1+self.taper_ratio))
        self.tip_c = self.taper_ratio*self.axe_c

        self.mac = self.span*(self.axe_c**2+self.tip_c**2+self.axe_c*self.tip_c)/(3.*self.area)
        y_mac = y_tip**2*(2*self.tip_c+self.axe_c)/(3*self.area)
        z_mac = z_tip**2*(2*self.tip_c+self.axe_c)/(3*self.area)
        x_tip_local = 0.25*(self.axe_c-self.tip_c) + y_tip*np.tan(self.sweep25)
        x_mac_local = y_tip*x_tip_local*(self.tip_c*2.+self.axe_c)/(3.*self.area)

        x_axe = vtp_root_loc[0] + 0.50*vtp_root_c - 0.2*self.axe_c

        x_tip = x_axe + x_tip_local
        x_mac = x_axe + x_mac_local

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_mac_loc[0] + 0.25*wing_mac)

        self.axe_loc = np.array([x_axe, y_axe, z_axe])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = self.axe_loc

        self.gross_wet_area = 1.63*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_aero_data(self):
        """WARNING : output values are in Wing reference area
        """
        wing_area = self.aircraft.airframe.wing.area
        cza_htp =  (np.pi*self.aspect_ratio)/(1+np.sqrt(1+(self.aspect_ratio/2)**2))*(self.area/wing_area)  # Helmbold formula
        xlc_htp = self.mac_loc[0] + 0.25*self.mac   # Position of HTP center of lift
        aoa_max_htp = unit.rad_deg(9.)              # Maximum angle of attack allowed for HTP
        ki_htp = 1.2/(np.pi*self.aspect_ratio)      # HTP induced drag coefficient
        return cza_htp, xlc_htp, aoa_max_htp, ki_htp

    def eval_mass(self):
        self.mass = self.mass_correction_factor * 22. * self.area
        self.cg = self.mac_loc + 0.20*np.array([self.mac, 0., 0.])

    def eval_area(self):
        wing_area = self.aircraft.airframe.wing.area
        wing_mac = self.aircraft.airframe.wing.mac

        self.area = self.volume_factor*(wing_area*wing_mac/self.lever_arm)

class HtpTtail(Hstab):

    def __init__(self, aircraft):
        super(HtpTtail, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.33*wing_area  # Design variable for hq_optim
        self.span = None
        self.aspect_ratio = aircraft.get_init(self,"aspect_ratio")
        self.taper_ratio = aircraft.get_init(self,"taper_ratio")
        self.height_ratio = aircraft.get_init(self,"height_ratio")
        self.toc = aircraft.get_init(self,"toc")
        self.sweep25 = aircraft.get_init(self,"sweep25", val=self.__sweep25())
        self.dihedral = aircraft.get_init(self,"dihedral")
        self.volume_factor = aircraft.get_init(self,"volume_factor")
        self.lever_arm = None

        self.axe_loc = np.full(3,None)     # Position of the central chord
        self.axe_c = None                  # Length of the central chord

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None
        
        self.mass_correction_factor = aircraft.get_init(self,"mass_correction_factor")

    def __sweep25(self):
        sweep25 = abs(self.aircraft.airframe.wing.sweep25) + unit.rad_deg(5)     # Design rule
        return sweep25

    def eval_geometry(self):
        body_height = self.aircraft.airframe.body.height
        vtp_tip_loc = self.aircraft.airframe.vertical_stab.tip_loc
        vtp_tip_c = self.aircraft.airframe.vertical_stab.tip_c
        vtp_height = self.aircraft.airframe.vertical_stab.height
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_mac = self.aircraft.airframe.wing.mac

        self.span = np.sqrt(self.aspect_ratio*self.area)
        y_axe = 0.
        y_tip = 0.5*self.span

        z_axe = (1.-self.height_ratio)*body_height + self.height_ratio*(body_height+vtp_height)
        z_tip = z_axe + y_tip*np.tan(self.dihedral)

        self.axe_c = 2.*self.area/(self.span*(1+self.taper_ratio))
        self.tip_c = self.taper_ratio*self.axe_c

        self.mac = self.span*(self.axe_c**2+self.tip_c**2+self.axe_c*self.tip_c)/(3.*self.area)
        y_mac = y_tip**2*(2*self.tip_c+self.axe_c)/(3*self.area)
        z_mac = z_tip**2*(2*self.tip_c+self.axe_c)/(3*self.area)
        x_tip_local = 0.25*(self.axe_c-self.tip_c) + y_tip*np.tan(self.sweep25)
        x_mac_local = y_tip*x_tip_local*(self.tip_c*2.+self.axe_c)/(3.*self.area)

        x_axe = vtp_tip_loc[0] + 0.30*vtp_tip_c - 0.80*self.tip_c

        x_tip = x_axe + x_tip_local
        x_mac = x_axe + x_mac_local

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_mac_loc[0] + 0.25*wing_mac)

        self.axe_loc = np.array([x_axe, y_axe, z_axe])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = self.axe_loc

        self.gross_wet_area = 2.01*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_aero_data(self):
        """WARNING : output values are in Wing reference area
        """
        wing_area = self.aircraft.airframe.wing.area
        cza_htp =  (np.pi*self.aspect_ratio)/(1+np.sqrt(1+(self.aspect_ratio/2)**2))*(self.area/wing_area)  # Helmbold formula
        xlc_htp = self.mac_loc[0] + 0.25*self.mac   # Position of HTP center of lift
        aoa_max_htp = unit.rad_deg(9.)              # Maximum angle of attack allowed for HTP
        ki_htp = 1.2/(np.pi*self.aspect_ratio)      # HTP induced drag coefficient
        return cza_htp, xlc_htp, aoa_max_htp, ki_htp

    def eval_mass(self):
        self.mass = self.mass_correction_factor * 22. * self.area
        self.cg = self.mac_loc + 0.20*np.array([self.mac, 0., 0.])

    def eval_area(self):
        wing_area = self.aircraft.airframe.wing.area
        wing_mac = self.aircraft.airframe.wing.mac

        self.area = self.volume_factor*(wing_area*wing_mac/self.lever_arm)

class HtpHtail(Hstab):

    def __init__(self, aircraft):
        super(HtpHtail, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.33*wing_area  # Design variable for hq_optim
        self.span = None
        self.aspect_ratio = aircraft.get_init(self,"aspect_ratio")
        self.taper_ratio = aircraft.get_init(self,"taper_ratio")
        self.toc = aircraft.get_init(self,"toc")
        self.sweep25 = aircraft.get_init(self,"sweep25", val=self.__sweep25())
        self.dihedral = aircraft.get_init(self,"dihedral")
        self.volume_factor = aircraft.get_init(self,"volume_factor")
        self.lever_arm = None

        self.axe_loc = np.full(3,None)     # Position of the virtual central chord
        self.axe_c = None                  # Length of the virtual central chord

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None
        
        self.mass_correction_factor = aircraft.get_init(self,"mass_correction_factor")

    def __sweep25(self):
        sweep25 = abs(self.aircraft.airframe.wing.sweep25) + unit.rad_deg(5)     # Design rule
        return sweep25

    def eval_geometry(self):
        body_length = self.aircraft.airframe.body.length
        body_height = self.aircraft.airframe.body.height
        body_cone_length = self.aircraft.airframe.body.tail_cone_length
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_mac = self.aircraft.airframe.wing.mac

        self.span = np.sqrt(self.aspect_ratio*self.area)
        y_axe = 0.
        y_tip = 0.5*self.span

        htp_z_wise_anchor = 0.80       # Locate HTP versus end body height
        z_axe = htp_z_wise_anchor*body_height
        z_tip = z_axe + y_tip*np.tan(self.dihedral)

        self.axe_c = 2.*self.area/(self.span*(1+self.taper_ratio))
        self.tip_c = self.taper_ratio*self.axe_c

        self.mac = self.span*(self.axe_c**2+self.tip_c**2+self.axe_c*self.tip_c)/(3.*self.area)
        y_mac = y_tip**2*(2*self.tip_c+self.axe_c)/(3*self.area)
        z_mac = z_tip**2*(2*self.tip_c+self.axe_c)/(3*self.area)
        x_tip_local = 0.25*(self.axe_c-self.tip_c) + y_tip*np.tan(self.sweep25)
        x_mac_local = y_tip*x_tip_local*(self.tip_c*2.+self.axe_c)/(3.*self.area)

        htp_x_wise_anchor = 0.85
        x_axe = body_length*(1-body_cone_length/body_length*(1-htp_x_wise_anchor)) - self.axe_c

        x_tip = x_axe + x_tip_local
        x_mac = x_axe + x_mac_local

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_mac_loc[0] + 0.25*wing_mac)

        self.axe_loc = np.array([x_axe, y_axe, z_axe])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = self.axe_loc

        self.gross_wet_area = 1.63*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_aero_data(self):
        """WARNING : output values are in Wing reference area
        """
        wing_area = self.aircraft.airframe.wing.area
        # Helmbold formula corrected with endplate effect
        cza_htp =  1.2*(np.pi*self.aspect_ratio)/(1+np.sqrt(1+(self.aspect_ratio/2)**2))*(self.area/wing_area)
        xlc_htp = self.mac_loc[0] + 0.25*self.mac   # Position of HTP center of lift
        aoa_max_htp = unit.rad_deg(9.)              # Maximum angle of attack allowed for HTP
        ki_htp = 1.1/(np.pi*self.aspect_ratio)      # HTP induced drag coefficient
        return cza_htp, xlc_htp, aoa_max_htp, ki_htp

    def eval_mass(self):
        self.mass = self.mass_correction_factor * 22. * self.area
        self.cg = self.mac_loc + 0.20*np.array([self.mac, 0., 0.])

    def eval_area(self):
        wing_area = self.aircraft.airframe.wing.area
        wing_mac = self.aircraft.airframe.wing.mac

        self.area = self.volume_factor*(wing_area*wing_mac/self.lever_arm)


class TankWingBox(Tank):

    def __init__(self, aircraft):
        super(TankWingBox, self).__init__(aircraft)

        self.gravimetric_index = aircraft.get_init(self,"gravimetric_index")
        self.volumetric_index = aircraft.get_init(self,"volumetric_index")

        self.fuel_pressure = aircraft.get_init(self,"fuel_pressure", val=self.fuel_over_pressure(aircraft))
        self.fuel_density = None

        self.cantilever_gross_volume = None
        self.central_gross_volume = None
        self.gross_volume = None
        self.cantilever_volume = None
        self.central_volume = None

        self.shell_specific_volume = None
        self.max_volume = None
        self.mfw_volume_limited = None

        self.fuel_max_fwd_cg = np.full(3,None)
        self.fuel_max_fwd_mass = None

        self.fuel_max_bwd_cg = np.full(3,None)
        self.fuel_max_bwd_mass = None

    def get_component_type(self):
        return "wing_box_tank"

    def eval_geometry(self):
        body_width = self.aircraft.airframe.body.width
        wing_area = self.aircraft.airframe.wing.area
        wing_mac = self.aircraft.airframe.wing.mac
        wing_root_loc = self.aircraft.airframe.wing.root_loc
        wing_root_toc = self.aircraft.airframe.wing.root_toc
        wing_root_c = self.aircraft.airframe.wing.root_c
        wing_kink_loc = self.aircraft.airframe.wing.kink_loc
        wing_kink_toc = self.aircraft.airframe.wing.kink_toc
        wing_kink_c = self.aircraft.airframe.wing.kink_c
        wing_tip_loc = self.aircraft.airframe.wing.tip_loc
        wing_tip_toc = self.aircraft.airframe.wing.tip_toc
        wing_tip_c = self.aircraft.airframe.wing.tip_c
        wing_fsr = self.aircraft.airframe.wing.front_spar_ratio
        wing_rsr = self.aircraft.airframe.wing.rear_spar_ratio
        fuel_type = self.aircraft.arrangement.fuel_type


        root_sec = (wing_rsr - wing_fsr)*wing_root_toc*wing_root_c**2
        kink_sec = (wing_rsr - wing_fsr)*wing_kink_toc*wing_kink_c**2
        tip_sec = (wing_rsr - wing_fsr)*wing_tip_toc*wing_tip_c**2

        self.cantilever_gross_volume = (2./3.)*(
            0.9*(wing_kink_loc[1]-wing_root_loc[1])*(root_sec+kink_sec+np.sqrt(root_sec*kink_sec))
          + 0.7*(wing_tip_loc[1]-wing_kink_loc[1])*(kink_sec+tip_sec+np.sqrt(kink_sec*tip_sec)))
        self.cantilever_volume = self.cantilever_gross_volume*(1.-self.volumetric_index)

        self.central_gross_volume = 0.8*(wing_rsr - wing_fsr) * body_width * wing_root_toc * wing_root_c**2
        self.central_volume = self.central_gross_volume*(1.-self.volumetric_index)

        self.gross_volume = self.central_gross_volume + self.cantilever_gross_volume

        if self.aircraft.arrangement.fuel_type in ["liquid_h2","compressed_h2"]:
            # Volume of the tank structure
            self.shell_specific_volume = self.gross_volume*(1.-self.volumetric_index)
            # Volume of the fuel
            self.max_volume = self.gross_volume*self.volumetric_index

        else:
            self.shell_specific_volume = 0.
            self.max_volume = self.gross_volume

        self.frame_origin = [wing_root_loc[0], 0., wing_root_loc[2]]

    def sketch_3view(self):
        return None

    def eval_mass(self):
        fuel_type = self.aircraft.arrangement.fuel_type
        wing_root_c = self.aircraft.airframe.wing.root_c
        wing_root_loc = self.aircraft.airframe.wing.root_loc
        wing_kink_c = self.aircraft.airframe.wing.kink_c
        wing_kink_loc = self.aircraft.airframe.wing.kink_loc
        wing_tip_c = self.aircraft.airframe.wing.tip_c
        wing_tip_loc = self.aircraft.airframe.wing.tip_loc

        self.fuel_cantilever_cg =  0.25*(wing_root_loc + 0.40*np.array([wing_root_c, 0., 0.])) \
                                  + 0.65*(wing_kink_loc + 0.40*np.array([wing_kink_c, 0., 0.])) \
                                  + 0.10*(wing_tip_loc + 0.40*np.array([wing_tip_c, 0., 0.]))

        self.fuel_central_cg = wing_root_loc + 0.40*np.array([wing_root_c, 0., 0.])

        self.fuel_total_cg = (  self.fuel_central_cg*self.central_volume
                              + self.fuel_cantilever_cg*self.cantilever_volume
                              ) / (self.central_volume + self.cantilever_volume)

        # REMARK : if fuel_type is "Battery", fuel density will be battery density
        self.fuel_density = earth.fuel_density(fuel_type, self.fuel_pressure)
        self.mfw_volume_limited = self.max_volume*self.fuel_density

        # Tank equiped structural mass
        if self.aircraft.arrangement.fuel_type in ["liquid_h2","compressed_h2"]:
            self.shell_specific_mass = self.gross_volume*(1./self.gravimetric_index-1.)*self.volumetric_index*self.fuel_density
        else:
            self.shell_specific_mass = 0.

        self.mass = self.shell_specific_mass
        self.cg = self.fuel_total_cg

        self.fuel_max_fwd_cg = self.fuel_central_cg    # Fuel max forward CG, central tank is forward only within backward swept wing
        self.fuel_max_fwd_mass = self.central_volume*self.fuel_density

        self.fuel_max_bwd_cg = self.fuel_cantilever_cg    # Fuel max Backward CG
        self.fuel_max_bwd_mass = self.cantilever_volume*self.fuel_density

class TankFuselageFloor(Tank):

    def __init__(self, aircraft):
        super(TankFuselageFloor, self).__init__(aircraft)

        self.gravimetric_index = aircraft.get_init(self,"gravimetric_index")
        self.volumetric_index = aircraft.get_init(self,"volumetric_index")

        self.length = None
        self.width = None
        self.height = None

        self.fuel_pressure = aircraft.get_init(self,"fuel_pressure", val=self.fuel_over_pressure(aircraft))
        self.fuel_density = None

        self.gross_volume = None
        self.shell_specific_volume = None
        self.shell_specific_mass = None

        self.max_volume = None
        self.mfw_volume_limited = None

        self.fuel_max_fwd_cg = np.full(3,None)
        self.fuel_max_fwd_mass = None

        self.fuel_max_bwd_cg = np.full(3,None)
        self.fuel_max_bwd_mass = None

    def get_component_type(self):
        return "rear_body_tank"

    def eval_geometry(self):
        body_loc = self.aircraft.airframe.body.frame_origin
        body_width = self.aircraft.airframe.body.width
        body_height = self.aircraft.airframe.body.height
        body_length = self.aircraft.airframe.body.length
        forward_ratio = self.aircraft.airframe.body.forward_ratio
        body_wall_thickness = self.aircraft.airframe.body.wall_thickness

        x_axe = body_loc[0] + forward_ratio*body_width
        y_axe = 0.
        z_axe = body_loc[2] + 0.05*body_height

        self.frame_origin = [x_axe, y_axe, z_axe]

        self.length = 0.20 * body_length
        self.width = body_width - 2.*body_wall_thickness
        self.height = 0.10 * body_height

        self.gross_volume = 0.85 * (self.height * self.width * self.length)

        if self.aircraft.arrangement.fuel_type in ["liquid_h2","compressed_h2"]:
            # Volume of the tank structure
            self.shell_specific_volume = self.gross_volume*(1.-self.volumetric_index)
            # Volume of the fuel
            self.max_volume = self.gross_volume*self.volumetric_index

        else:
            self.shell_specific_volume = 0.
            self.max_volume = self.gross_volume

    def sketch_3view(self):
        return None

    def eval_mass(self):
        fuel_type = self.aircraft.arrangement.fuel_type
        body_width = self.aircraft.airframe.body.width
        body_tail_cone_ratio = self.aircraft.airframe.body.tail_cone_ratio
        body_rear_bulkhead_ratio = self.aircraft.airframe.body.rear_bulkhead_ratio

        # REMARK : if fuel is "Battery", fuel density will be battery density
        self.fuel_density = earth.fuel_density(fuel_type, self.fuel_pressure)
        self.mfw_volume_limited = self.max_volume*self.fuel_density

        # Tank equiped structural mass
        if self.aircraft.arrangement.fuel_type in ["liquid_h2","compressed_h2"]:
            self.shell_specific_mass = self.gross_volume*(1./self.gravimetric_index-1.)*self.volumetric_index*self.fuel_density
        else:
            self.shell_specific_mass = 0.

        self.mass = self.shell_specific_mass

        self.cg = self.frame_origin[0] + 0.5 * self.length

        self.fuel_max_fwd_cg = self.cg    # Fuel max Forward CG
        self.fuel_max_fwd_mass = self.max_volume*self.fuel_density

        self.fuel_max_bwd_cg = self.cg    # Fuel max Backward CG
        self.fuel_max_bwd_mass = self.max_volume*self.fuel_density

class TankRearFuselage(Tank):

    def __init__(self, aircraft):
        super(TankRearFuselage, self).__init__(aircraft)

        self.gravimetric_index = aircraft.get_init(self,"gravimetric_index")
        self.volumetric_index = aircraft.get_init(self,"volumetric_index")

        self.length = aircraft.get_init(self,"length")
        self.width_rear_factor = aircraft.get_init(self,"width_rear_factor")

        self.width_rear = None
        self.width_front = None

        self.fuel_pressure = aircraft.get_init(self,"fuel_pressure", val=self.fuel_over_pressure(aircraft))
        self.fuel_density = None

        self.gross_volume = None
        self.shell_specific_volume = None
        self.shell_specific_mass = None

        self.max_volume = None
        self.mfw_volume_limited = None

        self.fuel_max_fwd_cg = np.full(3,None)
        self.fuel_max_fwd_mass = None

        self.fuel_max_bwd_cg = np.full(3,None)
        self.fuel_max_bwd_mass = None

    def get_component_type(self):
        return "rear_body_tank"

    def eval_geometry(self):
        body_loc = self.aircraft.airframe.body.frame_origin
        body_width = self.aircraft.airframe.body.width
        body_length = self.aircraft.airframe.body.length
        body_wall_thickness = self.aircraft.airframe.body.wall_thickness
        body_tail_cone_ratio = self.aircraft.airframe.body.tail_cone_ratio
        body_rear_bulkhead_ratio = self.aircraft.airframe.body.rear_bulkhead_ratio

        x_axe = body_loc[0] + body_length - body_tail_cone_ratio*body_width - self.length
        y_axe = 0.
        z_axe = body_loc[2] + 0.6*body_width

        self.frame_origin = [x_axe, y_axe, z_axe]

        x = self.length/((body_tail_cone_ratio-body_rear_bulkhead_ratio)*body_width)
        self.width_front = min(1.,self.width_rear_factor+(1.-self.width_rear_factor)*x)*(body_width-2.*body_wall_thickness)
        self.width_rear = self.width_rear_factor*(body_width-2.*body_wall_thickness)

        # Tank is supposed to be composed of an eventual cylindrical part of length lcyl and a cone trunc
        lcyl = max(0.,self.length - body_width*(body_tail_cone_ratio-body_rear_bulkhead_ratio))

        self.gross_volume = 0.9 * (  0.25*np.pi*lcyl*self.width_front**2
                                   + (1./12.)*np.pi*self.length*(self.width_front**2+self.width_front*self.width_rear+self.width_rear**2))

        if self.aircraft.arrangement.fuel_type in ["liquid_h2","compressed_h2"]:
            # Volume of the tank structure
            self.shell_specific_volume = self.gross_volume*(1.-self.volumetric_index)
            # Volume of the fuel
            self.max_volume = self.gross_volume*self.volumetric_index

        else:
            self.shell_specific_volume = 0.
            self.max_volume = self.gross_volume

    def sketch_3view(self):
        return None

    def eval_mass(self):
        fuel_type = self.aircraft.arrangement.fuel_type
        body_width = self.aircraft.airframe.body.width
        body_tail_cone_ratio = self.aircraft.airframe.body.tail_cone_ratio
        body_rear_bulkhead_ratio = self.aircraft.airframe.body.rear_bulkhead_ratio

        # REMARK : if fuel is "Battery", fuel density will be battery density
        self.fuel_density = earth.fuel_density(fuel_type, self.fuel_pressure)
        self.mfw_volume_limited = self.max_volume*self.fuel_density

        # Tank equiped structural mass
        if self.aircraft.arrangement.fuel_type in ["liquid_h2","compressed_h2"]:
            self.shell_specific_mass = self.gross_volume*(1./self.gravimetric_index-1.)*self.volumetric_index*self.fuel_density
        else:
            self.shell_specific_mass = 0.

        self.mass = self.shell_specific_mass

        lcyl = max(0.,self.length - body_width*(body_tail_cone_ratio-body_rear_bulkhead_ratio))
        V = (1./12.)*self.length*np.sqrt(self.width_front**2+self.width_front*self.width_rear+self.width_rear**2)   # Tank conic part
        v = (1./12.)*self.length*np.sqrt(self.width_rear**2)                                                        # Un-used tail part
        self.cg = self.frame_origin[0] + 0.5*lcyl + 0.25*self.length*(1.+body_rear_bulkhead_ratio*body_width/self.length-3.*v/V)

        self.fuel_max_fwd_cg = self.cg    # Fuel max Forward CG
        self.fuel_max_fwd_mass = self.max_volume*self.fuel_density

        self.fuel_max_bwd_cg = self.cg    # Fuel max Backward CG
        self.fuel_max_bwd_mass = self.max_volume*self.fuel_density

class TankWingPod(Pod):

    def __init__(self, aircraft, side):
        super(TankWingPod, self).__init__(aircraft)

        self.airplane_side = side

        n_pax_ref = self.aircraft.requirement.n_pax_ref
        n_pax_front = self.aircraft.airframe.cabin.n_pax_front
        n_aisle = self.aircraft.airframe.cabin.n_aisle

        self.span_ratio = aircraft.get_init(self,"span_ratio")
        self.fuel_pressure = aircraft.get_init(self,"fuel_pressure", val=self.fuel_over_pressure(aircraft))
        self.fuel_density = None

        length = 0.35*(7.8*(0.38*n_pax_front + 1.05*n_aisle + 0.55) + 0.005*(n_pax_ref/n_pax_front)**2.25)
        width = 0.80*(0.38*n_pax_front + 1.05*n_aisle + 0.55)

        self.dry_bay_length = aircraft.get_init(self,"dry_bay_length")
        self.length = aircraft.get_init(self,"length", val=length)
        self.width = aircraft.get_init(self,"width", val=width)
        self.x_loc_ratio = aircraft.get_init(self,"x_loc_ratio")
        self.z_loc_ratio = aircraft.get_init(self,"z_loc_ratio")
        self.wing_axe_c = None
        self.wing_axe_x = None
        self.wing_axe_z = None

        self.max_volume = None
        self.mfw_volume_limited = None

        self.fuel_max_fwd_cg = np.full(3,None)
        self.fuel_max_fwd_mass = None

        self.fuel_max_bwd_cg = np.full(3,None)
        self.fuel_max_bwd_mass = None

    def get_side(self):
        return {"right":1., "left":-1.}.get(self.airplane_side)

    def get_component_type(self):
        return "wing_pod_tank"

    def eval_geometry(self):
        body_width = self.aircraft.airframe.body.width
        tank_width = self.aircraft.airframe.tank.width

        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_dihedral = self.aircraft.airframe.wing.dihedral
        wing_root_loc = self.aircraft.airframe.wing.root_loc
        wing_kink_c = self.aircraft.airframe.wing.kink_c
        wing_kink_loc = self.aircraft.airframe.wing.kink_loc
        wing_tip_c = self.aircraft.airframe.wing.tip_c
        wing_tip_loc = self.aircraft.airframe.wing.tip_loc

        lateral_margin = self.aircraft.airframe.nacelle.lateral_margin

        tan_phi0 = 0.25*(wing_kink_c-wing_tip_c)/(wing_tip_loc[1]-wing_kink_loc[1]) + np.tan(wing_sweep25)

        if (self.aircraft.arrangement.nacelle_attachment == "pods"):
            y_axe = 0.6 * body_width + (0.5 + lateral_margin)*tank_width
        else:
            y_axe = self.span_ratio * wing_tip_loc[1]

        x_axe = wing_root_loc[0] + (y_axe-wing_root_loc[1])*tan_phi0 - self.x_loc_ratio*self.length
        z_axe = wing_root_loc[2] + (y_axe-wing_root_loc[2])*np.tan(wing_dihedral) - self.z_loc_ratio*self.width

        self.frame_origin = [x_axe, y_axe*self.get_side(), z_axe]

        self.wing_axe_c = wing_kink_c - (wing_kink_c-wing_tip_c)/(wing_tip_loc[1]-wing_kink_loc[1])*(y_axe-wing_kink_loc[1])
        self.wing_axe_x = wing_kink_loc[0] - (wing_kink_loc[0]-wing_tip_loc[0])/(wing_tip_loc[1]-wing_kink_loc[1])*(y_axe-wing_kink_loc[1])
        self.wing_axe_z = wing_kink_loc[2] - (wing_kink_loc[2]-wing_tip_loc[2])/(wing_tip_loc[1]-wing_kink_loc[1])*(y_axe-wing_kink_loc[1])

        self.gross_wet_area = 0.85*3.14*self.width*self.length
        self.net_wet_area = 0.95*self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.05

        self.size_fuel_tank("external")

        self.max_volume = self.fuel_volume

    def eval_mass(self):
        fuel_type = self.aircraft.arrangement.fuel_type

        # REMARK : if fuel is "Battery", fuel density will be battery density
        self.fuel_density = earth.fuel_density(fuel_type, self.fuel_pressure)
        self.mfw_volume_limited = self.max_volume*self.fuel_density

        self.mass_fuel_tank("external")


        # Tank equiped structural mass
        if self.aircraft.arrangement.fuel_type in ["liquid_h2","compressed_h2"]:
            self.shell_specific_mass = self.external_pod_volume*(1./self.gravimetric_index-1.)*self.volumetric_index*self.fuel_density
        else:
            self.shell_specific_mass = 0.

        self.mass = self.structure_shell_mass + self.shell_specific_mass
        self.cg = self.frame_origin + 0.45*np.array([self.length, 0., 0.])

        self.fuel_max_fwd_cg = self.cg    # Fuel max Forward CG
        self.fuel_max_fwd_mass = self.max_volume*self.fuel_density

        self.fuel_max_bwd_cg = self.cg    # Fuel max Backward CG
        self.fuel_max_bwd_mass = self.max_volume*self.fuel_density

class TankPiggyBack(Pod):

    def __init__(self, aircraft):
        super(TankPiggyBack, self).__init__(aircraft)

        n_pax_ref = self.aircraft.requirement.n_pax_ref
        n_pax_front = self.aircraft.airframe.cabin.n_pax_front
        n_aisle = self.aircraft.airframe.cabin.n_aisle

        self.fuel_pressure = aircraft.get_init(self,"fuel_pressure", val=self.fuel_over_pressure(aircraft))
        self.fuel_density = None

        # Estimations based on fuselage dimension estimation
        length = 0.75*(7.8*(0.38*n_pax_front + 1.05*n_aisle + 0.55) + 0.005*(n_pax_ref/n_pax_front)**2.25)
        width = 0.80*(0.38*n_pax_front + 1.05*n_aisle + 0.55)

        self.dry_bay_length = aircraft.get_init(self,"dry_bay_length")
        self.length = aircraft.get_init(self,"length", val=length)
        self.width = aircraft.get_init(self,"width", val=width)
        self.x_loc_ratio = aircraft.get_init(self,"x_loc_ratio")
        self.z_loc_ratio = aircraft.get_init(self,"z_loc_ratio")
        self.wing_axe_c = None
        self.wing_axe_x = None
        self.wing_axe_z = None

        self.max_volume = None
        self.mfw_volume_limited = None

        self.fuel_max_fwd_cg = np.full(3,None)
        self.fuel_max_fwd_mass = None

        self.fuel_max_bwd_cg = np.full(3,None)
        self.fuel_max_bwd_mass = None

    def get_component_type(self):
        return "piggyback_tank"

    def eval_geometry(self):
        body_width = self.aircraft.airframe.body.width
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_root_c = self.aircraft.airframe.wing.root_c
        wing_root_loc = self.aircraft.airframe.wing.root_loc

        x_axe = wing_mac_loc[0] - self.x_loc_ratio*self.length
        y_axe = 0.
        z_axe = 1.07*body_width + self.z_loc_ratio*self.width

        self.frame_origin = [x_axe, y_axe, z_axe]

        self.wing_axe_c = wing_root_c
        self.wing_axe_x = wing_root_loc[0]
        self.wing_axe_z = wing_root_loc[2]

        self.gross_wet_area = 0.85*3.14*self.width*self.length
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length
        self.form_factor = 1.05

        self.size_fuel_tank("external")

        self.max_volume = self.fuel_volume

    def eval_mass(self):
        fuel_type = self.aircraft.arrangement.fuel_type

        # REMARK : if fuel is "Battery", fuel density will be battery density
        self.fuel_density = earth.fuel_density(fuel_type, self.fuel_pressure)
        self.mfw_volume_limited = self.max_volume*self.fuel_density

        self.mass_fuel_tank("external")

        self.mass = self.structure_shell_mass + self.shell_specific_mass
        self.cg = self.frame_origin[0] + 0.45*np.array([self.length, 0., 0.])

        self.fuel_max_fwd_cg = self.cg    # Fuel max Forward CG
        self.fuel_max_fwd_mass = self.max_volume*self.fuel_density

        self.fuel_max_bwd_cg = self.cg    # Fuel max Backward CG
        self.fuel_max_bwd_mass = self.max_volume*self.fuel_density

    def pylon_sketch(self):
        pyl_fle = np.array([[self.frame_origin[0] + 0.25*self.length, 0 , self.frame_origin[2] - 0.45*self.width],
                            [self.frame_origin[0] + 0.25*self.length, 0 , 0.90*self.aircraft.airframe.body.width]])

        pyl_fte = np.array([[self.frame_origin[0] + 0.30*self.length, 0 , self.frame_origin[2] - 0.45*self.width],
                            [self.frame_origin[0] + 0.30*self.length, 0 , 0.90*self.aircraft.airframe.body.width]])

        pyl_ble = np.array([[self.frame_origin[0] + 0.75*self.length, 0 , self.frame_origin[2] - 0.45*self.width],
                            [self.frame_origin[0] + 0.75*self.length, 0 , 0.90*self.aircraft.airframe.body.width]])

        pyl_bte = np.array([[self.frame_origin[0] + 0.80*self.length, 0 , self.frame_origin[2] - 0.45*self.width],
                            [self.frame_origin[0] + 0.80*self.length, 0 , 0.90*self.aircraft.airframe.body.width]])

        pyl_toc = np.array([0.15, 0.15])

        return {"fle":pyl_fle, "fte":pyl_fte, "ble":pyl_ble, "bte":pyl_bte, "toc":pyl_toc}


class RetractableLandingGear(Component):

    def __init__(self, aircraft):
        super(RetractableLandingGear, self).__init__(aircraft)
        
        self.mass_correction_factor = aircraft.get_init(self,"mass_correction_factor")

    def eval_geometry(self):
        wing_root_c = self.aircraft.airframe.wing.root_c
        wing_root_loc = self.aircraft.airframe.wing.root_loc

        self.frame_origin = wing_root_loc[0] + 0.85*wing_root_c

    def sketch_3view(self):
        return None

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        mlw = self.aircraft.weight_cg.mlw

        self.mass = (0.015*mtow**1.03 + 0.012*mlw) * self.mass_correction_factor    # Landing gears
        self.cg = self.frame_origin


class BareFixedLandingGear(Component):

    def __init__(self, aircraft):
        super(BareFixedLandingGear, self).__init__(aircraft)

        self.wheel_count = aircraft.get_init(self,"wheel_count")
        self.wheel_diameter = None
        self.wheel_width = None

        self.wheel_drag_area_factor = aircraft.get_init(self,"wheel_drag_area_factor")
        self.mass_correction_factor = aircraft.get_init(self,"mass_correction_factor")

    def eval_geometry(self):
        mtow = self.aircraft.weight_cg.mtow

        self.wheel_diameter = 0.0254 * 1.8 * (mtow/self.wheel_count)**0.25
        self.wheel_width = 0.45 * self.wheel_diameter

        self.gross_wet_area = self.wheel_diameter * self.wheel_width * (1 + self.wheel_count) # Nose wheel and its leg stands for a main wheel
        self.net_wet_area = self.gross_wet_area

        # Drag model is based on drag area
        self.form_factor = self.wheel_drag_area_factor
        self.aero_length = 0.

        wing_root_c = self.aircraft.airframe.wing.root_c
        wing_root_loc = self.aircraft.airframe.wing.root_loc

        self.frame_origin = wing_root_loc[0] + 0.80*wing_root_c

    def sketch_3view(self):
        return None

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        mlw = self.aircraft.weight_cg.mlw

        self.mass = (0.02*mtow**1.03 + 0.012*mlw) * self.mass_correction_factor    # Landing gears
        self.cg = self.frame_origin


