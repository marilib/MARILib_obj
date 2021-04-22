#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
import pickle as pickle


def to_binary_file(obj,filename):
    """Save the obj as a binary file .pkl

    :param obj: the object to save
    :param filename: the path
    :return: None
    """
    try:  # Add .pkl extension if not specified
        last_point_position = filename.rindex(r'\.')
        filename = filename[:last_point_position]+".pkl"
    except ValueError:  # pattern not found
        filename = filename + ".pkl"

    with open(filename,'wb') as f:
        pickle.dump(obj,f)
    return


def from_binary_file(filename):
    """Load a .pkl file as a python object

    :param filename: the binary filepath
    :return: the object
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj


def get_section(length, toc):

    nose2 = np.array([[ 0.0000 , 0.0000 ,  0.0000 ] ,
                      [ 0.0050 , 0.0335 , -0.0335 ] ,
                      [ 0.0191 , 0.0646 , -0.0646 ] ,
                      [ 0.0624 , 0.1196 , -0.1196 ] ,
                      [ 0.1355 , 0.1878 , -0.1878 ] ,
                      [ 0.1922 , 0.2297 , -0.2297 ] ,
                      [ 0.2773 , 0.2859 , -0.2859 ] ,
                      [ 0.4191 , 0.3624 , -0.3624 ] ,
                      [ 0.5610 , 0.4211 , -0.4211 ] ,
                      [ 0.7738 , 0.4761 , -0.4761 ] ,
                      [ 0.9156 , 0.4976 , -0.4976 ] ,
                      [ 1.0000 , 0.5000 , -0.5000 ]])

    cone2 = np.array([[ 0.0000 , 0.5000 , -0.5000 ] ,
                      [ 0.0213 , 0.5000 , -0.5000 ] ,
                      [ 0.0638 , 0.4956 , -0.4956 ] ,
                      [ 0.1064 , 0.4875 , -0.4875 ] ,
                      [ 0.1489 , 0.4794 , -0.4794 ] ,
                      [ 0.1915 , 0.4720 , -0.4720 ] ,
                      [ 0.2766 , 0.4566 , -0.4566 ] ,
                      [ 0.3617 , 0.4330 , -0.4330 ] ,
                      [ 0.4894 , 0.3822 , -0.3822 ] ,
                      [ 0.6170 , 0.3240 , -0.3240 ] ,
                      [ 0.7447 , 0.2577 , -0.2577 ] ,
                      [ 0.8723 , 0.1834 , -0.1834 ] ,
                      [ 0.8936 , 0.1679 , -0.1679 ] ,
                      [ 0.9149 , 0.1524 , -0.1524 ] ,
                      [ 0.9362 , 0.1333 , -0.1333 ] ,
                      [ 0.9574 , 0.1097 , -0.1097 ] ,
                      [ 0.9787 , 0.0788 , -0.0788 ] ,
                      [ 0.9894 , 0.0589 , -0.0589 ] ,
                      [ 1.0000 , 0.0162 , -0.0162 ]])

    r_nose = 0.15       # Leading edga evolutive part
    r_cone = 0.35       # Trailing edge evolutive part

    width = length * toc

    leading_edge_xy = np.stack([nose2[0:,0]*length*r_nose , nose2[0:,1]*width , nose2[0:,2]*width], axis=1)
    trailing_edge_xy = np.stack([(1-r_cone)*length + cone2[0:,0]*length*r_cone , cone2[0:,1]*width , cone2[0:,2]*width], axis=1)
    section_xy = np.vstack([leading_edge_xy , trailing_edge_xy])

    return section_xy
