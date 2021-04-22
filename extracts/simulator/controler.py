#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np

import time
import pygame


class MyJoystick():

    def __init__(self):
        pygame.init()

        # Count the connected joysticks
        joystick_count = pygame.joystick.get_count()
        if joystick_count==1:
            # Use joystick #0 and initialize it
            self.my_joystick = pygame.joystick.Joystick(0)
            self.my_joystick.init()
            self.done = False
        else:
            # No or too many joysticks !
            raise Exception("Error, one joystick must be connected")

        # Let the lib to set up
        time.sleep(0.5)

        # Capture initial positions
        event_list = pygame.event.get()
        self.dx0 = -self.my_joystick.get_axis(1)
        self.dl0 = self.my_joystick.get_axis(0)
        self.dm0 = -self.my_joystick.get_axis(2)
        self.dn0 = self.my_joystick.get_axis(4)

    def get_axis(self):

        event_list = pygame.event.get()
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         done = True

        dx = -self.my_joystick.get_axis(1) - self.dx0
        dl = self.my_joystick.get_axis(0) - self.dl0
        dm = -self.my_joystick.get_axis(2) - self.dm0
        dn = self.my_joystick.get_axis(4) - self.dn0

        return dx,dl,dm,dn



if __name__ == "__main__":

    ctrl = MyJoystick()

    while True :
        dx,dl,dm,dn = ctrl.get_axis()

        print("  dx = ","%.2f"%dx,
              "  dl = ","%.2f"%dl,
              "  dm = ","%.2f"%dm,
              "  dn = ","%.2f"%dn)

