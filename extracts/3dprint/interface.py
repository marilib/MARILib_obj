#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

from tools import from_binary_file, get_section
import matplotlib.pyplot as plt

comp = from_binary_file('aircraft_curves.pkl')

print(comp["name"])


print("-----------------------------------------")
print("Example of section call")
ch = 2.5
er = 0.12
print("Chord = ", ch, "  er = ", er)
sec = get_section(ch, er)
print("    X              Zu             Zd")
print(sec)
plt.plot(sec[:,0],sec[:,1],label="up")
plt.plot(sec[:,0],sec[:,2],label="down")
plt.legend()
plt.show()

print("-----------------------------------------")
print("number of surfaces = ", len(comp["surface"]))
for surf in comp["surface"]:
    print("-----------------------------------------")
    print("Leading edge")
    print("     X            Y            Z")
    print(surf["le"])
    print("Trailing edge")
    print("     X            Y            Z")
    print(surf["te"])
    print("Thickness over chord")
    print(surf["toc"])

print("-----------------------------------------")
print("number of bodies = ", len(comp["body"]))
for body in comp["body"]:
    print("-----------------------------------------")
    print("XZ curves")
    print("     X             Zup           Zdown")
    print(body["xz"])
    print("XY curves")
    print("     X             Yright        Yleft")
    print(body["xy"])

print("-----------------------------------------")
print("number of nacelles = ", len(comp["nacelle"]))
for nac in comp["nacelle"]:
    print("-----------------------------------------")
    print("Leading edge")
    print("     X            Y            Z")
    print(nac["le"])
    print("Trailing edge")
    print("     X            Y            Z")
    print(nac["te"])
    print("Thickness over chord")
    print(nac["toc"])

