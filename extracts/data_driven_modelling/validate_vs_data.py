#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 30 08:54 2021
@author: Nicolas Peteilh
"""

import extracts.data_driven_modelling.analyse_data as anadata

from marilib.utils import earth, unit
from marilib.aircraft.aircraft_root import Arrangement, Aircraft
from marilib.aircraft.requirement import Requirement
from marilib.utils.read_write import MarilibIO
from marilib.aircraft.design import process


def compute_one_ac(npax, rng, mach, sref, slst, wing_att, n_eng):
    # Configure airplane arrangement
    # ---------------------------------------------------------------------------------------------------------------------
    if n_eng==2:
        nb_engine = "twin"
    elif n_eng==4:
        nb_engine = "quadri"

    agmt = Arrangement(body_type="fuselage",  # "fuselage" or "blended"
                       wing_type="classic",  # "classic" or "blended"
                       wing_attachment=wing_att,  # "low",  # "low" or "high"
                       stab_architecture="classic",  # "classic", "t_tail" or "h_tail"
                       tank_architecture="wing_box",  # "wing_box", "piggy_back" or "pods"
                       number_of_engine=nb_engine,  # "twin",  # "twin", "quadri" or "hexa"
                       nacelle_attachment="wing",  # "wing", "rear" or "pods"
                       power_architecture="tf",  # "tf", "tp", "ef", "ep", "pte", "pte", "extf", "exef"
                       power_source="fuel",  # "fuel", "battery", "fuel_cell"
                       fuel_type="kerosene")  # "kerosene", "liquid_h2", "Compressed_h2", "battery"

    reqs = Requirement(n_pax_ref=npax,
                       design_range=rng,
                       cruise_mach=mach,
                       cruise_altp=unit.m_ft(35000.))

    ac = Aircraft("This_plane")  # Instantiate an Aircraft object

    ac.factory(agmt,
               reqs)  # Configure the object according to Arrangement, WARNING : arrangement must not be changed after this line

    # Eventual update of some values
    # ------------------------------------------------------------------------------------------------------
    ac.power_system.reference_thrust = slst  # unit.N_kN(110.)
    ac.airframe.wing.area = sref  # 110.

    # Run MDA analysis
    # ------------------------------------------------------------------------------------------------------
    process.mda(ac, mass_mission_matching=True)  # Run an MDA on the object, with mass - mission adaptation

    #- surface empennage HTP et VTP
    vtp_area = ac.airframe.vertical_stab.area
    htp_area = ac.airframe.horizontal_stab.area
    #- bras de levier moteur
    y_ext = ac.airframe.nacelle.cg[1]
    #- diamètre nacelle
    nac_width = ac.airframe.nacelle.width
    #- diamètre et longueur du fuselage
    fuse_width = ac.airframe.body.width
    #- envergure
    wing_span = ac.airframe.wing.span
    #- OWE
    owe = ac.weight_cg.owe
    #- MTOW
    mtow = ac.weight_cg.mtow
    #- MLW
    mlw = ac.weight_cg.mlw
    #- MFW
    mfw = ac.weight_cg.mfw
    #- longueur de piste déco TOFL
    tofl = ac.performance.take_off.tofl_eff
    #- vitesse d 'approche Approach_speed
    app_speed = ac.performance.approach.app_speed_eff

    print(vtp_area)
    print(htp_area)
    print(y_ext)
    print(nac_width)
    print(fuse_width)
    print(wing_span)
    print(owe)
    print(mtow)
    print(mlw)
    print(mfw)
    print(tofl)
    print(app_speed)

    return ac, vtp_area, htp_area, y_ext, nac_width, fuse_width, wing_span, owe, mtow, mlw, mfw, tofl, app_speed


if __name__ == '__main__':

    # Read data
    #-------------------------------------------------------------------------------------------------------------------
    path_to_data_base = "All_Data_v3.xlsx"
    # path_to_data_base = "All_Data_v2.xlsx"

    df,un = anadata.read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    df = df[df['name']!='A380-800'].reset_index(drop=True)
    # keep turbofan only
    df = df[df['engine_type']=='turbofan'].reset_index(drop=True)
    df = df[df['airplane_type']!='business'].reset_index(drop=True)


    list_vtp_area   = []
    list_htp_area   = []
    list_y_ext      = []
    list_nac_width  = []
    list_fuse_width = []
    list_wing_span  = []
    list_owe        = []
    list_mtow       = []
    list_mlw        = []
    list_mfw        = []
    list_tofl       = []
    list_app_speed  = []

    for index, row in df.iterrows():
        try:
            myac, vtp_area, htp_area, y_ext, nac_width, fuse_width, \
            wing_span, owe, mtow, mlw, mfw, tofl, app_speed\
                = compute_one_ac(npax    = row['n_pax'        ],
                                 rng     = row['nominal_range'],
                                 mach    = row['max_speed'    ],
                                 sref    = row['wing_area'    ],
                                 slst    = row['max_thrust'   ],
                                 wing_att= row['wing_position'],
                                 n_eng   = row['n_engine'     ])
        except:
            vtp_area  = None
            htp_area  = None
            y_ext     = None
            nac_width = None
            fuse_width= None
            wing_span = None
            owe       = None
            mtow      = None
            mlw       = None
            mfw       = None
            tofl      = None
            app_speed = None


        list_vtp_area  .append(vtp_area  )
        list_htp_area  .append(htp_area  )
        list_y_ext     .append(y_ext     )
        list_nac_width .append(nac_width )
        list_fuse_width.append(fuse_width)
        list_wing_span .append(wing_span )
        list_owe       .append(owe       )
        list_mtow      .append(mtow      )
        list_mlw       .append(mlw       )
        list_mfw       .append(mfw       )
        list_tofl      .append(tofl      )
        list_app_speed .append(app_speed )

    df["model_vtp_area"  ] = list_vtp_area
    df["model_htp_area"  ] = list_htp_area
    df["model_y_ext"     ] = list_y_ext
    df["model_nac_width" ] = list_nac_width
    df["model_fuse_width"] = list_fuse_width
    df["model_wing_span "] = list_wing_span
    df["model_owe"       ] = list_owe
    df["model_mtow"      ] = list_mtow
    df["model_mlw"       ] = list_mlw
    df["model_mfw"       ] = list_mfw
    df["model_tofl"      ] = list_tofl
    df["model_app_speed" ] = list_app_speed
