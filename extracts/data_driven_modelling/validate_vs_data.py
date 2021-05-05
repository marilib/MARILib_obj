#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 30 08:54 2021
@author: Nicolas Peteilh
"""

import extracts.data_driven_modelling.analyse_data as anadata
import numpy as np

from marilib.utils import earth, unit
from marilib.aircraft.aircraft_root import Arrangement, Aircraft
from marilib.aircraft.requirement import Requirement
from marilib.utils.read_write import MarilibIO
from marilib.aircraft.design import process
import pandas as pd


def compute_one_ac(npax, rng, mach, sref, slst, wing_att, n_eng, n_pax_row):
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
    ac.airframe.cabin.n_pax_front = n_pax_row
    ac.power_system.reference_thrust = slst  # unit.N_kN(110.)
    ac.airframe.wing.area = sref  # 110.

    # Run MDA analysis
    # ------------------------------------------------------------------------------------------------------
    process.mda(ac, mass_mission_matching=True)  # Run an MDA on the object, with mass - mission adaptation

    VTP_area         = ac.airframe.vertical_stab.area        #- surface empennage HTP et VTP
    HTP_area         = ac.airframe.horizontal_stab.area
    engine_y_arm     = ac.airframe.nacelle.cg[1]             #- bras de levier moteur
    rotor_diameter   = ac.airframe.nacelle.width             #- diamètre nacelle
    fuselage_length  = ac.airframe.body.length                #- diamètre et longueur du fuselage
    fuselage_width   = ac.airframe.body.width                #- diamètre et longueur du fuselage
    wing_span        = ac.airframe.wing.span                 #- envergure
    owe              = ac.weight_cg.owe                      #- OWE
    mtow             = ac.weight_cg.mtow                     #- MTOW
    mlw              = ac.weight_cg.mlw                      #- MLW
    mfw              = ac.weight_cg.mfw                      #- MFW
    tofl             = ac.performance.take_off.tofl_eff      #- longueur de piste déco TOFL
    app_speed        = ac.performance.approach.app_speed_eff #- vitesse d 'approche Approach_speed

    #print(VTP_area)
    #print(HTP_area)
    #print(engine_y_arm)
    #print(rotor_diameter)
    #print(fuselage_width)
    #print(wing_span)
    #print(owe)
    #print(mtow)
    #print(mlw)
    #print(mfw)
    #print(tofl)
    #print(app_speed)

    return ac, VTP_area, HTP_area, engine_y_arm, rotor_diameter, fuselage_length, fuselage_width, wing_span, owe, mtow, mlw, mfw, tofl, app_speed


def compute_all_aircrafts_and_save(df):  # path_to_data_base="All_Data_v3.xlsx"):

    # df,un = anadata.read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    # df = df[df['name']!='A380-800'].reset_index(drop=True)
    # keep turbofan only
    df = df[df['engine_type']=='turbofan'].reset_index(drop=True)
    df = df[df['airplane_type']!='business'].reset_index(drop=True)


    list_VTP_area        = []
    list_HTP_area        = []
    list_engine_y_arm    = []
    list_rotor_diameter  = []
    list_fuselage_length = []
    list_fuselage_width  = []
    list_wing_span       = []
    list_owe             = []
    list_mtow            = []
    list_mlw             = []
    list_mfw             = []
    list_tofl            = []
    list_app_speed       = []
    list_valid           = [] # is valid computation ?

    errors = []
    for index, row in df.iterrows():
        print("%s, npax_front = %d" %(row['name'], row['n_pax_front_estimate']) )
        try:
            myac, vtp_area, htp_area, y_ext, nac_width, fuse_length, fuse_width, \
            wing_span, owe, mtow, mlw, mfw, tofl, app_speed\
                = compute_one_ac(npax      = row['n_pax'        ],
                                 rng       = row['nominal_range'],
                                 mach      = row['max_speed'    ],
                                 sref      = row['wing_area'    ],
                                 slst      = row['max_thrust'   ],
                                 wing_att  = row['wing_position'],
                                 n_eng     = row['n_engine'     ],
                                 n_pax_row = row['n_pax_front_estimate'    ])
            valid = True

        except Exception as e:
            print(e)
            vtp_area   = None
            htp_area   = None
            y_ext      = None
            nac_width  = None
            fuse_length= None
            fuse_width = None
            wing_span  = None
            owe        = None
            mtow       = None
            mlw        = None
            mfw        = None
            tofl       = None
            app_speed  = None
            valid      = False   # not valid computation, error was raised
            errors.append(row['name'])


        list_VTP_area       .append(vtp_area   )
        list_HTP_area       .append(htp_area   )
        list_engine_y_arm   .append(y_ext      )
        list_rotor_diameter .append(nac_width  )
        list_fuselage_length.append(fuse_length)
        list_fuselage_width .append(fuse_width )
        list_wing_span      .append(wing_span  )
        list_owe            .append(owe        )
        list_mtow           .append(mtow       )
        list_mlw            .append(mlw        )
        list_mfw            .append(mfw        )
        list_tofl           .append(tofl       )
        list_app_speed      .append(app_speed  )
        list_valid          .append(valid)

    df["model_VTP_area"]       = list_VTP_area
    df["model_HTP_area"]       = list_HTP_area
    df["model_engine_y_arm"]   = list_engine_y_arm
    df["model_rotor_diameter"] = list_rotor_diameter
    df["model_total_length"]   = list_fuselage_length
    df["model_fuselage_width"] = list_fuselage_width
    df["model_wing_span"]      = list_wing_span
    df["model_OWE"]            = list_owe
    df["model_MTOW"]           = list_mtow
    df["model_MLW"]            = list_mlw
    df["model_max_fuel"]       = list_mfw
    df["model_tofl"]           = list_tofl
    df["model_approach_speed"] = list_app_speed
    df["model_valid"]          = list_valid

    return errors, df

def add_npax_front(df):
    """Estimate the number of front passenger and add it to the database"""
    list_n_pax_row = []
    for index, row in df.iterrows():
        n_aisle = 1
        if row['airplane_type']=="wide_body":
            n_aisle = 2
        n_pax_row = np.floor((row['fuselage_width']-0.4)/0.5) - n_aisle
        list_n_pax_row.append(n_pax_row)

    df["n_pax_front_estimate"] = list_n_pax_row
    return df

if __name__=="__main__":

    df,un = anadata.read_db("All_Data_v3.xlsx")     # read database
    df = add_npax_front(df)                         # add the column "n_pax_front_estimate"
    errors, df = compute_all_aircrafts_and_save(df) # compute all aircraft with marilib

    # Save files
    df.to_csv("data_and_model.csv", sep=";") # save database + model results
    with open("errors.txt","w") as f:        # save error list
        f.write(str(errors))

    # df = pd.read_csv("data_and_model.csv",sep=";",index_col=0)

    param_list = ["VTP_area"      ,
                  "HTP_area"      ,
                  "engine_y_arm"  ,
                  "rotor_diameter",
                  "total_length"  ,
                  "fuselage_width",
                  "wing_span"     ,
                  "OWE"           ,
                  "MTOW"          ,
                  "MLW"           ,
                  "max_fuel"      ,
                  "tofl"          ,
                  "approach_speed"]

    for var in param_list:
        un['model_' + var] = un[var]
        anadata.draw_reg(df, un, var, 'model_' + var, [[0, max(df[var])], [0, max(df[var])]], anadata.coloration)




