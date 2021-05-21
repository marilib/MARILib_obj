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
    rotor_diameter   = ac.airframe.nacelle.fan_width             #- diamètre nacelle
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


def compute_all_aircraft(df,un):

    # Remove A380-800 row and reset index
    # df = df[df['name']!='A380-800'].reset_index(drop=True)
    # keep turbofan only
    df = df[df['engine_type']=='turbofan'].reset_index(drop=True)
    df = df[df['airplane_type']!='business'].reset_index(drop=True)

    # initialize the computed params dict
    computed_params = {}
    param_list = ["VTP_area"      ,
                  "HTP_area"      ,
                  "engine_y_arm"  ,
                  "rotor_diameter",
                  "fuselage_width",
                  "total_length"  ,
                  "wing_span"     ,
                  "OWE"           ,
                  "MTOW"          ,
                  "MLW"           ,
                  "max_fuel"      ,
                  "tofl"          ,
                  "approach_speed"]

    for k in param_list:
        computed_params[k] = [] # intialize each entry with an empty list
    computed_params['valid'] = []

    errors = {} # dict of error messages
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
            errors[row['name']]=str(e) # save aircraft name and error message in a dict

        computed_params['VTP_area']       .append(vtp_area   )
        computed_params['HTP_area']       .append(htp_area   )
        computed_params['engine_y_arm']   .append(y_ext      )
        computed_params['rotor_diameter'] .append(nac_width  )
        computed_params['fuselage_width'] .append(fuse_width )
        computed_params['total_length']   .append(fuse_length)
        computed_params['wing_span']      .append(wing_span  )
        computed_params['OWE']            .append(owe        )
        computed_params['MTOW']           .append(mtow       )
        computed_params['MLW']            .append(mlw        )
        computed_params['max_fuel']       .append(mfw        )
        computed_params['tofl']           .append(tofl       )
        computed_params['approach_speed'] .append(app_speed  )
        computed_params['valid']          .append(valid      )

    for key,val in computed_params.items(): # add 'model_' columns in the dataframe
        print(key,val[1:3])
        df['model_'+key] = val

    for p in param_list:                      # add units for the new columns
        un['model_'+p] = un[p]
    un['valid'] = 'no_dim'

    return errors, df, un


def add_npax_front(df,un):
    """Estimate the number of front passenger and add it to the database"""
    list_n_pax_row = []
    for index, row in df.iterrows():
        n_aisle = 1
        if row['airplane_type']=="wide_body":
            n_aisle = 2
        n_pax_row = np.floor((row['fuselage_width']-0.4)/0.5) - n_aisle
        list_n_pax_row.append(n_pax_row)

    df["n_pax_front_estimate"] = list_n_pax_row
    un["n_pax_front_estimate"] = 'int'
    return df,un


if __name__=="__main__":

    df,un = anadata.read_db("All_Data_v4.xlsx")     # read database
    df,un = add_npax_front(df,un)                   # add the column "n_pax_front_estimate"
    errors, df, un = compute_all_aircraft(df,un) # compute all aircraft with marilib
    # Save files
    df.to_csv("data_with_model.csv", sep=";")        # save database + model results
    un.to_csv("unit_with_model.csv", sep=";")        # save database + model results
    with open("errors.txt","w") as f:               # save error list
        f.write(str(errors))

    # df = pd.read_csv("data_with_model.csv",sep=";",index_col=0)
    # un = pd.read_csv("unit_with_model.csv",sep=";",index_col=0)

    param_list = ["fuselage_width",
                  "total_length",
                  "VTP_area",
                  "HTP_area",
                  "rotor_diameter",
                  "engine_y_arm",
                  "wing_span",
                  "OWE",
                  "MTOW",
                  "MLW",
                  "max_fuel",
                  "tofl",
                  "approach_speed"]

    for i,j in zip([0,3,6,9,11],[3,6,9,11,13]):
        anadata.subplots_by_varname(df,un,param_list[i:j],savefig=True)

    #for var in param_list:
    #    un['model_' + var] = un[var]
    #    anadata.(df, un, var, 'model_' + var, [[0, max(df[var])], [0, max(df[var])]], anadata.coloration)






