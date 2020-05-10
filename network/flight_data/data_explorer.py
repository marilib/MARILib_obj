#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 05 09:53 2020
@author: Nicolas Peteilh, Thierry Druot, Weichang Lyu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle


def analyse_data_base(oag_file_name, range_interval, capacity_interval):
    """Analyse traffic data base and store results into various format

    :param data_base_file_name: .csv file with the traffic description
    :param range_interval: Range interval to do the analysis, ex : 200 NM
    :param capacity_interval: Capacity interval to do the analysis, ex 20 pax
    :return:
    """
    oag_df = pd.read_csv(oag_file_name)

    print("Noms des colonnes = ", oag_df.keys())
    print("Total number of lines = ", len(oag_df))
    print("Total number of flights = ", sum(oag_df['Frequency']))

    oag_df['Seats_per_flight'] = oag_df['Seats']/oag_df['Frequency']

    seats_list = np.arange(0, max(oag_df['Seats_per_flight']), capacity_interval)
    range_list = np.arange(0, max(oag_df['Distance (KM)']), range_interval)
    bins_list = [range_list, seats_list]

    # plt.hist is used only to get data (not for plotting)
    # data[0] contains the histogram values (array of array)
    # data[1] contains labels for range
    # data[2] contains labels for seat capacity
    data = plt.hist2d(oag_df['Distance (KM)'], oag_df['Seats_per_flight'],
                      weights = oag_df['Frequency'],
                      bins = bins_list)

    data_matrix = {"matrix":np.array(data[0].T), "range_step":range_interval, "npax_step":capacity_interval}
    oag_data = {"data_frame":oag_df, "range_step":range_interval, "npax_step":capacity_interval}
    return data_matrix, oag_data


def draw_matrix(file_name, data_matrix):
    """Draw the figure of the flight matrix and store it in a file

    :param file_name: file to store the figure
    :param data_matrix: data source
    :return:
    """

    range_interval = data_matrix["range_step"]
    capacity_interval = data_matrix["npax_step"]

    nc,nr = data_matrix["matrix"].shape
    range_list = [int(range_interval*j) for j in range(nr+1)]
    capa_list = [int(capacity_interval*j) for j in range(nc+1)]

    fig, ax = plt.subplots(figsize=(14, 7))

    im = ax.pcolormesh(data_matrix["matrix"],
                       edgecolors='b',
                       linewidth=0.01,
                       cmap="rainbow",
                       norm=colors.LogNorm(vmin=data_matrix["matrix"].min()+0.1, vmax=data_matrix["matrix"].max()))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlabel('Ranges (km)',
                  fontsize=16)
    ax.set_ylabel('Seat capacity',
                  fontsize=16)
    ax.xaxis.set_ticks(range(len(range_list)))
    ax.xaxis.set_ticklabels(range_list,
                            fontsize=8,
                            rotation = 'vertical')
    ax.yaxis.set_ticks(range(len(capa_list)))
    ax.yaxis.set_ticklabels(capa_list,
                            fontsize=8)
    plt.title('Number of flights per seat capacity and range',
              fontsize=16)
    cbar = fig.colorbar(im, ax=ax,
                        orientation='horizontal',
                        aspect=40.)
    plt.savefig(file_name, dpi=500, bbox_inches='tight')
    # plt.show()



def store_data_to_file(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
        return


def load_data_from_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data



if __name__ == '__main__':

    path_to_oag_file = "../input_data/2016_All_JobId651257.csv"

    range_interval = 200.   # km
    capacity_interval = 20. # npax

    data_matrix, oag_data = analyse_data_base(path_to_oag_file, range_interval, capacity_interval)

    matrix_file_name = "all_flights_2016_matrix.bin"
    store_data_to_file(matrix_file_name, data_matrix)

    dframe_file_name = "all_flights_2016_dframe.bin"
    store_data_to_file(dframe_file_name, oag_data)

    heatmap_file_name = "all_flights_2016_heatmap.png"
    draw_matrix(heatmap_file_name, data_matrix)
