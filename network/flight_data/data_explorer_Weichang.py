#!/usr/bin/env python
# coding: utf-8

# # produce dictionary and matrix

# In[1]:


import math
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle

d = {}
new_d = {}


def analyse_data_base(data_base_file_name, range_interval, capacity_interval):
    """Analyse traffic data base and store results into various format

    :param data_base_file_name: .csv file with the traffic description
    :param range_interval: Range interval to do the analysis, ex : 200 NM
    :param capacity_interval: Capacity interval to do the analysis, ex 20 pax
    :return:
    """
    data_frame = pd.read_csv(data_base_file_name, delimiter=",", header=None)

    # data = data_frame.iloc[:,:].values             # Extract matrix of data

    data_frame1 = data_frame.drop(0)  # remove first line
    data_frame1 = data_frame1[[8,10,12]]
    data_frame1[8] = data_frame1[8].astype(int)
    data_frame1[12] = data_frame1[12].astype(int)  # change to int

    maxDis = max(data_frame1[8])  # maximun distance
    n_column = maxDis//range_interval + 1  # the number of column
    y = []
    for i in range(n_column):
        data_frame2 = data_frame1[(i*range_interval <= data_frame1[8]) & (data_frame1[8] < (i+1)*range_interval)]
        if len(data_frame2):
            n_cap = max(data_frame2[12])//capacity_interval + 1
            y.append(len(data_frame2))
            d1 = {}
            for k in range(n_cap):
                data_frame3 = data_frame2[(k*capacity_interval <= data_frame2[12]) & (data_frame2[12] < (k+1)*capacity_interval)]

                if len(data_frame3):
                    list1 = [len(data_frame3), list(set(data_frame3[10]))]
                    d1[(k+1)*capacity_interval] = list1

            d[(i+1)*range_interval] = [len(data_frame2), d1]

    for key in d:
        value = {}
        for key1 in d[key][1]:
            value[key1] = d[key][1][key1][0]
        new_d[key] = value

    stocks_dict = new_d
    df = pd.DataFrame.from_dict(stocks_dict)
    df = df.fillna(0)

    max_columns = max(df.columns)
    for i in range(range_interval, max_columns+range_interval, range_interval):
        if i not in df.columns:
            df[i] = 0.0

    max_index = max(df.index)
    for i in range(capacity_interval, max_index+capacity_interval, capacity_interval):
        if i not in df.index:
            df.loc[i] = 0.0

    df = df.astype('int')
    matrix = df.values
    array = np.array(matrix)
    return d, array, df, range_interval, capacity_interval, data_frame1


def store_dict_to_file(dictionnary_file_name, data_dictionnary, range_interval, capacity_interval):
    data_dict = {"range_step":range_interval, "capacity_step":capacity_interval, "data_dict":data_dictionnary}
    with open(dictionnary_file_name, 'wb') as f:
        pickle.dump(data_dict, f)
        return


def store_matrix_to_file(matrix_file_name, data_matrix, range_interval, capacity_interval):
    matrix = {"range_step":range_interval, "capacity_step":capacity_interval, "matrix":data_matrix}
    with open(matrix_file_name, 'wb') as f:
        pickle.dump(matrix, f)
        return


def store_dataframe_to_file(data_frame_file_name, data_frame, range_interval, capacity_interval):
    dframe = {"range_step":range_interval, "capacity_step":capacity_interval, "data_frame":data_frame}
    with open(data_frame_file_name, 'wb') as f:
        pickle.dump(dframe, f)
        return


def load_dictionnary_from_file(dictionnary_file_name):
    with open(dictionnary_file_name, 'rb') as f:
        data_dictionnary = pickle.load(f)
    return data_dictionnary


def load_matrix_from_file(matrix_file_name):
    with open(matrix_file_name, 'rb') as f:
        data_matrix = pickle.load(f)
    return data_matrix


def load_dataframe1_from_file(matrix_file_name):
    with open(matrix_file_name, 'rb') as f:
        data_frame1 = pickle.load(f)
    return data_frame1


def draw_matrix(file_name, data_matrix, range_interval, capacity_interval):
    matrix = data_matrix  # open numpy
    data = matrix[::-1] + 1
    sns.set_context({"figure.figsize": (15, 15)})
    log_norm = LogNorm(vmin=data.min(), vmax=data.max())
    cbar_ticks = [math.pow(10, i) for i in range(math.floor(
        math.log10(data.min())), math.ceil(math.log10(data.max())))]
    y_axis_labels = list(range(matrix.shape[0] * capacity_interval, 0, -capacity_interval))
    x_axis_labels = list(range(range_interval, (matrix.shape[1] + 1) * range_interval, range_interval))

    heatmap1 = sns.heatmap(data, square=True, cmap="RdBu_r", linewidths=0.3, linecolor="grey", xticklabels=x_axis_labels,
                           yticklabels=y_axis_labels, norm=log_norm, cbar_kws={"ticks": cbar_ticks, "orientation": "horizontal"})
    heatmap1.set_xlabel('Range', fontsize=15)
    heatmap1.set_ylabel('Capacity', fontsize=15)
    plt.savefig(file_name, dpi=500, bbox_inches='tight')
    return


def data_analysis(data_frame1, range_interval, capacity_interval):
    data_frame1 = data_frame1[(range_interval[0] <= data_frame1[8]) & (
            data_frame1[8] <= range_interval[1])]
    data_frame1 = data_frame1[(capacity_interval[0] <= data_frame1[12]) & (
            data_frame1[12] <= capacity_interval[1])]
    return len(data_frame1)


def get_data_types(data_frame1, range_interval, capacity_interval):
    data_frame1 = data_frame1[(range_interval[0] <= data_frame1[8]) & (
        data_frame1[8] <= range_interval[1])]
    data_frame1 = data_frame1[(capacity_interval[0] <= data_frame1[12]) & (
        data_frame1[12] <= capacity_interval[1])]
    return list(set(data_frame1[10]))



# ======================================================================================================
# Traffic analysis
# ------------------------------------------------------------------------------------------------------
file="../input_data/2019_All_JobId1448413.csv"
# file="2019_All_JobId1448413_extract.csv"
dict, matrix, df, range_interval, capacity_interval, data_frame1=analyse_data_base(file, 200, 20)

store_dict_to_file('all_flights_2019_dictionary.bin', dict, range_interval, capacity_interval)
store_matrix_to_file('all_flights_2019_matrix.bin', matrix, range_interval, capacity_interval)
store_dataframe_to_file('all_flights_2019_dataframe.bin', data_frame1, range_interval, capacity_interval)

# data_dictionnary = load_dictionnary_from_file('dictionary.bin')
# matrix = load_matrix_from_file('matrix.bin')
# data_frame1 = load_dataframe1_from_file('data_frame1.bin')

draw_matrix('all_flights_2019_heatmap.png', matrix, 200, 20)

# total = data_analysis(data_frame1, [150, 625], [0, 270])
# print(total)
#
# type_flight=get_data_types(data_frame1, [150, 625], [0, 270])
# print(type_flight)
#
# draw_matrix(matrix,200, 20)





