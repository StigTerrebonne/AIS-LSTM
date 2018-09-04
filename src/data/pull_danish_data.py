#!/usr/bin/env python

"""
Title: pull_danish_data.py
Objective: Pull and sort data to be preprocessed
Creator: Stig Terrebonne
Date: July 25th, 2018
"""

import pandas as pd
import numpy as np

# pull only specific columns out
# 'Timestamp', 'MMSI', 'Latitude', 'Longitude', 'SOG', 'COG'
fields = [0, 2, 3, 4, 7, 8]
n_rows = 30000              # Pulls this many rows of data, because all of it is too much
df = pd.read_csv('../../data/raw/aisdk_20180621.csv', skipinitialspace=True, usecols=fields, nrows=n_rows)

# get rid of nan rows (in speed and course) - could just set to -1
df = df.dropna()

# change dataframe to numpy array
df = df.values

# new number of rows and columns
n_rows, n_cols = df.shape

# sort by MMSI, then by time/date
df = df[np.lexsort((df[:, 0], df[:, 1]))]

# convert time to int
for i in range(n_rows):
    df[i][0] = int(df[i][0][11:13])*3600 + int(df[i][0][14:16])*60 + int(df[i][0][17:])

# create timedeltas
i = 0
while i in range(n_rows):
    end = False
    temp = []
    start = i
    try:
        while df[i+1][1] == df[i][1]:
            temp.append(df[i][0])
            i += 1
            end = True
    except: pass

    if end is True:
        temp.append(df[i][0])
        diff_array = np.diff(temp)

        df[start][0] = 0
        df[start+1:i+1, 0] = diff_array
    i += 1

np.savez('../../data/interim/danish_sorted_data.npz', sorted_data=df)

print('----------------------------------------------------')
print('End of danish_pull_data.py')
print('----------------------------------------------------')
