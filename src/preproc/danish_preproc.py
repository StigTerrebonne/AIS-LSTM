#!/usr/bin/env python

"""
Title: danish_preproc.py
Objective: Clean sorted data to prepare for training
Creator: Stig Terrebonne, Michael Doyle
Date: July 25th, 2018
"""

import numpy as np

sorted_data = np.load('../../data/interim/danish_sorted_data.npz')
df = sorted_data['sorted_data']

# take out first column
second_column = df[:, 1]

# find number of batches
batch_count = 0
unique_vals, unique_count = np.unique(second_column, return_counts=True)
for i in unique_count:
    if i >= 5:
        batch_count += (i - 4)

# create empty x,y train
x_train = np.empty([batch_count, 5, 6])
y_train = np.empty([batch_count, 6])

first_lat = []
first_long = []
k = 0

# make data set into relative latitudes and longitudes
for i in range(len(unique_count)):
    first_lat.append(df[k][2])
    first_long.append(df[k][3])
    k += unique_count[i]

k = 0
last_count = 0
for i in range(len(df)):
    df[i, 2] = df[i, 2] - first_lat[k]
    df[i, 3] = df[i, 3] - first_long[k]
    if (last_count - i) == unique_count[k]:
        last_count = i
        k += 1

# Passing the Lats and Longs to make them absolute in post-processing
def firsts():
    lats = first_lat
    longs = first_long
    unique = unique_count
    return [lats, longs, unique]

# normalize dataset
lat_min = -90
lat_max = 90
long_min = -180
long_max = 180
speed_min = 0
speed_max = np.amax(df[:, [4]])
time_min = 0
time_max = np.amax(df[:, [0]])
course_min = 0
course_max = np.max(df[:, [5]])

# change data from 0 to 1
# df[:, [2]] = ((df[:, [2]] - lat_min)/(lat_max - lat_min))
# df[:, [3]] = ((df[:, [3]] - long_min)/(long_max - long_min))
df[:, [0]] = (df[:, [0]] - time_min) / (time_max - time_min)
df[:, [4]] = (df[:, [4]] - speed_min) / (speed_max - speed_min)
df[:, [5]] = (df[:, [5]] - course_min) / (course_max - course_min)

# fill batches
i = 0

for count, k in enumerate(second_column):
    try:
        if second_column[count + 5] == k:
            x_train[i][0][:] = df[count][:]
            x_train[i][1][:] = df[count + 1][:]
            x_train[i][2][:] = df[count + 2][:]
            x_train[i][3][:] = df[count + 3][:]
            x_train[i][4][:] = df[count + 4][:]

            y_train[i][:] = df[count + 5][:]
            i += 1
    except:
        pass

# take out name feature column, cut to size of i above
x_train = x_train[0:i - 1, :, [0, 2, 3, 4, 5]]
y_train = y_train[0:i - 1, [2, 3]]

# splice data for training and testing
train_length = int(i * 0.8)
y_test = y_train[train_length:, :]
x_test = x_train[train_length:, :, :]

y_train = y_train[:train_length + 1, :]
x_train = x_train[:train_length + 1, :, :]

np.savez('../../data/processed/danish_train_test_data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

print('----------------------------------------------------')
print('End of danish_preproc.py')
print('----------------------------------------------------')

