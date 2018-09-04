import sys
sys.path.append('../../src/preproc')
sys.path.append('../../models')

from danish_preproc import firsts
from math import sin, cos, sqrt, atan2, radians
from keras.models import model_from_json
from keras import optimizers
from keras import backend
import pandas as pd
import numpy as np
import glob
import os
import folium


# User Inputs
GET_LATEST_FILE = False      # if true will get the the latest model from the models folder
model_name_json = '../../models/danish_model_linear_30ep_2018-08-14-15-15.json'        # if false, specify filename here
model_name_h5 = '../../models/danish_model_linear_30ep_2018-08-14-15-15.h5'          # if false, specify filename here

# Pull Data From Numpy File
training_data = np.load('../../data/processed/danish_train_test_data.npz')
x_test = training_data['x_test']
y_test = training_data['y_test']

# Finding the most recent files
list_of_jsons = glob.glob('../../models/*.json')
list_of_jsons.sort(key=os.path.getctime, reverse=True)

list_of_h5 = glob.glob('../../models/*.h5')
list_of_h5.sort(key=os.path.getctime, reverse=True)

# Load json and create model
for i in range(len(list_of_jsons)):
    if GET_LATEST_FILE and 'danish' in str(list_of_jsons[i]):
        json_file = open(list_of_jsons[i], 'r')
        break
else:
    json_file = open(model_name_json, 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
for i in range(len(list_of_h5)):
    if GET_LATEST_FILE and 'danish' in str(list_of_h5[i]):
        loaded_model.load_weights(list_of_h5[i])
        print("Loaded most recent model from disk: \n", str(list_of_h5[i]))
        break
else:
    loaded_model.load_weights(model_name_h5)
    print("Loaded model based on user input: \n", model_name_h5)


# Function Definitions
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# Get distance between pairs of lat-lon points (in meters)
def distance(lat1, lon1, lat2, lon2):
    r = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    dist = r*c*1000

    return dist


# Custom adam optimizer
adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# evaluate loaded model on test data
loaded_model.compile(loss='mse',
                     optimizer=adam,
                     metrics=[rmse])

# Predict Outputs
prediction = loaded_model.predict(x_test)

# Post Processing
firsts = firsts()
first_lat = firsts[0]
first_long = firsts[1]
unique_count = firsts[2]

# Adding lats and longs back to give actual predictions
k = 0
last_count = 0
for i in range(len(y_test)):
    prediction[i, 0] = prediction[i, 0] + first_lat[k]
    prediction[i, 1] = prediction[i, 1] + first_long[k]
    y_test[i, 0] = y_test[i, 0] + first_lat[k]
    y_test[i, 1] = y_test[i, 1] + first_long[k]
    if (last_count - i) == unique_count[k]:
        last_count = i
        k += 1

# Determining average distance between prediction and y_test
df_lls = pd.DataFrame({'lat1': prediction[:, 0], 'lon1': prediction[:, 1], 'lat2': y_test[:, 0], 'lon2': y_test[:, 1]})

dist = np.empty(len(df_lls['lat1']))

for i in range(dist.size):
    dist[i] = distance(df_lls['lat1'][i],
                          df_lls['lon1'][i],
                          df_lls['lat2'][i],
                          df_lls['lon2'][i])

# Find the average distance in meters
#  avg_dist = np.mean(dist) Maybe work
nine_sort = np.sort(dist)
avg_dist = nine_sort[int(0.9*len(dist))]  # currently the bottom 90% distances of sorted distances

print('----------------------------------------------------')
print('Average Distance (m): ', avg_dist, ' m')
print('Average Distance (km): ', avg_dist / 1000, ' km')
print('Average Distance (NM): ', avg_dist * 0.00053996, 'NM')
print('----------------------------------------------------')
print('End of danish_predict.py')
print('----------------------------------------------------')

# graph AOU

location_index = 50

center = [prediction[location_index, 0], prediction[location_index, 1]]

m = folium.Map(location=center, tiles="Stamen Toner", zoom_start=12)
'''
# Real Location
folium.Circle(
    radius=20,
    location=[y_test[location_index, 0], y_test[location_index, 1]],
    popup='Real Location',
    color='crimson',
    fill=True,
    fill_color='#ffcccb'
).add_to(m)

# AOU
folium.Circle(
    location=[prediction[location_index, 0], prediction[location_index, 1]],
    radius=avg_dist,  # might want to take largest distance!
    popup='AOU Radius: ' + str(avg_dist) + ' meters',
    color='#3186cc',
    fill=True,
    fill_color='#3186cc'
).add_to(m)
'''
size, index = prediction.shape
for i in range(size):
    folium.Circle(
        radius=20,
        location=[y_test[i, 0], y_test[i, 1]],
        popup='Real Location',
        color='crimson',
        fill=True,
        fill_color='#ffcccb'
    ).add_to(m)

    # AOU
    folium.Circle(
        location=[prediction[i, 0], prediction[i, 1]],
        radius=avg_dist,  # might want to take largest distance!
        popup='AOU Radius: ' + str(avg_dist) + ' meters',
        color='#3186cc',
        fill=True,
        fill_color='#3186cc'
    ).add_to(m)

m.save("map.html")

# TODO:
# can graph multiple prediction points and AOU's
