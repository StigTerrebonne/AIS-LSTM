#!/usr/bin/env python

"""
Title: danish_graph_tracks.py
Objective: Graph Danish Data
Creator: Stig Terrebonne
Date: July 25th, 2018
"""

import numpy as np
import folium

sorted_data = np.load('../../data/interim/danish_sorted_data.npz')
df = sorted_data['sorted_data']

test = []
temp = []

# take out first column
second_column = df[:, 1]

k = 0
for count in range(second_column.size):

    i = count
    temp = []
    begin = True
    try:
        while second_column[i] == second_column[i+1]:
            if begin:
                temp.append(tuple([df[i][2], df[i][3]]))
                begin = False

            temp.append(tuple([df[i+1][2], df[i+1][3]]))
            i += 1

        test.append(temp)
    except: pass

center = [df[i-1][2], df[i-1][3]]

m = folium.Map(location=center, tiles="Stamen Toner", zoom_start=12)

for i in range(len(test)):
    folium.PolyLine(
        test[i],
        color='crimson',
        weight=2.5
    ).add_to(m)

m.save("danish_tracks.html")
