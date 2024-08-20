import pandas as pd
import pickle
import networkx as nx
import numpy as np
from udlayer.layer import *

start_lat = 30
end_lat = 32.4
start_lon = 120
end_lon = 122
step_lat = 0.02
step_lon = 0.02
# Process discontinuous grid data
lat_count = round((end_lat - start_lat) / step_lat)
lon_count = round((end_lon - start_lon) / step_lon)
grid_data = np.full(shape=(lat_count, lon_count), fill_value=0)
data = pd.read_csv('shanghai_inter.csv')
 
# Count the number of points falling in each grid and save as a pickle file
for index, row in data.iterrows():
    lat = row['lon']
    lon = row['lat']
    if lat < start_lat or lat > end_lat or lon < start_lon or lon > end_lon:
        continue
    row_index = int((end_lat - lat) / step_lat)
    col_index = int((lon - start_lon) / step_lon)
    grid_data[row_index][col_index] += 1

grid_layer = GridLayer('Shanghai_inter', start_lat, end_lat, step_lat, start_lon, end_lon, step_lon)
grid_layer.construct_grid(grid_data)
# Convert grid_data to a pandas DataFrame and save as CSV
df = pd.DataFrame(grid_data)
df.to_csv('grid_data.csv', index=False)

grid_layer.save_grid('Shanghai_inter.pickle')

