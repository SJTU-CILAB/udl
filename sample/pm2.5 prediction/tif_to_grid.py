# Author: Tianyu Wang
import rasterio
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj import Transformer
from datalayer import *
import networkx as nx
import pickle
import pandas as pd
import time


def construct_node(layer_name, node_name, node_attribute):
    # return the list of 2-tuples of the form (node, node_attribute_dict)
    return [(label, {layer_name: attr}) for label, attr in zip(node_name, node_attribute)]


if __name__ == "__main__":
    name = 'NewYorkState_pop'
    year = 2020  # light->2016, pop->2020, pm2.5->2019
    dir = name+'.tif'
    dataset = rasterio.open(dir)
    print(dataset.meta)
    print(dataset.transform)
    T1 = time.time()
    # Statistics for each band
    band1 = dataset.read(1)
    # print(band1)
    pos = np.where((band1 >= 0) & (band1 < 65535))
    real_pos = rasterio.transform.xy(dataset.transform, pos[0], pos[1])
    # print(real_pos)
    precipitation = pd.DataFrame(
        np.vstack(
            (np.matmul([dataset.width, 1], pos), real_pos, band1[pos])),
        index=['FID_precipitation_Output_Features', 'lon', 'lat', year]
    ).T
    del pos, real_pos
    # precipitation.to_csv(name+'.csv')

    # construct graph
    node_att = ['r', 'g', 'b']
    g = GraphLayer('Color')
    g.construct_graph([4, 5, 6], node_att)
    g.add_node(9, color='y')
    # print(g.nodes.data())
    g.save_graph('color_graph.pickle')

    # precipitation = pd.read_csv('precipitation.csv')
    data = precipitation.sort_values(by=['lat', 'lon'])

    # start_lat = min(precipitation['lat'])
    # end_lat = max(precipitation['lat'])
    # start_lon = min(precipitation['lon'])
    # end_lon = max(precipitation['lon'])
    ### edit start_lat, end_lat, start_lon, end_lon ###
    ### Shanghai ###
    # start_lat = 30
    # end_lat = 32.4
    # start_lon = 120
    # end_lon = 122
    ### NewYorkState ###
    start_lat = 40
    end_lat = 45
    start_lon = -80
    end_lon = -70
    ### NewYorkCity ###
    # start_lat = 40.4
    # end_lat = 41
    # start_lon = -74.3
    # end_lon = -73.7

    step_lat = 0.05
    step_lon = 0.05
    # Process discontinuous raster data
    lat_count = round((end_lat - start_lat) / step_lat)
    lon_count = round((end_lon - start_lon) / step_lon)

    # Initialize grid_data and count
    grid_data = np.full(shape=(lat_count, lon_count), fill_value=np.nan)
    count = np.full(shape=(lat_count, lon_count), fill_value=0)

    # Calculate row and column indices
    data['row_index'] = ((end_lat - data['lat']) / step_lat).astype(int)
    data['col_index'] = ((data['lon'] - start_lon) / step_lon).astype(int)

    # Filter out data that is out of range
    data = data[(data['lat'] >= start_lat) & (data['lat'] <= end_lat) & 
                (data['lon'] >= start_lon) & (data['lon'] <= end_lon)]

    # Use groupby to accumulate
    grouped = data.groupby(['row_index', 'col_index'])[year].agg(['sum', 'count'])

    # Update grid_data and count
    for index, row in grouped.iterrows():
        grid_data[index] = row['sum']
        count[index] = row['count']

    # Calculate the average value
    grid_data = grid_data / count

    lats = np.arange(start_lat, end_lat, step_lat)
    lons = np.arange(end_lon, start_lon, -step_lon)
    lat_idx = np.digitize(data['lat'], lats) - 1
    lon_idx = np.digitize(data['lon'], lons) - 1
    valid_idx = (lat_idx >= 0) & (lat_idx < lat_count) & (
        lon_idx >= 0) & (lon_idx < lon_count)
    lat_idx = lat_idx[valid_idx]
    # print(lat_idx)
    lon_idx = lon_idx[valid_idx]
    values = data.loc[valid_idx]
    # values.to_csv(name+'.csv')
    values = data.loc[valid_idx, year].to_numpy()
    grid_data[lat_idx, lon_idx] = values
        
    # lats = np.arange(start_lat, end_lat, step_lat)
    # lons = np.arange(start_lon, end_lon, step_lon)
    # lat_idx = np.digitize(data['lat'], lats) - 1
    # lon_idx = np.digitize(data['lon'], lons) - 1
    # valid_idx = (lat_idx >= 0) & (lat_idx < lat_count) & (
    #     lon_idx >= 0) & (lon_idx < lon_count)
    # lat_idx = lat_idx[valid_idx]
    # print(lat_idx)
    # lon_idx = lon_idx[valid_idx]
    # values = data.loc[valid_idx]
    # # values.to_csv(name+'.csv')
    # values = data.loc[valid_idx, year].to_numpy()
    # grid_data[lat_idx, lon_idx] = values
    T2 = time.time()
    print('time cost:', (T2-T1))
    print(grid_data)

    grid = GridLayer(name, start_lat, end_lat,
                     step_lat, start_lon, end_lon, step_lon, year)
    grid.construct_grid(grid_data)

    # print(grid.get_value(31.677, 121.158))
    # print(grid.get_value(31.678, 121.158))
    # print(grid.get_value(31.678, 121.159))
    # print(grid.get_value(31.679, 121.159))
    # print(grid.get_value(31.680, 121.162))
    grid.save_grid(name+'.pickle')
