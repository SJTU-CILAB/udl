import numpy as np
import pandas as pd
from typing import List
from layer.gridlayer import GridLayer
from layer.graphlayer import GraphLayer
from layer.pointlayer import PointLayer
import random
# from alignment import *


def grid_align(
    gridlayer: GridLayer,
    start_lat: float,
    end_lat: float,
    start_lon: float,
    end_lon: float,
    target_lat_step: float,
    target_lon_step: float,
):
    """
    Align the gridlayer to the target grid by changing the step size.

    Parameters
    --------------------
        gridlayer (GridLayer): the gridlayer to be aligned
        [start_lat, end_lat, start_lon, end_lon] (float): the range of the target grid
        target_lat_step (float): the target latitude step size
        target_lon_step (float): the target longitude step size
    """
    new_grid = GridLayer(
        gridlayer.name,
        start_lat,
        end_lat,
        start_lon,
        end_lon,
        target_lat_step,
        target_lon_step,
    )

    lat_range = np.arange(end_lat, start_lat, -target_lat_step)
    lon_range = np.arange(start_lon, end_lon, target_lon_step)

    lat_idx, lon_idx = np.meshgrid(lat_range, lon_range, indexing="ij")

    new_grid.data = gridlayer.get_value_by_grid(lat_idx, lon_idx)

    return new_grid


def grid_fusion(
    grid_layer_list: List[GridLayer],
    name=None,
    year=None,
    start_lat=None,
    end_lat=None,
    start_lon=None,
    end_lon=None,
    step_lat=None,
    step_lon=None,
    mode="stack"
):
    """
    Fuse multiple GridLayer into one GridLayer
    Return a numpy array if mode is "stack"
    else return a GridLayer

    Parameters
    --------------------
        grid_layer_list (List[GridLayer]): a list of GridLayer
        name (str, optional): name of the fused GridLayer. Defaults to None.(use the name of the first GridLayer)
        year (int, optional): year of the fused GridLayer. Defaults to None.(use the year of the first GridLayer)
        [start_lat, end_lat, start_lon, end_lon] (float, optional): the range of the fused GridLayer. Defaults to None.(use the range of all GridLayers)
        step_lat, step_lon (float, optional): the step of the fused GridLayer. Defaults to None.(use the minimum step of all GridLayers)
        mode (str): Mode of fusion. “stack”, “sum”, “avg”, “max”, “min” or “random”. Defaults to “concat”.  
    """
    assert isinstance(grid_layer_list, list), "The grid_layer_list is not a list"
    assert len(grid_layer_list) > 0, "The grid_layer_list is empty"
    assert mode in ["stack", "sum", "avg", "max", "min", "rand"], "The mode is not supported"
    if name is None:
        name = grid_layer_list[0].name
    if year is None:
        year = grid_layer_list[0].year
    if start_lat is None:
        start_lat = min([grid_layer.start_lat for grid_layer in grid_layer_list])
    if end_lat is None:
        end_lat = max([grid_layer.end_lat for grid_layer in grid_layer_list])
    if start_lon is None:
        start_lon = min([grid_layer.start_lon for grid_layer in grid_layer_list])
    if end_lon is None:
        end_lon = max([grid_layer.end_lon for grid_layer in grid_layer_list])
    if step_lat is None:
        step_lat = min([grid_layer.step_lat for grid_layer in grid_layer_list])
    if step_lon is None:
        step_lon = min([grid_layer.step_lon for grid_layer in grid_layer_list])

    for i in range(len(grid_layer_list)):
        grid_layer_list[i] = grid_align(grid_layer_list[i], start_lat, end_lat, start_lon, end_lon, step_lat, step_lon)
    if mode == "stack":
        if len(grid_layer_list) == 1:
            combine_feature = np.expand_dims(grid_layer_list[0].data, axis=0)
        else:
            combine_feature = np.stack([grid.data for grid in grid_layer_list])
        return combine_feature
    fused_grid = GridLayer(name, start_lat, end_lat, start_lon, end_lon, step_lat, step_lon, year)
    lat_count = round((end_lat - start_lat) / step_lat)
    lon_count = round((end_lon - start_lon) / step_lon)
    if mode == "rand":
        grid_data = np.empty(shape=(lat_count, lon_count), dtype=object)
        for i in range(lat_count):
            for j in range(lon_count):
                grid_data[i, j] = []
    else:
        grid_data = np.full(shape=(lat_count, lon_count), fill_value=np.nan)
    grid_data_count = np.full(shape=(lat_count, lon_count), fill_value=0)
    for grid in grid_layer_list:
        for i in range(lat_count):
            for j in range(lon_count):
                if np.isnan(grid.data[i,j]):
                    continue
                if mode !="rand" and np.isnan(grid_data[i,j]):
                    if mode == "min":
                        grid_data[i,j] = np.inf
                    elif mode == "max":
                        grid_data[i,j] = -np.inf
                    else:
                        grid_data[i,j] = 0
                if mode == "avg" or mode == "sum":
                    grid_data[i,j] += grid.data[i,j]
                elif mode == "min":
                    grid_data[i,j] = min(grid_data[i,j], grid.data[i,j])
                elif mode == "max":
                    grid_data[i,j] = max(grid_data[i,j], grid.data[i,j])
                elif mode == "rand":
                    grid_data[i,j].append(grid.data[i,j])
                grid_data_count[i,j] += 1
    if mode == "avg":
        grid_data = grid_data / grid_data_count
    elif mode == "rand":
        for i in range(lat_count):
            for j in range(lon_count):
                grid_data[i,j] = random.choice(grid_data[i,j])
    fused_grid.construct_grid(grid_data)
    return fused_grid
                

def graph_fusion(
        graph_layer_list: List[GraphLayer], 
        name=None, 
        year=None, 
        start_lat=None, 
        end_lat=None, 
        start_lon=None, 
        end_lon=None, 
        mode="concat"
):
    """
    Fuse multiple GraphLayer into one GraphLayer

    Parameters
    --------------------
        graph_layer_list (list(GraphLayer)): List of graph layers to be fused
        name (str, optional): name of the fused GraphLayer. Defaults to None.(use the name of the first GraphLayer)
        year (int, optional): year of the fused GraphLayer. Defaults to None.(use the year of the first GraphLayer)
        [start_lat, end_lat, start_lon, end_lon] (float, optional): the range of the fused GraphLayer. Defaults to None.(use the range of all GraphLayers)
        mode (str): Mode of fusion. “concat”, “sum”, “avg”, “max”, “min” or “random”. Defaults to “concat”.  
    """

    assert isinstance(graph_layer_list, list), "The graph_layer_list is not a list"
    assert len(graph_layer_list) > 0, "The graph_layer_list is empty"
    if name is None:
        name = graph_layer_list[0].name
    if year is None:
        year = graph_layer_list[0].year
    if column_list is None:
        column_list = graph_layer_list[0].feature_name
    if start_lat is None:
        start_lat = min([min(node[1]["lat"] for node in graph_data.data.nodes) for graph_data in graph_layer_list])
    if end_lat is None:
        end_lat = max([max(node[1]["lat"] for node in graph_data.data.nodes) for graph_data in graph_layer_list])
    if start_lon is None:
        start_lon = min([min(node[1]["lon"] for node in graph_data.data.nodes) for graph_data in graph_layer_list])
    if end_lon is None:
        end_lon = max([max(node[1]["lon"] for node in graph_data.data.nodes) for graph_data in graph_layer_list])
    fused_graph_data = GraphLayer(name,year)
    for node in graph_layer_list[0].data.nodes:
        node_attr = []
        if len(node[1])>2:
            node_attr.append(node[1][graph_layer_list[0].name])
        for graph in graph_layer_list[1:]:
            for node2 in graph.data.nodes:
                if node[1]['lat']==node2[1]['lat'] and node[1]['lon']==node2[1]['lon'] and len(node2[1])>2:
                    node_attr.append(node2[1][graph.name])
                    break
        if len(node_attr)!=0:
            fused_graph_node = fused_graph_data.construct_node(node[0],node[1]['lat'],node[1]['lon'],node_attr)
            fused_graph_data.add_node(fused_graph_node)
        else: fused_graph_data.add_node(node)
    for edge in graph_layer_list[0].data.edges:
        edge_attr = []
        if len(edge)>2 :
            edge_attr.append(edge[2]['edge_attribute'])
        for graph in graph_layer_list[1:]:
            for edge2 in graph.data.edges:
                if (edge[0]==edge2[0] and edge[0]==edge2[0]) or (edge[0]==edge2[1] and edge[1]==edge2[0]):
                    if len(edge2)>2:
                        edge_attr.append(edge2[2]['edge_attribute'])
                        break
        if len(node_attr)==0:
            fused_graph_data.add_edge(edge)
        else:
            fused_graph_edge = fused_graph_data.construct_edge(edge[0],edge[1],edge_weight=edge_attr)
            fused_graph_data.add_edge(fused_graph_edge)
    return fused_graph_data     
            
   

def point_fusion(
    point_layer_list: List[PointLayer],
    name=None,
    year=None,
    column_list=None,
    start_lat=None,
    end_lat=None,
    start_lon=None,
    end_lon=None,
):
    """
    Fuse multiple PointLayer into one PointLayer

    Parameters
    --------------------
        point_layer_list (List[PointLayer]): a list of PointLayer
        name (str, optional): name of the fused PointLayer. Defaults to None.(use the name of the first PointLayer)
        year (int, optional): year of the fused PointLayer. Defaults to None.(use the year of the first PointLayer)
        column_list (List[str], optional): the columns to be fused. Defaults to None.(use the columns of the first PointLayer)
        [start_lat, end_lat, start_lon, end_lon] (float, optional): the range of the fused PointLayer. Defaults to None.(use the range of all PointLayers)
    """
    
    assert isinstance(point_layer_list, list), "The point_layer_list is not a list"
    assert len(point_layer_list) > 0, "The point_layer_list is empty"
    if name is None:
        name = point_layer_list[0].name
    if year is None:
        year = point_layer_list[0].year
    if column_list is None:
        column_list = point_layer_list[0].feature_name
    if start_lat is None:
        start_lat = min([min(point_data.data["lat"]) for point_data in point_layer_list])
    if end_lat is None:
        end_lat = max([max(point_data.data["lat"]) for point_data in point_layer_list])
    if start_lon is None:
        start_lon = min([min(point_data.data["lon"]) for point_data in point_layer_list])
    if end_lon is None:
        end_lon = max([max(point_data.data["lon"]) for point_data in point_layer_list])
    fused_point_data = PointLayer(name, year, column_list)
    for point_data in point_layer_list:
        fused_point_data.add_points(point_data.get_value_by_range(start_lat, end_lat, start_lon, end_lon, column_list))
    return fused_point_data

def grid_point_fusion(
    grid_layer_list: List[GridLayer],
    density: int,
    name=None,
    year=None,
    start_lat=None,
    end_lat=None,
    start_lon=None,
    end_lon=None,
    step_lat=None,
    step_lon=None,
    random_seed=21,
):
    '''
    Fuse multiple GridLayers into one PointLayer
    For each GridLayer, there will be points randomly spreaded according to the density times the value of the corresponding grid cell

    Parameters
    --------------------
        grid_layer_list (List[GridLayer]): a list of GridLayer
        name (str, optional): name of the fused PointLayer. Defaults to None.(use the name of the first GridLayer)
        year (int, optional): year of the fused PointLayer. Defaults to None.(use the year of the first GridLayer)
        [start_lat, end_lat, start_lon, end_lon] (float, optional): the range of the fused PointLayer. Defaults to None.(use the range of all GridLayers)
        step_lat, step_lon (float, optional): the step of the fused PointLayer. Defaults to None.(use the minimum step of all GridLayers)
        density (int): the relative density of the fused PointLayer
        random_seed (int, optional): the random seed. Defaults to 21.
    ''' 
    assert isinstance(grid_layer_list, list), "The grid_layer_list is not a list"
    assert len(grid_layer_list) > 0, "The grid_layer_list is empty"
    if name is None:
        name = grid_layer_list[0].name
    if year is None:
        year = grid_layer_list[0].year
    if start_lat is None:
        start_lat = min([grid_layer.start_lat for grid_layer in grid_layer_list])
    if end_lat is None:
        end_lat = max([grid_layer.end_lat for grid_layer in grid_layer_list])
    if start_lon is None:
        start_lon = min([grid_layer.start_lon for grid_layer in grid_layer_list])
    if end_lon is None:
        end_lon = max([grid_layer.end_lon for grid_layer in grid_layer_list])
    if step_lat is None:
        step_lat = min([grid_layer.step_lat for grid_layer in grid_layer_list])
    if step_lon is None:
        step_lon = min([grid_layer.step_lon for grid_layer in grid_layer_list])

    fused_point_data = PointLayer(name, year)
    for grid in grid_layer_list:
        high_node = grid.data[~np.isnan(grid.data)].max() / density
        lat_count = round((end_lat - start_lat) / step_lat)
        lon_count = round((end_lon - start_lon) / step_lon)
        grid_data = grid.data

        lats = np.arange(start_lat, end_lat, step_lat)
        lons = np.arange(end_lon, start_lon, -step_lon)
        random.seed(random_seed)
        new_points = []
        for i in range(lat_count-1):
            for j in range(lon_count-1):
                if np.isnan(grid_data[i,j]) or grid_data[i,j] == 0:
                    continue
                lat, lon = lats[i],lons[j]
                lat_, lon_ = lats[i+1],lons[j+1]
                for k in range(int(grid_data[i,j]/high_node)):
                    new_points.append((random.uniform(lon, lon_), random.uniform(lat, lat_)))
        fused_point_data.add_points(new_points)
    return fused_point_data
        
