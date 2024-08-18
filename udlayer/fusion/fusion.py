import numpy as np
import pandas as pd
from typing import Callable, List, Optional
from ..layer.gridlayer import GridLayer
from ..layer.graphlayer import GraphLayer
from ..layer.pointlayer import PointLayer
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
    mode="concat",
    user_defined_func=None,
):
    """
    Fuse multiple GridLayer into one GridLayer
    Return a numpy array if mode is "concat"
    else return a GridLayer

    Parameters
    --------------------
        grid_layer_list (List[GridLayer]): a list of GridLayer
        name (str, optional): name of the fused GridLayer. Defaults to None.(use the name of the first GridLayer)
        year (int, optional): year of the fused GridLayer. Defaults to None.(use the year of the first GridLayer)
        [start_lat, end_lat, start_lon, end_lon] (float, optional): the range of the fused GridLayer. Defaults to None.(use the range of all GridLayers)
        step_lat, step_lon (float, optional): the step of the fused GridLayer. Defaults to None.(use the minimum step of all GridLayers)
        mode (str): Mode of fusion. "concat", "sum", "avg", "max", "min", "random", or "user_defined". Defaults to "concat".  
        user_defined_func (function, optional): the user defined function for fusion. Defaults to None. The function should take a list of values and return a single value.
    """
    assert isinstance(grid_layer_list, list), "The grid_layer_list is not a list"
    assert len(grid_layer_list) > 0, "The grid_layer_list is empty"
    assert mode in ["concat", "sum", "avg", "max", "min", "rand", "user_defined"], "The mode is not supported"
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
    if mode == "concat":
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
    for i in range(lat_count):
        for j in range(lon_count):
            if mode == "user_defined":
                grid_data[i, j] = user_defined_func([grid.data[i, j] for grid in grid_layer_list])
                continue
            for grid in grid_layer_list:
                if np.isnan(grid.data[i,j]):
                    continue
                if mode !="rand" and np.isnan(grid_data[i,j]):
                    if mode == "min":
                        grid_data[i,j] = np.inf
                    elif mode == "max":
                        grid_data[i,j] = -np.inf
                    elif mode != "user_defined":
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
        mode="concat",
        user_defined_func: Optional[Callable] = None
):
    """
    Fuse multiple GraphLayer into one GraphLayer

    Parameters
    --------------------
        graph_layer_list (list(GraphLayer)): List of graph layers to be fused
        name (str, optional): name of the fused GraphLayer. Defaults to None. (use the name of the first GraphLayer)
        year (int, optional): year of the fused GraphLayer. Defaults to None. (use the year of the first GraphLayer)
        [start_lat, end_lat, start_lon, end_lon] (float, optional): the range of the fused GraphLayer. Defaults to None. (use the range of all GraphLayers)
        mode (str): Mode of fusion. “concat”, “sum”, “avg”, “max”, “min” or “random”. Defaults to “concat”.
        user_defined_func (Callable, optional): A custom function that processes nodes and edges during their fusion.
    """
    if not graph_layer_list:
        raise ValueError("The graph_layer_list cannot be empty.")
    
    # Use the name and year of the first GraphLayer if not provided
    if name is None:
        name = graph_layer_list[0].name
    if year is None:
        year = graph_layer_list[0].year
    
    # Create a new GraphLayer for fusion
    fused_graph = GraphLayer(name=name, year=year, directed=graph_layer_list[0].directed)
    
    node_data = {}
    
    # Process each GraphLayer
    for layer in graph_layer_list:
        for node, attr in layer.data.nodes(data=True):
            lat = attr.get("lat")
            lon = attr.get("lon")
            if start_lat is not None and (lat < start_lat or lat > end_lat):
                continue
            if start_lon is not None and (lon < start_lon or lon > end_lon):
                continue

            if node not in node_data:
                node_data[node] = {"lat": lat, "lon": lon, "attributes": {}}
                for k, v in attr.items():
                    if k not in ["lat", "lon"]:
                        node_data[node]["attributes"][k] = [v]
            else:
                for k, v in attr.items():
                    if k in ["lat", "lon"]:
                        continue
                    node_data[node]["attributes"][k].append(v)

    # Apply the custom function to each attribute
    for node, data in node_data.items():
        for k, v in data["attributes"].items():
            if user_defined_func:
                node_data[node][k] = user_defined_func(v)
            elif mode == "avg":
                node_data[node][k] = sum(v) / len(v)
            elif mode == "sum":
                node_data[node][k] = sum(v)
            elif mode == "max":
                node_data[node][k] = max(v)
            elif mode == "min":
                node_data[node][k] = min(v)
            elif mode == "random":
                node_data[node][k] = random.choice(v)
        del node_data[node]["attributes"]

    # Add nodes and edges to the fused graph
    fused_graph.data.add_nodes_from([(node, attr) for node, attr in node_data.items()])
    fused_graph.data.add_edges_from(graph_layer_list[0].data.edges(data=True))

    return fused_graph
    

def point_fusion(
    point_layer_list: List[PointLayer],
    name=None,
    year=None,
    column_list=None,
    start_lat=None,
    end_lat=None,
    start_lon=None,
    end_lon=None,
    mode="concat",
    user_defined_func=None
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
        mode (str): Mode of fusion. "concat", "avg", "sum", "max", "min" or "random". Defaults to "concat".
        user_defined_func (Callable, optional): A custom function that processes nodes during their fusion
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
    # Initialize the fused PointLayer
    fused_layer = PointLayer(name=name, year=year, column_list=column_list)
    
    if mode == "concat":
        # Start with the first layer's data
        concatenated_data = point_layer_list[0].data.copy()
        existing_columns = set(concatenated_data.columns)

        # Iterate over the remaining layers
        for layer in point_layer_list[1:]:
            layer_data = layer.data.copy()
            layer_columns = set(layer_data.columns)

            # Check for duplicate columns
            duplicate_columns = existing_columns.intersection(layer_columns)
            if duplicate_columns:
                print(f"Warning: Duplicate columns found: {duplicate_columns}. Values from the first layer will be used.")
                # Drop the duplicate columns from the current layer
                layer_data = layer_data.drop(columns=duplicate_columns)

            # Merge the new columns into the concatenated data
            concatenated_data = pd.concat([concatenated_data, layer_data], axis=1)

            # Update the existing columns set
            existing_columns.update(layer_columns)

        # Filter by lat/lon range if specified
        if start_lat is not None and end_lat is not None:
            concatenated_data = concatenated_data[(concatenated_data["lat"] >= start_lat) & (concatenated_data["lat"] <= end_lat)]
        if start_lon is not None and end_lon is not None:
            concatenated_data = concatenated_data[(concatenated_data["lon"] >= start_lon) & (concatenated_data["lon"] <= end_lon)]
        
        fused_layer.add_points(concatenated_data.reset_index(drop=True))
        return fused_layer

    # For other modes: avg, sum, max, min, random
    fused_data = []

    # Combine all data frames into one for easier group by operation
    combined_data = pd.concat([layer.data for layer in point_layer_list], ignore_index=True)
    
    # Filter by lat/lon range if specified
    if start_lat is not None and end_lat is not None:
        combined_data = combined_data[(combined_data["lat"] >= start_lat) & (combined_data["lat"] <= end_lat)]
    if start_lon is not None and end_lon is not None:
        combined_data = combined_data[(combined_data["lon"] >= start_lon) & (combined_data["lon"] <= end_lon)]
    
    # Group by lat/lon and fuse the data according to the specified mode
    for (lat, lon), group in combined_data.groupby(["lat", "lon"]):
        fused_row = {"lat": lat, "lon": lon}
        
        for col in column_list:
            values = group[col].values
            if user_defined_func:
                fused_row[col] = user_defined_func(values)
            else:
                if mode == "avg":
                    fused_row[col] = values.mean()
                elif mode == "sum":
                    fused_row[col] = values.sum()
                elif mode == "max":
                    fused_row[col] = values.max()
                elif mode == "min":
                    fused_row[col] = values.min()
                elif mode == "random":
                    fused_row[col] = random.choice(values)

        fused_data.append(fused_row)
    
    # Create a DataFrame from the fused data
    fused_data = pd.DataFrame(fused_data)

    # Add the fused data to the new PointLayer
    fused_layer.add_points(fused_data)

    return fused_layer


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
        
