import numpy as np
import rasterio
from typing import List, Tuple
from functools import reduce
import rasterio
from pyproj import Transformer
from tqdm import tqdm
from ..layer.gridlayer import GridLayer
from ..layer.graphlayer import GraphLayer
from ..layer.pointlayer import PointLayer
from ..layer.linestringlayer import LinestringLayer
from ..layer.polygonlayer import PolygonLayer
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import networkx as nx
import math

# Raw to Layer


def tif_to_grid(
    name: str,
    filedir: List[str],
    start_lat: float,
    end_lat: float,
    start_lon: float,
    end_lon: float,
    step_lat=0.01,
    step_lon=0.01,
    year=None,
    band=1,
    coord_type="latlon",
    mode="avg",
):
    """
    This function converts a tif file to a grid layer.
    grid[row, col] will become np.nan if there is no data in the corresponding area.
    The tif's coordinate system should be EPSG:4326

    Parameters
    --------------------
        name (str): Name of the output GridLayer.
        filedir (List[str]): Paths of input tif files.
        [start_lat, end_lat] (float): Latitude range(closed) of the output GridLayer.
        [start_lon, end_lon] (float): Longitude range(closed) of the output GridLayer.
        step_lat, step_lon (float): Step of latitude and longitude.
        year (int, optional): Year of the data(which will be written into GridLayer). Defaults to None.
        band (int, optional): Band of tif file. Defaults to 1.
        coord_type (str, optional): Coordinate type of tif file. "latlon" or "lonlat". Defaults to "latlon".
        mode (str, optional): The way of aggregation. "sum", "avg", "max", "min" or "rand". Defaults to "avg".

    TODO:
        Test "rand" mode
    """
    assert -90 <= start_lat <= end_lat <= 90, "latitude range error"
    assert -180 <= start_lon <= end_lon <= 180, "longitude range error"
    assert mode in ["sum", "avg", "max", "min", "rand"], "parameter error: mode"
    assert coord_type in ["latlon", "lonlat"], "parameter error: coord_type"
    dataset_and_data = list(
        map(
            lambda file_path: (
                rasterio.open(file_path),
                rasterio.open(file_path).read(band),
            ),
            filedir,
        )
    )

    def dataset_to_griddata(
        dataset: rasterio.io.DatasetReader, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        crs = dataset.crs
        no_data = dataset.meta["nodata"]
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        def rowcol2lonlat(row, col):
            """
            input: coordinate (row, col) in numpy.ndarray(data)
            output: a tuple with the format (lon, lat)
            """
            x, y = dataset.transform * (col, row)
            lon, lat = transformer.transform(x, y)
            return (lon, lat) if coord_type == "latlon" else (lat, lon)

        lat_count, lon_count = (
            math.ceil((end_lat - start_lat) / step_lat),
            math.ceil((end_lon - start_lon) / step_lon),
        )
        if mode == "rand":
            grid_data = np.empty(shape=(lat_count, lon_count), dtype=object)
            for i in range(lat_count):
                for j in range(lon_count):
                    grid_data[i, j] = []
        else:
            grid_data = np.full(shape=(lat_count, lon_count), fill_value=np.nan)
        grid_data_count = np.full(shape=(lat_count, lon_count), fill_value=np.nan)

        for row in tqdm(range(data.shape[0])):
            for col in range(data.shape[1]):
                lon, lat = rowcol2lonlat(row, col)
                if (
                    data[row, col] != no_data
                    and start_lat <= lat <= end_lat
                    and start_lon <= lon <= end_lon
                ):
                    grid_row, grid_col = (
                        int((end_lat - lat) // step_lat),
                        int((lon - start_lon) // step_lon),
                    )
                    if mode != "rand" and np.isnan(grid_data[grid_row, grid_col]):
                        if mode == "min":
                            grid_data[grid_row, grid_col] = np.inf
                        elif mode == "max":
                            grid_data[grid_row, grid_col] = -np.inf
                        else:
                            grid_data[grid_row, grid_col] = 0
                        grid_data_count[grid_row, grid_col] = 0
                    if mode == "avg" or mode == "sum":
                        grid_data[grid_row, grid_col] += data[row, col]
                    elif mode == "max":
                        grid_data[grid_row, grid_col] = max(
                            data[row, col], grid_data[grid_row, grid_col]
                        )
                    elif mode == "min":
                        grid_data[grid_row, grid_col] = min(
                            data[row, col], grid_data[grid_row, grid_col]
                        )
                    elif mode == "rand":
                        grid_data[grid_row, grid_col].append(data[row, col])
                    grid_data_count[grid_row, grid_col] += 1
        return grid_data, grid_data_count

    def merge_add(x, y):
        if np.isnan(x) and np.isnan(y):
            return np.nan
        elif np.isnan(x) and np.isnan(y) == False:
            return y
        elif np.isnan(x) == False and np.isnan(y):
            return x
        else:
            return x + y

    def merge(
        data_list: List[np.ndarray], count_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        --------------------
            data_list (List[np.ndarray]): a list of numpy arrays, each converted from a tif file
            count_list (List[np.ndarray]): the corresponding count array of data_list

        Returns:
        --------------------
            Tuple[np.ndarray, np.ndarray]: merged_grid_data and merged_grid_data_count
        """
        merged_grid_data = np.full(data_list[0].shape, fill_value=np.nan)
        merged_grid_data_count = np.full(count_list[0].shape, fill_value=np.nan)
        for row in range(merged_grid_data.shape[0]):
            for col in range(merged_grid_data.shape[1]):
                merged_grid_data[row, col] = reduce(
                    merge_add, map(lambda data: data[row, col], data_list)
                )
                merged_grid_data_count[row, col] = reduce(
                    merge_add, map(lambda data: data[row, col], count_list)
                )
        return merged_grid_data, merged_grid_data_count

    data_list = list()
    count_list = list()
    for dataset, data in tqdm(dataset_and_data):
        grid_data, grid_data_count = dataset_to_griddata(dataset, data)
        data_list.append(grid_data)
        count_list.append(grid_data_count)
    merged_grid_data, merged_grid_data_count = merge(data_list, count_list)
    if mode == "avg":
        final_grid_data = merged_grid_data / merged_grid_data_count
    elif mode == "rand":
        final_grid_data = np.array(
            list(map(lambda x: np.random.choice(x), merged_grid_data))
        )
    else:
        final_grid_data = merged_grid_data

    gridlayer = GridLayer(
        name, start_lat, end_lat, start_lon, end_lon, step_lat, step_lon, year
    )
    gridlayer.construct_grid(final_grid_data)
    return gridlayer


def csv_to_grid(
    name: str,
    filedir: str,
    start_lat: float,
    end_lat: float,
    start_lon: float,
    end_lon: float,
    step_lat: float,
    step_lon: float,
    year=None,
    mode="avg",
):
    """
    This function converts a csv file to a grid layer.
    The csv file should have the following columns:
    "latitude", "longitude", name, ...
    grid[row, col] will become np.nan if there is no data in the corresponding area.

    Parameters
    --------------------
        name (str): Name of the output GridLayer.
        filedir (str): Path of input csv file.
        [start_lat, end_lat] (float): Latitude range(closed) of the output GridLayer.
        [start_lon, end_lon] (float): Longitude range(closed) of the output GridLayer.
        step_lat, step_lon (float): Step of latitude and longitude.
        year (int, optional): Year of the data(which will be written into GridLayer). Defaults to None.
        mode (str, optional): The way of aggregation. "sum", "avg", "max", "min" or "rand". Defaults to "avg".
    """
    assert -90 <= start_lat <= end_lat <= 90, "latitude range error"
    assert -180 <= start_lon <= end_lon <= 180, "longitude range error"
    assert mode in ["sum", "avg", "max", "min", "rand"], "parameter error: mode"
    lat_count = round((end_lat - start_lat) / step_lat)
    lon_count = round((end_lon - start_lon) / step_lon)
    if mode == "rand":
        grid_data = np.empty(shape=(lat_count, lon_count), dtype=object)
        for i in range(lat_count):
            for j in range(lon_count):
                grid_data[i, j] = []
    else:
        grid_data = np.full(shape=(lat_count, lon_count), fill_value=np.nan)
    grid_data_count = np.full(shape=(lat_count, lon_count), fill_value=np.nan)
    df = pd.read_csv(filedir)
    # TODO: speed up by numpy
    for index, row in df.iterrows():
        lat, lon, val = row["latitude"], row["longitude"], row["value"]
        if start_lat <= lat <= end_lat and start_lon <= lon <= end_lon:
            grid_pos = (
                int((end_lat - lat) // step_lat),
                int((lon - start_lon) // step_lon),
            )
            if mode != "rand " and np.isnan(grid_data[grid_pos[0], grid_pos[1]]):
                if mode == "min":
                    grid_data[grid_pos[0], grid_pos[1]] = np.inf
                elif mode == "max":
                    grid_data[grid_pos[0], grid_pos[1]] = -np.inf
                else:
                    grid_data[grid_pos[0], grid_pos[1]] = 0
                grid_data_count[grid_pos[0], grid_pos[1]] = 0
            if mode == "avg" or mode == "sum":
                grid_data[grid_pos[0], grid_pos[1]] += val
            elif mode == "max":
                grid_data[grid_pos[0], grid_pos[1]] = max(
                    val, grid_data[grid_pos[0], grid_pos[1]]
                )
            elif mode == "min":
                grid_data[grid_pos[0], grid_pos[1]] = min(
                    val, grid_data[grid_pos[0], grid_pos[1]]
                )
            elif mode == "rand":
                grid_data[grid_pos[0], grid_pos[1]].append(val)
            grid_data_count[grid_pos[0], grid_pos[1]] += 1
    if mode == "avg":
        final_grid_data = grid_data / grid_data_count
    elif mode == "rand":
        final_grid_data = np.array(list(map(lambda x: np.random.choice(x), grid_data)))
    else:
        final_grid_data = grid_data
    gridlayer = GridLayer(
        name, start_lat, end_lat, start_lon, end_lon, step_lat, step_lon, year
    )
    gridlayer.construct_grid(final_grid_data)
    return gridlayer


def json_to_graph(
    name, filedir, year=None, directed=False, edge_attribute=None, edge_weight=None
):
    """
    This function converts a json file to a graph layer.
    The json file should have the following format:
    {
        "nodes": [
            {
                "id": 0,
                "latitude": 0.0,
                "longitude": 0.0,
                name: float
            }
        ],
        "edges": [
            {
                "source": 0,
                "target": 0,
                "weight": float
            }
        ]
    }
    
    Parameters
    --------------------
        name (str): Name of the output GridLayer.
        filedir (str): Path of input json file.
        year (int, optional): Year of the data(which will be written into GridLayer). Defaults to None.
        directed (bool, optional): Whether the graph is a directed graph. Defaults to False.
        edge_attribute (str, optional): The attribute name of the edge. Defaults to None which means the edges do not have any feature.
        edge_weight (list[type], optional): The list of the weight of edge.
    """
    graphlayer = GraphLayer(name, year, directed)
    node_info = json.load(open(filedir, "r"))["nodes"]
    node_label = list(map(lambda node: node["id"], node_info))
    latitude = list(map(lambda node: node["latitude"], node_info))
    longitude = list(map(lambda node: node["longitude"], node_info))
    node_attribute = list(map(lambda node: node[name], node_info))
    if edge_attribute != None:
        try:
            edge_info = json.load(open(filedir, "r"))["edges"]
            source = list(map(lambda edge: edge["source"], edge_info))
            target = list(map(lambda edge: edge["target"], edge_info))
            try:
                edge_weight = list(map(lambda edge: edge["weight"], edge_info))
            except KeyError:
                print(
                    "edge_attribute is not None, but the json file does not have the true information of edge weight."
                )
                edge_weight = None
        except KeyError:
            print("the json file does not contain the true information of egdes")
            edge_attribute = None
            edge_weight = None
    else:
        edge_info = json.load(open(filedir, "r"))["edges"]
        source = list(map(lambda edge: edge["source"], edge_info))
        target = list(map(lambda edge: edge["target"], edge_info))

    graphlayer.construct_graph(
        node_label,
        latitude,
        longitude,
        source,
        target,
        node_attribute,
        edge_attribute,
        edge_weight,
    )
    return graphlayer


def json_to_polygon(
    name: str, filedir: str, year=None, output_dir=None, encoding="utf-8"
):
    """
    This function converts a json file to a polygon layer.
    The json file should have the following format:
    {
        "features":[
            {
                "type":"Feature",
                "geometry":{
                    "type":"Polygon",
                    "coordinates":[[[120.992531,30.955028],[120.991683,30.958211],...]]
                },
                ...
            }
        ]
    }

    Parameters
    --------------------
        name (str): Name of the output PolygonLayer.
        filedir (str): Path of input json file.
        year (int, optional): Year of the data(which will be written into PolygonLayer). Defaults to None.
        output_dir (str, optional): Path of output shp file. Defaults to None.
        encoding (str, optional): Encoding of output shp file. Defaults to 'utf-8'.
    """

    with open(filedir, "r", encoding=encoding) as f:
        data = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(data["features"])
    for column in gdf.columns:
        if isinstance(gdf[column].iloc[0], list):
            gdf[column] = gdf[column].apply(str)
    polygonlayer = PolygonLayer(name, year)
    polygonlayer.add_polygons(gdf)
    if not output_dir is None:
        gdf.to_file(output_dir, encoding=encoding)
    return polygonlayer


def csv_to_graph(
    name, node_filedir, edge_filedir, year=None, directed=False, edge_attribute=None
):
    """
    This function converts a node csv along with a edge csv file to a graph layer.
    
    The node csv file should have the following columns:
        "id", "latitude", "longitude", name (optional)
        
    The edge csv file should have the following columns:
        "source", "target", "edge_weight" (optional)

    Parameters
    --------------------
        name : Name of graphlayer.
        node_filedir : Path of csv file for constructing nodes.
        edge_filedir: Path of csv file for constructing edges.
        year : Year of the data.
        directed (bool, optional): Whether the graph is a directed graph. Defaults to False.
        edge_attribute (str, optional): The attribute name of the edge. Defaults to None which means the edges do not have any feature.
    """

    df = pd.read_csv(node_filedir)
    node_label = df["id"].values
    latitude = df["latitude"].values
    longitude = df["longitude"].values
    if name in df.columns:
        node_attribute = df[name].values
    else:
        node_attribute = None

    edge_df = pd.read_csv(edge_filedir)
    source = edge_df["source"]
    target = edge_df["target"]
    if edge_attribute != None:
        if "edge_weight" in edge_df.columns:
            edge_weight = edge_df["edge_weight"].values
        else:
            print(
                "Edge weights are not provided in csv file or the column is named incorrectly, the name should be 'edge_weight'"
            )
            edge_weight = None
    else:
        edge_weight = None
    graph_layer = GraphLayer(name, year, directed)
    graph_layer.construct_graph(
        node_label,
        latitude,
        longitude,
        source,
        target,
        node_attribute,
        edge_attribute,
        edge_weight,
    )

    return graph_layer


def csv_to_point(name, filedir, year=None, column_list=[]):
    """
    This function convert a csv file to a point layer.
    The csv file should have the following columns:
    "latitude", "longitude", name, ...
    The column_list needn't contain the latitude and longitude information.
    
    Parameters
    --------------------
        name (str): Name of the output PointLayer.
        filedir (str): Path of input csv file.
        year (int, optional): Year of the data(which will be written into PointLayer). Defaults to None.
        column_list (list, optional): The list of column names in the csv file. Defaults to [].
    """
    df = pd.read_csv(filedir)
    pointlayer = PointLayer(name, year, column_list.copy())
    column_list.extend(["latitude", "longitude"])
    new_df = df[column_list]
    new_df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)
    pointlayer.add_points(new_df)
    return pointlayer


def json_to_point(name, filedir, year=None):
    """
    This function converts a json file to a point layer.
    The json file should have the following format:
    {
        "points": [
            {
                "latitude": 0.0,
                "longitude": 0.0,
                ,
            }
        ]
    }
    
    Parameters
    --------------------
        name (str): Name of the output GridLayer.
        filedir (str): Path of input json file.
        year (int, optional): Year of the data(which will be written into GridLayer). Defaults to None.
    """
    try:
        point_info = json.load(open(filedir, "r"))["points"]
    except KeyError:
        print(
            "the json file does not contain the true information of points, the json file should have the following format: {'points': [{'latitude': 0.0, 'longitude': 0.0, ...}]}"
        )
        return None
    try:
        column_list = list(point_info[0].keys())
        column_list.remove("latitude")
        column_list.remove("longitude")
        pointlayer = PointLayer(name, year, column_list)
    except KeyError:
        print(
            "the json file does not contain the true information of coordinates, the json file should have the following format: {'points': [{'latitude': 0.0, 'longitude': 0.0, ...}]}"
        )
        return None
    pd_json = {
        name: list(map(lambda point: point[name], point_info))
        for name in list(point_info[0].keys())
    }
    df = pd.DataFrame(pd_json)
    df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)
    pointlayer.add_points(df)
    return pointlayer


def csv_to_linestring(name, filedir, sort_column, group_column, year=None):
    """
    This function converts a csv file to a point layer.
    The csv file should have the following columns:
    "latitude", "longitude",  ...
    sort_column: the column name to form the linestring (the sequence of points in one linestring)
    group_column: the column name to group the linestring (which points are in one linestring)
    
    Parameters
    --------------------
        name (str): Name of the output Linestring layer.
        filedir (str): Path of input csv file.
        sort_column (str): the column name to form the linestring (the sequence of points in one linestring).
        group_column (str): the column name to group the linestring (which points are in one linestring).
        year (int, optional): Year of the data(which will be written into GridLayer). Defaults to None.
    """
    df = pd.read_csv(filedir)
    group_df = df.groupby(group_column)
    group_df.apply(lambda x: x.sort_values(sort_column))
    ls_geometry = []
    groups = []
    for g in group_df:
        linestring = [(x, y) for x, y in zip(g[1]["longitude"], g[1]["latitude"])]
        if len(linestring) > 1:
            linestring = LineString(linestring)
        else:
            linestring = LineString(linestring + linestring)
        ls_geometry.append(linestring)
        groups.append(g[0])
    new_df = pd.DataFrame(columns=[group_column, "geometry"])
    new_df[group_column] = groups
    gdf = gpd.GeoDataFrame(new_df, crs="EPSG:4326", geometry=ls_geometry)
    linestringlayer = LinestringLayer(name, year=year)
    linestringlayer.add_linestrings(gdf)
    return linestringlayer


# Layer to Layer
def point_to_grid(
    point_data: PointLayer,
    feature_name: str,
    start_lat: float,
    end_lat: float,
    start_lon: float,
    end_lon: float,
    step_lat: float,
    step_lon: float,
    target_name=None,
    year=None,
    mode="avg",
):
    """
    Convert a point layer to a grid layer.
    The value of the grid layer is the sum/average/max/min/random_sample of the points in the corresponding area.

    Parameters
    --------------------
        point_data (PointLayer): The input point layer.
        If target_name is None, the target_name will be the name of the point layer.
        target_name (str): The name of the attribute of the point layer that will be used to construct the grid layer.
        [start_lat, end_lat] (float): Latitude range(closed) of the output GridLayer.
        [start_lon, end_lon] (float): Longitude range(closed) of the output GridLayer.
        step_lat, step_lon (float): Step of latitude and longitude.
        year(int, optional): Year of the data(which will be written into GridLayer). Defaults to None.
        mode (str, optional): The way of aggregation. "sum", "avg", "max", "min", "rand" or "count". Defaults to "avg". "count" means only counting the number of points.
    """
    assert -90 <= start_lat <= end_lat <= 90, "latitude range error"
    assert -180 <= start_lon <= end_lon <= 180, "longitude range error"
    assert mode in [
        "sum",
        "avg",
        "max",
        "min",
        "rand",
        "count",
    ], "parameter error: mode"
    assert (mode == "count") or (
        feature_name in point_data.data.columns
    ), "parameter error: feature_name"
    lat_count = round((end_lat - start_lat) / step_lat)
    lon_count = round((end_lon - start_lon) / step_lon)
    if mode == "rand":
        grid_data = np.empty(shape=(lat_count, lon_count), dtype=object)
        for i in range(lat_count):
            for j in range(lon_count):
                grid_data[i, j] = []
    else:
        grid_data = np.full(shape=(lat_count, lon_count), fill_value=np.nan)
    grid_data_count = np.full(shape=(lat_count, lon_count), fill_value=np.nan)
    for index, row in point_data.data.iterrows():
        lat, lon = row["lat"], row["lon"]
        val = 1 if mode == "count" else row[feature_name]
        if start_lat <= lat <= end_lat and start_lon <= lon <= end_lon:
            grid_pos = (
                int((end_lat - lat) // step_lat),
                int((lon - start_lon) // step_lon),
            )
            if mode != "rand" and np.isnan(grid_data[grid_pos[0], grid_pos[1]]):
                if mode == "min":
                    grid_data[grid_pos[0], grid_pos[1]] = np.inf
                elif mode == "max":
                    grid_data[grid_pos[0], grid_pos[1]] = -np.inf
                else:
                    grid_data[grid_pos[0], grid_pos[1]] = 0
                grid_data_count[grid_pos[0], grid_pos[1]] = 0
            if mode == "avg" or mode == "sum" or mode == "count":
                grid_data[grid_pos[0], grid_pos[1]] += val
            elif mode == "max":
                grid_data[grid_pos[0], grid_pos[1]] = max(
                    val, grid_data[grid_pos[0], grid_pos[1]]
                )
            elif mode == "min":
                grid_data[grid_pos[0], grid_pos[1]] = min(
                    val, grid_data[grid_pos[0], grid_pos[1]]
                )
            elif mode == "rand":
                grid_data[grid_pos[0], grid_pos[1]].append(val)
            grid_data_count[grid_pos[0], grid_pos[1]] += 1
    if mode == "avg":
        grid_data /= grid_data_count
    elif mode == "rand":
        grid_data = np.array(list(map(lambda x: np.random.choice(x), grid_data)))

    if year == None:
        year = point_data.year
    if target_name == None:
        target_name = feature_name
    gridlayer = GridLayer(
        target_name, start_lat, end_lat, start_lon, end_lon, step_lat, step_lon, year
    )
    gridlayer.construct_grid(grid_data)
    return gridlayer


def lines_to_point(lines_data: LinestringLayer, target_name=None, year=None):
    """
    Convert a linestring layer to a point layer.
    The value of the point layer is the number of the linestring vertexes in the corresponding area.

    Parameters
    --------------------
        lines_data (LinestringLayer): The input linestring layer
        target_name (str): The name of the attribute of the point layer that will be used to construct the grid layer
        year (int, optional): Year of the data(which will be written into GridLayer). Defaults to None.
    """
    if target_name == None:
        name = lines_data.name
    else:
        name = target_name
    if year == None:
        year = lines_data.year
    point_data = pd.DataFrame(columns=["lat", "lon", name])
    for index, row in lines_data.data.iterrows():
        linestring = row["geometry"]
        for coord in linestring.coords:
            lat, lon = coord[1], coord[0]
            point_data.loc[len(point_data)] = [lat, lon, 1]
    pointlayer = PointLayer(name, year)
    pointlayer.add_points(point_data)
    return pointlayer


def linestring_to_grid(
    linestring_data: LinestringLayer,
    start_lat: float,
    end_lat: float,
    start_lon: float,
    end_lon: float,
    step_lat: float,
    step_lon: float,
    target_name=None,
    year=None,
):
    """
    Convert a linestring layer to a grid layer.
    The value of the grid layer is the number of the linestring vertexes in the corresponding area.
    
    Parameters
    --------------------
        linestring_data (LinestringLayer): The input linestring layer
        target_name (str): The name of the attribute of the linestring layer that will be used to construct the grid layer
        [start_lat, end_lat] (float): Latitude range(closed) of the output GridLayer
        [start_lon, end_lon] (float): Longitude range(closed) of the output GridLayer
        step_lat, step_lon (float): Step of latitude and longitude
    """

    if target_name == None:
        name = linestring_data.name
    else:
        name = target_name
    if year == None:
        year = linestring_data.year
    assert -90 <= start_lat <= end_lat <= 90, "latitude range error"
    assert -180 <= start_lon <= end_lon <= 180, "longitude range error"
    lat_count = round((end_lat - start_lat) / step_lat)
    lon_count = round((end_lon - start_lon) / step_lon)
    grid_data = np.full(shape=(lat_count, lon_count), fill_value=0)
    for index, row in linestring_data.data.iterrows():
        linestring = row["geometry"]
        for coord in linestring.coords:
            lat, lon = coord[1], coord[0]
            if start_lat <= lat <= end_lat and start_lon <= lon <= end_lon:
                grid_pos = (
                    int((end_lat - lat) // step_lat),
                    int((lon - start_lon) // step_lon),
                )
                grid_data[grid_pos[0], grid_pos[1]] += 1

    gridlayer = GridLayer(
        name, start_lat, end_lat, start_lon, end_lon, step_lat, step_lon, year
    )
    gridlayer.construct_grid(grid_data)
    return gridlayer


def polygon_to_grid(
    polygon_data: PolygonLayer,
    target_name: str,
    start_lat: float,
    end_lat: float,
    start_lon: float,
    end_lon: float,
    step_lat: float,
    step_lon: float,
):
    """
    Convert a polygon layer to a grid layer.
    The value of the grid layer is 1 if the polygon contains the corresponding area, otherwise 0.

    Parameters
    --------------------
        polygon_data (PolygonLayer): The input polygon layer
        target_name (str): The name of the attribute of the polygon layer that will be used to construct the grid layer
        [start_lat, end_lat] (float): Latitude range(closed) of the output GridLayer
        [start_lon, end_lon] (float): Longitude range(closed) of the output GridLayer
        step_lat, step_lon (float): Step of latitude and longitude
    """

    lat_count = round((end_lat - start_lat) / step_lat)
    lon_count = round((end_lon - start_lon) / step_lon)
    grid_data = np.full(shape=(lat_count, lon_count), fill_value=0)
    for index, row in polygon_data.data.iterrows():
        polygon = row["geometry"]
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        for lat in np.arange(min_lat, max_lat, step_lat):
            for lon in np.arange(min_lon, max_lon, step_lon):
                if (
                    start_lat <= lat <= end_lat
                    and start_lon <= lon <= end_lon
                    and polygon.contains(
                        Polygon(
                            [
                                (lon, lat),
                                (lon + step_lon, lat),
                                (lon + step_lon, lat + step_lat),
                                (lon, lat + step_lat),
                            ]
                        )
                    )
                ):
                    grid_pos = (
                        int((end_lat - lat) // step_lat),
                        int((lon - start_lon) // step_lon),
                    )
                    grid_data[grid_pos[0], grid_pos[1]] = 1
    gridlayer = GridLayer(
        target_name if target_name != None else polygon_data.name,
        start_lat,
        end_lat,
        start_lon,
        end_lon,
        step_lat,
        step_lon,
        polygon_data.year,
    )
    gridlayer.construct_grid(grid_data)
    return gridlayer


def polygon_to_graph(polygon_data, target_name=None):
    """
    Convert a polygon layer to a graph layer.
    The vertices of the polygon are transformed to nodes of graph. 
    The segments of the polygon are transformed to edges of graph. 
    The edge attributes default to None.
    The node_labels are assigned by the sequence of the vertices of the polygon.

    Parameters
    --------------------
        polygon_data (PolygonLayer): The input polygon layer data.
        target_name (str, optional): The name of the attribute of the polygon layer that will be used to construct the graph layer. Defaults to None.
    """
    if target_name == None:
        name = polygon_data.name
    else:
        name = target_name
    graph_layer = GraphLayer(name=name, year=polygon_data.year)

    node_label = []
    latitude = []
    longitude = []
    source = []
    target = []

    for index, row in polygon_data.data.iterrows():
        polygon = row["geometry"]

        for label, point in enumerate(polygon.exterior.coords):
            node_label.append(f"{index}_{label}")
            latitude.append(point[1])
            longitude.append(point[0])

        for i in range(len(polygon.exterior.coords) - 1):
            source.append(f"{index}_{i}")
            target.append(f"{index}_{i+1}")

    graph_layer.construct_graph(node_label, latitude, longitude, source, target)
    return graph_layer


def grid_to_point(grid_data: GridLayer, target_name=None, year=None):
    """
    Convert a grid layer to a point layer.
    The value of each point in point_layer.data is the value of the grid layer in the corresponding area.

    Parameters
    --------------------
        grid_data (GridLayer): The input grid layer
        If target_name is None, the target_name will be the name of the grid layer
        the same for the following
        target_name (str): The name of the attribute of the point layer that will be used to construct the grid layer
        [start_lat, end_lat] (float): Latitude range(closed) of the output GridLayer
        [start_lon, end_lon] (float): Longitude range(closed) of the output GridLayer
        step_lat, step_lon (float): Step of latitude and longitude
        year ([type], optional): Year of the data(which will be written into GridLayer). Defaults to None.
    """
    if target_name == None:
        target_name = grid_data.name
    if year == None:
        year = grid_data.year

    start_lat = grid_data.start_lat
    end_lat = grid_data.end_lat
    start_lon = grid_data.start_lon
    end_lon = grid_data.end_lon
    step_lat = grid_data.step_lat
    step_lon = grid_data.step_lon

    point_data = pd.DataFrame(columns=["lat", "lon", target_name])
    for row in range(grid_data.data.shape[0]):
        for col in range(grid_data.data.shape[1]):
            lat, lon = end_lat - row * step_lat, start_lon + col * step_lon
            if not np.isnan(grid_data.data[row, col]):
                point_data.loc[len(point_data)] = [lat, lon, grid_data.data[row, col]]
    pointlayer = PointLayer(target_name, year)
    pointlayer.add_points(point_data)
    return pointlayer


def grid_to_graph(grid_data, target_name=None, year=None):
    """
    Convert a grid layer to a graph layer.
    The grids of the grid layer are transformed to nodes of graph. 
    The grids that are adjacent to each other have edges.
    The edge attributes default to None and can be added later.
    The node_labels are assigned by the sequence of the vertices of the grid whose format are {row}_{col}.

    Parameters
    --------------------
        grid_data (GridLayer): The input grid layer
        If target_name is None, the target_name will be the name of the grid layer
        the same for the following
        target_name (str, optional): The name of the attribute of the point layer that will be used to construct the grid layer
        year (int, optional): Year of the data(which will be written into GridLayer). Defaults to None.
    """
    if target_name == None:
        target_name = grid_data.name
    if year == None:
        year = grid_data.year

    end_lat = grid_data.end_lat
    start_lon = grid_data.start_lon
    step_lat = grid_data.step_lat
    step_lon = grid_data.step_lon
    latitude = []
    longitude = []
    node_label = []
    source = []
    target = []
    node_attribute = []
    for row in range(grid_data.data.shape[0]):
        for col in range(grid_data.data.shape[1]):
            lat, lon = end_lat - row * step_lat, start_lon + col * step_lon
            latitude.append(lat)
            longitude.append(lon)
            node_attribute.append(grid_data.data[row, col])
            node_label.append(f"{row}_{col}")
            if row > 0:
                source.append(f"{row}_{col}")
                target.append(f"{row-1}_{col}")
            if col > 0:
                source.append(f"{row}_{col}")
                target.append(f"{row}_{col-1}")

    graphlayer = GraphLayer(target_name, year)
    graphlayer.construct_graph(
        node_label, latitude, longitude, source, target, node_attribute
    )
    return graphlayer


def graph_to_point(graph_data, target_name=None, year=None):
    """
    Convert a graph layer to a point layer.
    The value of each point in point_layer.data is the value of each node in the graph layer. (the information of edges are ignored)
    
    Parameters
    --------------------
        graph_data (GraphLayer): The input graph layer.
        target_name (str, optional): The name of the attribute of the point layer that will be used to construct the grid layer. Defaults to None.
        year (int, optional): Year of the data(which will be written into GridLayer). Defaults to None.
    """
    data = graph_data.data
    if target_name == None:
        target_name = graph_data.name
    if year == None:
        year = graph_data.year
    feature_json = nx.get_node_attributes(data, graph_data.name)
    lat_json = nx.get_node_attributes(data, "lat")
    lon_json = nx.get_node_attributes(data, "lon")
    point_data = pd.DataFrame(columns=["lat", "lon", target_name])
    point_data["lat"] = list(lat_json.values())
    point_data["lon"] = list(lon_json.values())
    point_data[target_name] = list(feature_json.values())
    pointlayer = PointLayer(target_name, year)
    pointlayer.add_points(point_data)
    return pointlayer


def graph_to_grid(
    graph_data,
    start_lat: float,
    end_lat: float,
    start_lon: float,
    end_lon: float,
    step_lat: float,
    step_lon: float,
    target_name=None,
    year=None,
    mode="avg",
):
    """
    Convert a graph layer to a grid layer.
    The value of each grid in grid_layer.data is the sum/average/max/min/random_sample of the nodes in the corresponding area.

    Parameters
    --------------------
        graph_data (GraphLayer): The input graph layer.
        start_lat, end_lat] (float): Latitude range(closed) of the output GridLayer.
        [start_lon, end_lon] (float): Longitude range(closed) of the output GridLayer.
        step_lat, step_lon (float): Step of latitude and longitude.
        target_name (str, optional): The name of the attribute of the point layer that will be used to construct the grid layer. Defaults to None.
        year (int, optional): Year of the data(which will be written into GridLayer). Defaults to None.
        mode (str, optional): The way of aggregation. "sum", "avg", "max", "min", "rand" or "count". Defaults to "avg". "count" means only counting the number of points.
    """
    if target_name == None:
        target_name = graph_data.name
    if year == None:
        year = graph_data.year
    assert -90 <= start_lat <= end_lat <= 90, "latitude range error"
    assert -180 <= start_lon <= end_lon <= 180, "longitude range error"
    assert mode in [
        "sum",
        "avg",
        "max",
        "min",
        "rand",
        "count",
    ], "parameter error: mode"
    lat_count = round((end_lat - start_lat) / step_lat)
    lon_count = round((end_lon - start_lon) / step_lon)
    if mode == "rand":
        grid_data = np.empty(shape=(lat_count, lon_count), dtype=object)
        for i in range(lat_count):
            for j in range(lon_count):
                grid_data[i, j] = []
    else:
        grid_data = np.full(shape=(lat_count, lon_count), fill_value=np.nan)
    grid_data_count = np.full(shape=(lat_count, lon_count), fill_value=np.nan)
    for node in graph_data.data.nodes(data=True):
        info = node[1]
        lat, lon = info["lat"], info["lon"]
        val = 1 if mode == "count" else info[target_name]
        if (
            graph_data.start_lat <= lat <= graph_data.end_lat
            and graph_data.start_lon <= lon <= graph_data.end_lon
        ):
            grid_pos = (
                int((graph_data.end_lat - lat) // graph_data.step_lat),
                int((lon - graph_data.start_lon) // graph_data.step_lon),
            )
            if mode != "rand" and np.isnan(grid_data[grid_pos[0], grid_pos[1]]):
                if mode == "min":
                    grid_data[grid_pos[0], grid_pos[1]] = np.inf
                elif mode == "max":
                    grid_data[grid_pos[0], grid_pos[1]] = -np.inf
                else:
                    grid_data[grid_pos[0], grid_pos[1]] = 0
                grid_data_count[grid_pos[0], grid_pos[1]] = 0
            if mode == "avg" or mode == "sum" or mode == "count":
                grid_data[grid_pos[0], grid_pos[1]] += val
            elif mode == "max":
                grid_data[grid_pos[0], grid_pos[1]] = max(
                    val, grid_data[grid_pos[0], grid_pos[1]]
                )
            elif mode == "min":
                grid_data[grid_pos[0], grid_pos[1]] = min(
                    val, grid_data[grid_pos[0], grid_pos[1]]
                )
            elif mode == "rand":
                grid_data[grid_pos[0], grid_pos[1]].append(val)
            grid_data_count[grid_pos[0], grid_pos[1]] += 1
    if mode == "avg":
        grid_data /= grid_data_count
    elif mode == "rand":
        grid_data = np.array(list(map(lambda x: np.random.choice(x), grid_data)))

    gridlayer = GridLayer(
        target_name, start_lat, end_lat, start_lon, end_lon, step_lat, step_lon, year
    )
    gridlayer.construct_grid(grid_data)
    return gridlayer


"""
def point_to_graph(
    point_data,
    target_name,
    year=None,
    directed=False,
    edge_attribute=None,
    edge_weight=None,
):
    graphlayer = GraphLayer(target_name, year, directed)
    node_label = list(map(lambda row: row["id"], point_data.data.iterrows()))
    latitude = list(map(lambda row: row["lat"], point_data.data.iterrows()))
    longitude = list(map(lambda row: row["lon"], point_data.data.iterrows()))
    node_attribute = list(map(lambda row: row[target_name], point_data.data.iterrows()))
    graphlayer.construct_graph(
        node_label,
        latitude,
        longitude,
        None,
        None,
        node_attribute,
        edge_attribute,
        edge_weight,
    )
    return graphlayer
    
def polygon_to_grid(
    polygon_data,
    target_name,
    start_lat,
    end_lat,
    step_lat,
    start_lon,
    end_lon,
    step_lon,
): 
    lat_count = round((end_lat - start_lat) / step_lat)
    lon_count = round((end_lon - start_lon) / step_lon)
    grid_data = np.full(shape=(lat_count, lon_count), fill_value=0)
    for polygon in polygon_data.data:
        min_lat, min_lon, max_lat, max_lon = polygon.bounds
        for lat in np.arange(min_lat, max_lat, step_lat):
            for lon in np.arange(min_lon, max_lon, step_lon):
                if (
                    start_lat <= lat <= end_lat
                    and start_lon <= lon <= end_lon
                    and polygon.contains(
                        Polygon(
                            [
                                (lat, lon),
                                (lat, lon + step_lon),
                                (lat + step_lat, lon),
                                (lat + step_lat, lon + step_lon),
                            ]
                        )
                    )
                ):
                    grid_pos = (
                        int((end_lat - lat) // step_lat),
                        int((lon - start_lon) // step_lon),
                    )
                    grid_data[grid_pos[0], grid_pos[1]] = 1
    gridlayer = GridLayer(
        target_name,
        start_lat,
        end_lat,
        step_lat,
        start_lon,
        end_lon,
        step_lon,
        polygon_data.year,
    )
    gridlayer.construct_grid(grid_data)
    return gridlayer

def point_to_graph(
    point_data,
    target_name,
    year=None,
    directed=False,
    edge_attribute=None,
    edge_weight=None,
):
    graphlayer = GraphLayer(target_name, year, directed)
    node_label = list(map(lambda row: row["id"], point_data.data.iterrows()))
    latitude = list(map(lambda row: row["lat"], point_data.data.iterrows()))
    longitude = list(map(lambda row: row["lon"], point_data.data.iterrows()))
    node_attribute = list(map(lambda row: row[target_name], point_data.data.iterrows()))
    graphlayer.construct_graph(
        node_label,
        latitude,
        longitude,
        None,
        None,
        node_attribute,
        edge_attribute,
        edge_weight,
    )
    return graphlayer
"""
