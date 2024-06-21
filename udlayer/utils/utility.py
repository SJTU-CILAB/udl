try:
    from osgeo import osr, gdal
except ImportError:
    raise (""" ERROR: Could not find the GDAL/OSR Python library bindings. 
               You can install it with this command:
               conda install gdal""") 
import torch
import networkx as nx
from torch_geometric.data import Data
from ..layer.graphlayer import GraphLayer
import numpy as np
from torch_geometric.utils.convert import from_networkx
from itertools import chain
import dgl

def get_coord_type(filedir: str, row: int, col: int):
    """
    Get the type of coordinates from the file directory

    Parameters
    --------------------
        filedir (str): The file directory of the coordinates
    
    Returns
    --------------------
        coord_type (tuple): The coordinate of the position of the row and column. To get the type of the coordinate, run `get_coord_type(filedir, 0, 0)` to get the coordinate of the top left corner.
        If the tuple is (latitute, longitude), then the parameter "coord_type" is "latlon". Otherwise, the parameter "coord_type" is "lonlat".
    """

    if not isinstance(filedir, str):
        raise TypeError("Input must be string.")

    # get the existing coordinate system
    ds = gdal.Open(filedir)
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system
    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs, new_cs)

    # get the point to transform, pixel (0,0) in this case
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()

    data = ds.ReadAsArray()

    y = gt[3] + row * gt[5] + col * gt[4]
    x = gt[0] + col * gt[1] + row * gt[2]
    coord = transform.TransformPoint(x, y)
    return coord


def graph_to_nx(graph: GraphLayer):
    """
    Convert a graph to a networkx graph

    Parameters
    --------------------:
        graph (Graphlayer): The Graphlayer to be transformed
    """

    if not isinstance(graph, GraphLayer):
        raise TypeError("Input must be GraphLayer.")
    
    return graph.data


def graph_to_torch(graph: GraphLayer, node_attr=None, edge_attr=None):
    """
    Convert a graph to a torch_geometric graph

    Parameters
    --------------------:
        graph (Graphlayer): The Graphlayer to be transformed
        node_attr (list): The list of node attributes
        edge_attr (list): The list of edge attributes
    """

    if not isinstance(graph, GraphLayer):
        raise TypeError("Input must be GraphLayer.")
    
    G = graph.data
    if node_attr is None:
        node_attr = set(chain.from_iterable(d.keys() for *_, d in G.nodes(data=True)))
    print("The default node attributes are: ", node_attr)
    if edge_attr is None:
        edge_attr = set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))
    print("The default edge attributes are: ", edge_attr)
    pyg = from_networkx(graph.data, group_node_attrs=node_attr, group_edge_attrs=edge_attr)
    return pyg


def graph_to_dgl(graph: GraphLayer, node_attr=None, edge_attr=None):
    """
    Convert a graph to a dgl graph

    Parameters
    --------------------:
        graph (Graphlayer): The Graphlayer to be transformed
        node_attr (list): The list of node attributes
        edge_attr (list): The list of edge attributes
    """

    if not isinstance(graph, GraphLayer):
        raise TypeError("Input must be GraphLayer.")
    
    G = graph.data
    if node_attr is None:
        node_attr = set(chain.from_iterable(d.keys() for *_, d in G.nodes(data=True)))
    print("The default node attributes are: ", node_attr)
    if edge_attr is None:
        edge_attr = set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))
    print("The default edge attributes are: ", edge_attr)
    g = dgl.from_networkx(graph.data, node_attrs=node_attr, edge_attrs=edge_attr)
    return g