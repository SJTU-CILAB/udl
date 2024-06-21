import math
from ..layer.gridlayer import GridLayer
from ..layer.graphlayer import GraphLayer
from ..layer.pointlayer import PointLayer
from ..layer.linestringlayer import LineStringLayer
from ..layer.polygonlayer import PolygonLayer
import numpy as np


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


def graph_align(
        graph_layer : GraphLayer,
        start_lat : float, 
        end_lat : float, 
        target_lat_step : float,
        start_lon : float, 
        end_lon : float,
        target_lon_step : float,
):
    
    """
    Align the graphlayer to the target graph by fuing all nodes in a certain range of latitude and longitude.

    Parameters
    --------------------
        graphlayer (GridLayer): the gridlayer to be aligned
        [start_lat, end_lat, start_lon, end_lon] (float): the range of the target graph
        target_lat_step (float): the target latitude step size
        target_lon_step (float): the target longitude step size
    """

    def fuse_node(slat : float,
                  elat : float,
                  slon : float,
             
                  elon : float,
                  ):
        """
        Fuse all nodes in the range

        Parameters
    --------------------
            [slat, elat, slon, elon] (float): the range
        """
        cnt = 0
        fuse_nodes = []
        fused_lat , fused_lon = 0 , 0
        for node in graph_layer.data.nodes :
            if [(node[1]['lat']>slat or node[1]['lat']==slat) and
                node[1]['lat']<elat and
                (node[1]['lon']>slon or node[1]['lon']==slon) and
                node[1]['lon']<elon
                ]:
                cnt+=1
                fuse_nodes.append(node)
                label = node[0]
                graph_layer.data.nodes.remove(node)

        for n in fuse_nodes :
            fused_lat += n[1]['lat']
            fused_lon += n[1]['lon']
            for edge in graph_layer.data.edges :
                if n[0]==edge[0] :
                    edge[0] = label
                    for k in fuse_nodes:
                        if k[0]==edge[1] :
                            graph_layer.data.edges.remove(edge)
                if n[0]==edge[1] :
                    edge[1] = label
                    for k in fuse_nodes:
                        if k[0]==edge[0] :
                            graph_layer.data.edges.remove(edge)
        fused_lat = (elat - slat)/2
        fused_lon = (elon - slon)/2
        node_fused =  [
                (label, {"lat": fused_lat, "lon": fused_lon})
            ]
        return node_fused

    
    new_graph = GraphLayer(
        graph_layer.name,
        graph_layer.year
    )
    step = max(math.ceil((end_lat-start_lat)/target_lat_step),math.ceil((end_lon-start_lon)/target_lon_step))
    for i in range(step):
        lat = start_lat + i*target_lat_step

        if lat<end_lat or lat==end_lat :

            for i in range(step):
                lon = start_lon + i*target_lon_step
                
                if lat<end_lat or lat==end_lat :
                    new_graph.add_node(fuse_node(lat,lat+target_lat_step,lon,lon+target_lon_step))
                else: break
        else: break
    new_graph.add_edges_from(graph_layer.data.edges)
    return new_graph


            
    

