from .base import BaseLayer
import networkx as nx


class GraphLayer(BaseLayer):
    def __init__(self, name, year=None, directed=False):
        super(GraphLayer, self).__init__(name, year)
        self.directed = directed
        if directed:
            self.data = nx.DiGraph()
        else:
            self.data = nx.Graph()

    def construct_node(
        self, node_label: list, latitude: list, longitude: list, node_attribute=None
    ):
        if node_attribute is None:
            return [
                (label, {"lat": lat, "lon": lon})
                for label, lat, lon in zip(node_label, latitude, longitude)
            ]
        else:
            return [
                (label, {"lat": lat, "lon": lon, self.name: attr})
                for label, lat, lon, attr in zip(node_label, latitude, longitude, node_attribute)
            ]

    def construct_edge(self, source: list, target: list, edge_attribute=None, edge_weight=None):
        if edge_attribute is None:
            return [(scr, tgt) for scr, tgt in zip(source, target)]
        else:
            return [
                (scr, tgt, {edge_attribute: attr})
                for scr, tgt, attr in zip(source, target, edge_weight)
            ]

    def construct_graph(
        self,
        node_label,
        latitude,
        longitude,
        source,
        target,
        node_attribute=None,
        edge_attribute=None,
        edge_weight=None,
    ):
        nodes = self.construct_node(node_label, latitude, longitude, node_attribute)
        edges = self.construct_edge(source, target, edge_attribute, edge_weight)
        self.data.add_nodes_from(nodes)
        self.data.add_edges_from(edges)
        return self

    