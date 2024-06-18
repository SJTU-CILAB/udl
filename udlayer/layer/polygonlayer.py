from .base import BaseLayer
from shapely.geometry import Polygon, MultiPolygon
import pandas as pd
import geopandas as gpd

class PolygonLayer(BaseLayer):
    def __init__(self, name, file=None, year=None, column_list=[]) -> None:
        super().__init__(name, year)
        self.feature_name = column_list
        column_list.extend(['geometry'])
        if file is not None:
            self.data = gpd.read_file(file)
            try:
                self.data = self.data[column_list]
            except KeyError:
                print("Warning: column_list is not valid, the column name should be in the file \"properties\"")
                print("Return all columns in the file")
        else:
            self.data = gpd.GeoDataFrame(columns=column_list)

    def add_polygons(self, add_gdf):
        self.data = pd.concat([self.data, add_gdf], ignore_index=True)

