from .base import BaseLayer
import pandas as pd
import geopandas as gpd


class LinestringLayer(BaseLayer):
    def __init__(self, name, file=None, year=None, column_list=[]) -> None:
        super().__init__(name, year)
        self.feature_name = column_list
        column_list.extend(['geometry'])
        if file is not None:
            self.data = gpd.read_file(file)
            self.data = self.data[[column_list]]
        else:
            self.data = gpd.GeoDataFrame(columns=column_list)

    def add_linestrings(self, data_gdf):
        self.data = pd.concat([self.data, data_gdf], ignore_index=True)


