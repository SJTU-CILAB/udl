from .base import BaseLayer
import pandas as pd
import geopandas as gpd


class PointLayer(BaseLayer):
    """columns_list: list of column names in the csv file"""
    def __init__(self, name, year=None, column_list=[]):
        super(PointLayer, self).__init__(name, year)
        self.feature_name = column_list
        if "lat" not in column_list:
            column_list.append("lat")
        if "lon" not in column_list:
            column_list.append("lon")
        if "latitude" in column_list or "longitude" in column_list:
            print("Warning: latitude and longitude are not supported, use lat and lon instead")
            try:
                column_list.remove("latitude")
                column_list.remove("longitude")
            except ValueError:
                pass
        self.data = pd.DataFrame(columns=column_list)

    def add_points(self, data_df):
        self.data = pd.concat([self.data, data_df], ignore_index=True)

    def delete_point(self, lat, lon):
        self.data = self.data.drop(
            self.data[
                (self.data["lat"] == lat) & (self.data["lon"] == lon)
            ].index
        )

    def get_value(self, lat, lon, feature_name=None):
        if feature_name is None:
            feature_name = self.feature_name
        return self.data[(self.data["lat"] == lat) & (self.data["lon"] == lon)][
            self.data[feature_name]
        ].to_numpy() 

    def get_value_by_range(self, start_lat, end_lat, start_lon, end_lon, feature_name=None):
        if feature_name is None:
            feature_name = self.feature_name
        return self.data[
            (self.data["lat"] >= start_lat)
            & (self.data["lat"] <= end_lat)
            & (self.data["lon"] >= start_lon)
            & (self.data["lon"] <= end_lon)
        ][self.data[feature_name]].to_numpy()
    
    def to_gpd(self):
        return gpd.GeoDataFrame(
            self.data, geometry=gpd.points_from_xy(self.data.lon, self.data.lat), crs="EPSG:4326")
