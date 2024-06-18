from .base import BaseLayer
import numpy as np


class GridLayer(BaseLayer):
    def __init__(
        self,
        name,
        start_lat,
        end_lat,
        start_lon,
        end_lon,
        step_lat,
        step_lon,
        year=None,
    ):
        super(GridLayer, self).__init__(name, year)
        self.start_lat = start_lat
        self.end_lat = end_lat
        self.start_lon = start_lon
        self.end_lon = end_lon
        self.step_lat = step_lat
        self.step_lon = step_lon
        self.num_step_lat = round((end_lat - start_lat) / step_lat)
        self.num_step_lon = round((end_lon - start_lon) / step_lon)
        self.data = np.full(
            shape=(self.num_step_lat, self.num_step_lon), fill_value=np.nan
        )

    def construct_grid(self, data):
        self.data = data
        return self

    def get_value(self, lat, lon):
        row = int((self.end_lat - lat) / self.step_lat)
        col = int((lon - self.start_lon) / self.step_lon)
        if row < 0 or row >= self.num_step_lat or col < 0 or col >= self.num_step_lon:
            return np.nan
        else:
            return self.data[row, col]

    def get_value_by_grid(self, lat, lon):
        row = np.round((self.end_lat - lat) / self.step_lat).astype(int)
        col = np.round((lon - self.start_lon) / self.step_lon).astype(int)
        mask = (
            (row < 0)
            | (row >= self.num_step_lat)
            | (col < 0)
            | (col >= self.num_step_lon)
        )
        result = np.full_like(lat, fill_value=np.nan)
        result[~mask] = self.data[row[~mask], col[~mask]]
        return result
