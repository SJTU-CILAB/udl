<div align=center>
	<img src="./docs/logo.png" width="200"/>
</div>
<br />

UDL (UrbanDataLayer) is a suite of standard data structure and pipeline for city data engineering, which processes city data from various raw data into a unified data format.  

**UDL** is featured for:
- Unified standardized formats for city data: five data layers (grid, graph, point, polygon and linestring).
- User-friendly APIs of data processing: scheme transformation, granularity alignment and feature fusion.

## Usage

1. Install Python >= 3.8. For convenience, execute the following command.

```
pip install udlayer
```
2. Construct a UDL layer data. The full layer data formats and data processing APIs are available in the [**documentation**](https://urbandatalayer-doc.readthedocs.io/en/latest/).

```python
# Example: Construct a polygon layer data from a geojson file
from udlayer.layer.polygonlayer import PolygonLayer

polygonlayer = PolygonLayer("Vermont", "sample_Vermont.geojson", year=2014, column_list=['tag'])
print(polygonlayer.data)

# Example: Transform a tiff file to a grid layer data
from udlayer.transformation.transformation import *
from udlayer.alignment.alignment import *
from udlayer.utils.utility import *

griddata = tif_to_grid("pm2.5", ["Shanghai_pm2.5.tif"], start_lat=30.975, start_lon=121.1, end_lat=31.514, end_lon=121.804, year=2014)
print(griddata.data)
```

3. Transform between different data layers. 

```python
# Example: Transform a polygon layer to a graph layer
from udlayer.transformation.transformation import *

with open('sample_Vermont.pickle', 'rb') as f:
    polygon = pickle.load(f)
graph_data = polygon_to_graph(polygon)