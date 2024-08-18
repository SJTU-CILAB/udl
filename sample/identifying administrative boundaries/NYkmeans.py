from datalayer import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from shapely.geometry import Point, Polygon
import matplotlib.cm as cm

# Create a figure and axes
fig, ax = plt.subplots()

import random
import shapefile

# Load the shapefile
sf = shapefile.Reader("NewYorkState.shp")


# Loop through the shapes in the shapefile
for shape in sf.shapes():
    color = (random.random(), random.random(), random.random())
    # Get the list of points
    points = shape.points
    parts = shape.parts
    parts.append(len(points))  # Add the end point

    # Loop through each part
    for i in range(len(parts) - 1):
        # Get the points for this part
        part_points = points[parts[i]:parts[i+1]]
        
        # Convert the points to arrays
        x = [point[0] for point in part_points]
        y = [point[1] for point in part_points]
        
        # Plot the part
        ax.plot(x, y,color=color)

points=pd.read_csv('NYpoi.csv')
points=points[['longitude','latitude']]
points=points.where((points['longitude']>-75)&(points['longitude']<-73.5)&(points['latitude']<41)).dropna()
grid = pickle.load(open("NewYorkState_inter.pickle",'rb')) 

# precipitation = pd.read_csv('precipitation.csv')
high_node = grid.data.max() / 5000

# start_lat = min(precipitation['latitude'])
# end_lat = max(precipitation['latitude'])
# start_lon = min(precipitation['longitude'])
# end_lon = max(precipitation['longitude'])
### edit start_lat, end_lat, start_lon, end_lon ###
### Shanghai ###
# start_lat = 30
# end_lat = 32.4
# start_lon = 120
# end_lon = 122
### NewYorkState ###
start_lat = 40
end_lat = 45
start_lon = -80
end_lon = -70
### NewYorkCity ###
# start_lat = 40.4
# end_lat = 41
# start_lon = -74.3
# end_lon = -73.7

step_lat = 0.05
step_lon = 0.05
# Process discontinuous grid data
lat_count = round((end_lat - start_lat) / step_lat)
lon_count = round((end_lon - start_lon) / step_lon)

# Initialize grid_data and count
grid_data = grid.data
# # Calculate row and column indices
# data['row_index'] = ((end_lat - data['latitude']) / step_lat).astype(int)
# data['col_index'] = ((data['longitude'] - start_lon) / step_lon).astype(int)

# # Filter out data that is not within the range
# data = data[(data['latitude'] >= start_lat) & (data['latitude'] <= end_lat) & 
#             (data['longitude'] >= start_lon) & (data['longitude'] <= end_lon)]

# # Use groupby to accumulate
# grouped = data.groupby(['row_index', 'col_index'])[year].agg(['sum', 'count'])

# # Update grid_data and count
# for index, row in grouped.iterrows():
#     grid_data[index] = row['sum']
#     count[index] = row['count']

# # Calculate the average
# grid_data = grid_data / count

lats = np.arange(start_lat, end_lat, step_lat)
lons = np.arange(end_lon, start_lon, -step_lon)
import random
random.seed(42)
new_points = []
for i in range(lat_count-1):
    for j in range(lon_count-1):
        if np.isnan(grid_data[i,j]) or grid_data[i,j] == 0:
            continue
        lat, lon = lats[i],lons[j]
        lat_, lon_ = lats[i+1],lons[j+1]
        for k in range(int(grid_data[i,j]/high_node)):
            new_points.append((random.uniform(lon, lon_), random.uniform(lat, lat_)))


from shapely.geometry import Point, Polygon

# Convert shapefile shapes to shapely Polygon objects
shapes = [shape for shape in sf.shapes()]
shape_polygons = [Polygon(shape.points) for shape in shapes]

# Initialize an empty list to store points within the shapefile
points_in_sf = []

# Check each newly generated point
for point in new_points:
    point_obj = Point(point)
    for polygon in shape_polygons:
        if polygon.contains(point_obj):
            points_in_sf.append(point)
            break
print(points)
points = pd.concat([points, pd.DataFrame(points_in_sf, columns=['longitude','latitude'])], ignore_index=True)


grid = pickle.load(open("NewYorkState_light_imputed.pickle",'rb')) 

# precipitation = pd.read_csv('precipitation.csv')
high_node = grid.data.max() / 5000

# start_lat = min(precipitation['latitude'])
# end_lat = max(precipitation['latitude'])
# start_lon = min(precipitation['longitude'])
# end_lon = max(precipitation['longitude'])
### edit start_lat, end_lat, start_lon, end_lon ###
### Shanghai ###
# start_lat = 30
# end_lat = 32.4
# start_lon = 120
# end_lon = 122
### NewYorkState ###
start_lat = 40
end_lat = 45
start_lon = -80
end_lon = -70
### NewYorkCity ###
# start_lat = 40.4
# end_lat = 41
# start_lon = -74.3
# end_lon = -73.7

step_lat = 0.05
step_lon = 0.05
# Process discontinuous grid data
lat_count = round((end_lat - start_lat) / step_lat)
lon_count = round((end_lon - start_lon) / step_lon)

# Initialize grid_data and count
grid_data = grid.data
# # Calculate row and column indices
# data['row_index'] = ((end_lat - data['latitude']) / step_lat).astype(int)
# data['col_index'] = ((data['longitude'] - start_lon) / step_lon).astype(int)

# # Filter out data that is not within the range
# data = data[(data['latitude'] >= start_lat) & (data['latitude'] <= end_lat) & 
#             (data['longitude'] >= start_lon) & (data['longitude'] <= end_lon)]

# # Use groupby to accumulate
# grouped = data.groupby(['row_index', 'col_index'])[year].agg(['sum', 'count'])

# # Update grid_data and count
# for index, row in grouped.iterrows():
#     grid_data[index] = row['sum']
#     count[index] = row['count']

# # Calculate the average
# grid_data = grid_data / count

lats = np.arange(start_lat, end_lat, step_lat)
lons = np.arange(end_lon, start_lon, -step_lon)
import random
random.seed(42)
new_points = []
for i in range(lat_count-1):
    for j in range(lon_count-1):
        if np.isnan(grid_data[i,j]) or grid_data[i,j] == 0:
            continue
        lat, lon = lats[i],lons[j]
        lat_, lon_ = lats[i+1],lons[j+1]
        for k in range(int(grid_data[i,j]/high_node)):
            new_points.append((random.uniform(lon, lon_), random.uniform(lat, lat_)))


from shapely.geometry import Point, Polygon

# Convert shapefile shapes to shapely Polygon objects
shapes = [shape for shape in sf.shapes()]
shape_polygons = [Polygon(shape.points) for shape in shapes]

# Initialize an empty list to store points within the shapefile
points_in_sf = []

# Check each newly generated point
for point in new_points:
    point_obj = Point(point)
    for polygon in shape_polygons:
        if polygon.contains(point_obj):
            points_in_sf.append(point)
            break
print(points)
points = pd.concat([points, pd.DataFrame(points_in_sf, columns=['longitude','latitude'])], ignore_index=True)


grid = pickle.load(open("NewYorkState_pop_imputed.pickle",'rb')) 

# precipitation = pd.read_csv('precipitation.csv')
high_node = grid.data.max() / 5000

# start_lat = min(precipitation['latitude'])
# end_lat = max(precipitation['latitude'])
# start_lon = min(precipitation['longitude'])
# end_lon = max(precipitation['longitude'])
### edit start_lat, end_lat, start_lon, end_lon ###
### Shanghai ###
# start_lat = 30
# end_lat = 32.4
# start_lon = 120
# end_lon = 122
### NewYorkState ###
start_lat = 40
end_lat = 45
start_lon = -80
end_lon = -70
### NewYorkCity ###
# start_lat = 40.4
# end_lat = 41
# start_lon = -74.3
# end_lon = -73.7

step_lat = 0.05
step_lon = 0.05
# Process discontinuous grid data
lat_count = round((end_lat - start_lat) / step_lat)
lon_count = round((end_lon - start_lon) / step_lon)

# Initialize grid_data and count
grid_data = grid.data
# # Calculate row and column indices
# data['row_index'] = ((end_lat - data['latitude']) / step_lat).astype(int)
# data['col_index'] = ((data['longitude'] - start_lon) / step_lon).astype(int)

# # Filter out data that is not within the range
# data = data[(data['latitude'] >= start_lat) & (data['latitude'] <= end_lat) & 
#             (data['longitude'] >= start_lon) & (data['longitude'] <= end_lon)]

# # Use groupby to accumulate
# grouped = data.groupby(['row_index', 'col_index'])[year].agg(['sum', 'count'])

# # Update grid_data and count
# for index, row in grouped.iterrows():
#     grid_data[index] = row['sum']
#     count[index] = row['count']

# # Calculate the average
# grid_data = grid_data / count

lats = np.arange(start_lat, end_lat, step_lat)
lons = np.arange(end_lon, start_lon, -step_lon)
import random
random.seed(42)
new_points = []
for i in range(lat_count-1):
    for j in range(lon_count-1):
        if np.isnan(grid_data[i,j]) or grid_data[i,j] == 0:
            continue
        lat, lon = lats[i],lons[j]
        lat_, lon_ = lats[i+1],lons[j+1]
        for k in range(int(grid_data[i,j]/high_node)):
            new_points.append((random.uniform(lon, lon_), random.uniform(lat, lat_)))


from shapely.geometry import Point, Polygon

# Convert shapefile shapes to shapely Polygon objects
shapes = [shape for shape in sf.shapes()]
shape_polygons = [Polygon(shape.points) for shape in shapes]

# Initialize an empty list to store points within the shapefile
points_in_sf = []

# Check each newly generated point
for point in new_points:
    point_obj = Point(point)
    for polygon in shape_polygons:
        if polygon.contains(point_obj):
            points_in_sf.append(point)
            break
print(points)
points = pd.concat([points, pd.DataFrame(points_in_sf, columns=['longitude','latitude'])], ignore_index=True)


print(points)
model=KMeans(n_init='auto',n_clusters=5, random_state=22)
model.fit(points)
labels=model.labels_
centers=model.cluster_centers_
points['cluster']=labels
count=points.cluster.value_counts()
plt.rc('font',family='SimHei')
plt.rcParams['axes.unicode_minus']=False
sampled_points = points.sample(frac=0.1, random_state=42)

# Plot the sampled points and centers
ax.scatter(sampled_points['longitude'], sampled_points['latitude'], c=sampled_points['cluster'], s=0.1, cmap=cm.GnBu)
ax.scatter(centers[:,0],centers[:,1],c='r',marker='x')
plt.savefig('NYcluster_inter+light.pdf')
# plt.show()
# pd.DataFrame(points).to_csv('NYcluster.csv',index=False)


from shapely.geometry import MultiPoint

# Calculate the convex hull of each cluster
clusters = points.groupby('cluster')
convex_hulls = clusters.apply(lambda cluster: MultiPoint(cluster[['longitude', 'latitude']].values).convex_hull)

# Convert shapefile shapes to shapely shapes and fix invalid polygons
shapes = [shape for shape in sf.shapes()]
shape_polygons = [Polygon(shape.points) for shape in shapes]
shape_polygons = [polygon if polygon.is_valid else polygon.buffer(0) for polygon in shape_polygons]

# Calculate the area of each polygon
areas = [polygon.area for polygon in shape_polygons]

# Calculate the IoU of each cluster with each shapefile polygon and take the maximum value
def calculate_max_iou(hull):
    ious = [hull.intersection(polygon).area / hull.union(polygon).area for polygon in shape_polygons]
    return max(ious)

max_ious = convex_hulls.apply(calculate_max_iou)

# Weight the IoU by the area of each polygon and sum
weighted_iou = sum(iou * area for iou, area in zip(max_ious, areas)) / sum(areas)

print("iou: ",weighted_iou)

def calculate_max_accuracy(cluster):
    # Calculate accuracy for each polygon
    ret=0
    for polygon in shape_polygons:
        tot=0
        for i in range(len(cluster)):
            if polygon.contains(Point(cluster.iloc[i]['longitude'], cluster.iloc[i]['latitude'])):
                tot+=1
        ret=max(ret,tot/len(cluster))
    # Return the maximum accuracy
    return ret

# Calculate the maximum accuracy of each cluster
max_accuracies = clusters.apply(calculate_max_accuracy)

# Weight the maximum accuracy by the area of each polygon and sum
weighted_max_accuracy = sum(max_accuracy * area for max_accuracy, area in zip(max_accuracies, areas)) / sum(areas)

print('accuracy:',weighted_max_accuracy)