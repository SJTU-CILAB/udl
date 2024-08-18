import shapefile
import random
from datalayer import *
from sklearn.cluster import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from shapely.geometry import Point, Polygon
# Create a figure and axes
fig, ax = plt.subplots()

# Load the shapefile
sf = shapefile.Reader("Shanghai.shp")


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
        ax.plot(x, y, color=color)

points = pickle.load(open("Shanghai_poi.pickle", 'rb'))
points = points.data.loc[:, ['lat', 'lon']]
print(points)
# Fit the model
model=KMeans(n_init='auto',n_clusters=16, random_state=22)
model.fit(points)
labels=model.labels_
points['cluster'] = labels
centers=model.cluster_centers_
count = points.cluster.value_counts()
plt.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False
ax.scatter(points['lon'], points['lat'], c=points['cluster'])
ax.scatter(centers[:,0],centers[:,1],c='r',marker='x')

points.to_csv('poi_cluster_spectual_clustering.csv', index=False)

plt.savefig('Shanghai_kmeans.pdf')
plt.show()

from shapely.geometry import MultiPoint

# Calculate the convex hull of each cluster
clusters = points.groupby('cluster')
convex_hulls = clusters.apply(lambda cluster: MultiPoint(cluster[['lon', 'lat']].values).convex_hull)

# Convert the shapes of the shapefile to shapely shapes and fix invalid polygons
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
            if polygon.contains(Point(cluster.iloc[i]['lon'], cluster.iloc[i]['lat'])):
                tot+=1
        ret=max(ret,tot/len(cluster))
    # Return the maximum accuracy
    return ret

# Calculate the maximum accuracy of each cluster
max_accuracies = clusters.apply(calculate_max_accuracy)

# Weight the maximum accuracy by the area of each polygon and sum
weighted_max_accuracy = sum(max_accuracy * area for max_accuracy, area in zip(max_accuracies, areas)) / sum(areas)

print('accuracy:',weighted_max_accuracy)