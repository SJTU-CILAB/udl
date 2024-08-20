import shapefile
import random
from udlayer.layer import *
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from shapely.geometry import Point, Polygon
import matplotlib.cm as cm
random.seed(0)
# Create a figure and axes
plt.rc('font', family='monospace')
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()

# Load the shapefile
sf = shapefile.Reader("NewYorkState.shp")


# Loop through the shapes in the shapefile
def plot_sf():
    for i in range(len(sf.shapes())):
        shape = sf.shape(i)
        color=cm.tab10(i)
        # Get the list of points
        shape_points = shape.points
        parts = shape.parts
        parts.append(len(shape_points))  # Add the end point

        # Loop through each part
        for i in range(len(parts)-1):
            # Get the points for this part
            part_points = shape_points[parts[i]:parts[i+1]]

            # Convert the shape_points to arrays
            x = [point[0] for point in part_points]
            y = [point[1] for point in part_points]

            # Plot the part
            ax.plot(x, y, color=color,linewidth=1)

new_points=pd.read_csv('NYpoi.csv')
new_points = new_points[new_points['city'] == "New York"]       
# new_points=pd.read_csv('NYCitypoints.csv')
new_points = new_points.loc[:, ['longitude','latitude']].values.tolist()
new_points = random.sample(new_points, int(len(new_points) / 1))

#print(new_points)
points_in_sf = []
shapes = [sf.shape(i) for i in range(len(sf.shapes()))]
shape_polygons = [Polygon(shape.points) for shape in shapes]
shape_polygons = [polygon if polygon.is_valid else polygon.buffer(0) for polygon in shape_polygons]

# Initialize an empty list to store points within the shapefile
points_in_sf = []

# Check each newly generated point
for point in new_points:
#    print(point)
    point_obj = Point(point)
    for polygon in shape_polygons:
        if polygon.contains(point_obj):
            points_in_sf.append(point)
            break
points_in_sf=pd.DataFrame(points_in_sf, columns=['longitude','latitude'])#.to_csv(name+'pois.csv',index=False)
points_in_sf.dropna(inplace=True)
#points=pd.read_csv('NewYorkStatepois.csv')
# name="NewYorkStateGauss_inter"
#points = points.data.loc[:, ['lat', 'lon']]
# points = pd.concat([points, pd.read_csv("NewYorkStateGauss_allinter.csv")], ignore_index=True)
#points = pd.concat([points, pd.read_csv("NewYorkStateGauss_alllight.csv")], ignore_index=True)
#points = pd.concat([points, pd.read_csv("NewYorkStateGauss_allpop.csv")], ignore_index=True)

def sprinkle(grid,density=100000):
    high_node = grid.data[~np.isnan(grid.data)].max() / density
    
    ### edit start_lat, end_lat, start_lon, end_lon ###
    ### Shanghai ###
    # start_lat = 30
    # end_lat = 32.4
    # start_lon = 120
    # end_lon = 122
    ### NewYorkState ###
    # start_lat = 40
    # end_lat = 45
    # start_lon = -80
    # end_lon = -70
    ### NewYorkCity ###
    start_lat = 40.4
    end_lat = 41
    start_lon = -74.3
    end_lon = -73.7

    step_lat = 0.05
    step_lon = 0.05
    # Handle discontinuous grid data
    lat_count = round((end_lat - start_lat) / step_lat)
    lon_count = round((end_lon - start_lon) / step_lon)

    # Initialize grid_data and count
    grid_data = grid.data

    lats = np.arange(start_lat, end_lat, step_lat)
    lons = np.arange(end_lon, start_lon, -step_lon)
    import random
    random.seed(21)
    new_points = []
    for i in range(lat_count-1):
        for j in range(lon_count-1):
            if np.isnan(grid_data[i,j]) or grid_data[i,j] == 0:
                continue
            lat, lon = lats[i],lons[j]
            lat_, lon_ = lats[i+1],lons[j+1]
            for k in range(int(grid_data[i,j]/high_node)):
                new_points.append((random.uniform(lon, lon_), random.uniform(lat, lat_)))


    # Initialize an empty list to store points within the shapefile
    points_in_sf = []

    # Check each newly generated point
    for point in new_points:
        point_obj = Point(point)
        for polygon in shape_polygons:
            if polygon.contains(point_obj):
                points_in_sf.append(point)
                break

    pd.DataFrame(points_in_sf, columns=['lon', 'lat']).to_csv(name+'light.csv',index=False)
    global points
    points = pd.concat([points, pd.DataFrame(points_in_sf, columns=['lon', 'lat'])], ignore_index=True)
    
def Main():
    plot_sf()
    global name
    global points
    # Fit the model
    model = GaussianMixture(n_components=5,init_params='k-means++',random_state=0)
    # model = KMeans(n_clusters=5,n_init='auto', random_state=0)
    labels=model.fit_predict(points)
    points['cluster'] = labels
    # Calculate cluster centers
    #centers = points.groupby('cluster').mean()
    count = points.cluster.value_counts()
    sampled_points = points.sample(frac=1, random_state=0)

    # Plot the sampled points and centers
    ax.scatter(sampled_points['longitude'], sampled_points['latitude'], c=sampled_points['cluster'], s=2, cmap=cm.GnBu)

    #ax.scatter(centers['lon'], centers['lat'], c='r', marker='x')

    #points.to_csv('poi_cluster_GaussianMixture.csv', index=False)

    plt.savefig(name+'.pdf')
    # plt.show()


    from shapely.geometry import MultiPoint

    # Calculate the convex hull for each cluster
    clusters = points.groupby('cluster')
    convex_hulls = clusters.apply(lambda cluster: MultiPoint(cluster[['longitude', 'latitude']].values).convex_hull)

    # Convert the shapes in the shapefile to shapely shapes and fix invalid polygons
    shapes = [sf.shape(i) for i in range(len(sf.shapes())-1)]

    shape_polygons = [Polygon(shape.points) for shape in shapes]
    shape_polygons = [polygon if polygon.is_valid else polygon.buffer(0) for polygon in shape_polygons]

    # Calculate the area of each polygon
    areas = [polygon.area for polygon in shape_polygons]

    # Calculate the IoU for each cluster with each shapefile polygon, then take the maximum value
    def calculate_max_iou(hull):
        ious = [hull.intersection(polygon).area / hull.union(polygon).area for polygon in shape_polygons]
        return max(ious)

    max_ious = convex_hulls.apply(calculate_max_iou)

    # Weight the IoU by the area of each polygon and sum
    weighted_iou = sum(iou * area for iou, area in zip(max_ious, areas)) / sum(areas)
    occupied=[0]*len(shape_polygons)
    
    def calculate_max_accuracy(cluster):
        # Calculate accuracy for each polygon
        ret=0
        now=None
        for j in range(len(shape_polygons)):
            polygon=shape_polygons[j]
            tot=0
            for i in range(len(cluster)):
                if polygon.contains(Point(cluster.iloc[i]['longitude'], cluster.iloc[i]['latitude'])):
                    tot+=1
            if tot/len(cluster) > ret and (now==None or not occupied[now]):
                ret=tot/len(cluster)
                if now!=None:
                    occupied[now]=0
                now=j
                occupied[now]=1
        # Return the maximum accuracy
        return ret

    # Calculate the maximum accuracy for each cluster
    max_accuracies = clusters.apply(calculate_max_accuracy)

    # Weight the maximum accuracy by the area of each polygon and sum
    weighted_max_accuracy = sum(max_accuracy * area for max_accuracy, area in zip(max_accuracies, areas)) / sum(areas)

    print(name+": {:.3f}".format(weighted_max_accuracy),end=' &')
    print('{: .3f}'.format(weighted_iou))
    plt.cla()
    
name='NewYorkState_imputeS_raw'
points=points_in_sf.copy()
Main()

name='NewYorkState_imputeS_inter'
points=points_in_sf.copy()
sprinkle(pickle.load(open("NewYorkState_inter.pickle",'rb')))
Main()

# name='NewYorkState_imputeS_light'
# points=points_in_sf.copy()
# sprinkle(pickle.load(open("NewYorkState_light_imputedS.pickle",'rb')))
# Main()

# name='NewYorkState_imputeS_pop'
# points=points_in_sf.copy()
# sprinkle(pickle.load(open("NewYorkState_pop_imputedS.pickle",'rb')))
# Main()

# name='NewYorkState_imputeS_inter+light'
# points=points_in_sf.copy()
# sprinkle(pickle.load(open("NewYorkState_light_imputedS.pickle",'rb')))
# sprinkle(pickle.load(open("NewYorkState_inter.pickle",'rb')))
# Main()

# name='NewYorkState_imputeS_light+pop'
# points=points_in_sf.copy()
# sprinkle(pickle.load(open("NewYorkState_light_imputedS.pickle",'rb')))
# sprinkle(pickle.load(open("NewYorkState_pop_imputedS.pickle",'rb')))
# Main()

# name='NewYorkState_imputeS_inter+pop'
# points=points_in_sf.copy()
# sprinkle(pickle.load(open("NewYorkState_pop_imputedS.pickle",'rb')))
# sprinkle(pickle.load(open("NewYorkState_inter.pickle",'rb')))
# Main()

name='NewYorkState_imputeS_all'
points=points_in_sf.copy()
sprinkle(pickle.load(open("NewYorkState_light_imputedS.pickle",'rb')))
sprinkle(pickle.load(open("NewYorkState_pop_imputedS.pickle",'rb')))
sprinkle(pickle.load(open("NewYorkState_inter.pickle",'rb')))
Main()
