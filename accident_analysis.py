import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import Point, LineString
from libpysal import weights
from esda.moran import Moran
from pointpats import k
from pyproj import Transformer
import numpy as np

# -----------------------------
# 1. Load and preprocess accident data
# -----------------------------
folder_path = "C:/Users/86153/Downloads/"
file_names = [
    "2014.csv",
    "2015.csv",
    "RTC2018_Leeds.csv",
    "Trafficaccidents_2019_Leeds.csv"
]
df_list = [pd.read_csv(folder_path + name, encoding='latin1') for name in file_names]
accidents = pd.concat(df_list, ignore_index=True)
accidents.columns = [col.strip().lower() for col in accidents.columns]

print("Original column names:", accidents.columns.tolist())

# Remove rows with missing coordinates
accidents = accidents.dropna(subset=['grid ref: easting', 'grid ref: northing'])

# Coordinate transformation: from EPSG:27700 (British National Grid) to EPSG:4326 (WGS84)
transformer = Transformer.from_crs("epsg:27700", "epsg:4326", always_xy=True)
accidents['longitude'], accidents['latitude'] = transformer.transform(
    accidents['grid ref: easting'].values,
    accidents['grid ref: northing'].values
)

# Create GeoDataFrame (in WGS84)
geometry = [Point(xy) for xy in zip(accidents['longitude'], accidents['latitude'])]
gdf = gpd.GeoDataFrame(accidents, geometry=geometry, crs="EPSG:4326")

# -----------------------------
# 2. Load road network using OSMnx
# -----------------------------
center_point = (53.7965, -1.5478)
G = ox.graph_from_point(center_point, dist=1000, network_type='drive', simplify=True)
nodes, edges = ox.graph_to_gdfs(G)

# -----------------------------
# 3. Plot accident distribution
# -----------------------------
# Create a polygon from the union of all edge geometries to define the query area
polygon = edges.geometry.unary_union.convex_hull  
gdf_in_area = gdf[gdf.geometry.within(polygon)]

fig, ax = plt.subplots(figsize=(10, 10))
edges.plot(ax=ax, linewidth=0.5, color='gray')
gdf_in_area.plot(ax=ax, color='red', markersize=5, alpha=0.6)
plt.title("Road Accidents Distribution in Leeds Centre (1 km²)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# -----------------------------
# 4. Spatial autocorrelation analysis (Moran's I)
# -----------------------------
# Calculate the number of accidents at each unique geometry and add as a new column 'accident_count'
gdf_in_area.loc[:, 'accident_count'] = gdf_in_area.groupby('geometry')['reference number'].transform('count')
print("Filtered column names:", gdf_in_area.columns.tolist())

values = gdf_in_area['accident_count']

# Reproject the accident points to a projected CRS (EPSG:27700) for distance computations
gdf_in_area = gdf_in_area.to_crs(epsg=27700)
w = weights.KNN.from_dataframe(gdf_in_area, k=10)
w.transform = 'r'

moran = Moran(values, w, permutations=999)
print("\n✅ Moran's I value:", moran.I)
print("p-value:", moran.p_sim)

# -----------------------------
# 5. Accident to intersection distance analysis (using OSMnx network)
# -----------------------------
# Project the road network to a CRS in meters
G_proj = ox.project_graph(G)
edges_proj = ox.graph_to_gdfs(G_proj, nodes=False)

# Reproject accident points to the same CRS as the projected graph
gdf_in_area_proj = gdf_in_area.to_crs(G_proj.graph['crs'])

# Extract x and y coordinates from accident points
acc_x = gdf_in_area_proj.geometry.x.to_numpy()
acc_y = gdf_in_area_proj.geometry.y.to_numpy()

# Call nearest_edges to find the nearest edge for each accident point;
# result is a list of tuples (u, v, key)
result = ox.distance.nearest_edges(G_proj, X=acc_x, Y=acc_y, return_dist=False)

# Unpack the result using list comprehensions
u_array = [r[0] for r in result]
v_array = [r[1] for r in result]
key_array = [r[2] for r in result]

fractions = []  # Store the distance fraction for each accident point
for idx, (u, v, key) in enumerate(zip(u_array, v_array, key_array)):
    edge_data = G_proj.edges[u, v, key]
    if 'geometry' in edge_data:
        line = edge_data['geometry']
    else:
        # Construct a LineString from node coordinates if geometry is missing
        point_u = (G_proj.nodes[u]['x'], G_proj.nodes[u]['y'])
        point_v = (G_proj.nodes[v]['x'], G_proj.nodes[v]['y'])
        line = LineString([point_u, point_v])
    L = line.length
    pt = gdf_in_area_proj.geometry.iloc[idx]
    # Compute the projected distance of the accident point on the edge
    d = line.project(pt)
    # Calculate fraction: min(distance, (L - distance)) divided by L
    frac = min(d, L - d) / L if L > 0 else np.nan
    fractions.append(frac)

fractions = np.array(fractions)

plt.hist(fractions[~np.isnan(fractions)], bins=20, color='orange', edgecolor='black')
plt.title("Fractional Distance of Accidents to Nearest Intersection")
plt.xlabel("Fraction of road segment away from intersection")
plt.ylabel("Number of accidents")
plt.grid(True)
plt.show()




