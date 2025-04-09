import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
from pyproj import Transformer
import networkx as nx  # For planarity check

# Set folder path where your CSV files are stored
folder_path = "C:/Users/86153/Downloads/"
file_names = [
    "2014.csv",
    "2015.csv",
    "RTC2018_Leeds.csv",
    "Trafficaccidents_2019_Leeds.csv"
]

# Read all accident data files (with appropriate encoding)
df_list = [pd.read_csv(folder_path + name, encoding='latin1') for name in file_names]
accidents = pd.concat(df_list, ignore_index=True)

# Clean column names (lowercase and strip spaces)
accidents.columns = [col.strip().lower() for col in accidents.columns]
print("Column names after cleaning:", accidents.columns)

# Drop rows with missing Easting/Northing values
accidents = accidents.dropna(subset=['grid ref: easting', 'grid ref: northing'])

# Convert OS National Grid (EPSG:27700) to Latitude/Longitude (EPSG:4326)
transformer = Transformer.from_crs("epsg:27700", "epsg:4326", always_xy=True)
accidents['longitude'], accidents['latitude'] = transformer.transform(
    accidents['grid ref: easting'].values,
    accidents['grid ref: northing'].values
)

# Create a GeoDataFrame from the accident data
geometry = [Point(xy) for xy in zip(accidents['longitude'], accidents['latitude'])]
gdf = gpd.GeoDataFrame(accidents, geometry=geometry, crs="EPSG:4326")

# Define the central point of Leeds (latitude, longitude)
center_point = (53.7965, -1.5478)

# Download the drivable road network within 1 km of Leeds centre
G = ox.graph_from_point(center_point, dist=1000, network_type='drive', simplify=True)

# Calculate and display basic network statistics
stats = ox.basic_stats(G)
print("\n✅ Basic Network Statistics:")
for key, val in stats.items():
    if isinstance(val, (int, float)):
        print(f"{key}: {val:.4f}")
    else:
        print(f"{key}: {val}")

print(f"\n✅ Average Circuity: {stats['circuity_avg']:.4f}")

# Check if the network is planar using networkx
is_planar, _ = nx.check_planarity(G)
print(f"\n✅ Is the network planar? {is_planar}")
if not is_planar:
    print("Explanation: The network includes overpasses or bridges, which introduce edge crossings, so it's non-planar.")

# Get the nodes as a GeoDataFrame (no need to unpack since edges=False)
nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
# Compute the convex hull of the nodes to form a boundary polygon
polygon = nodes.unary_union.convex_hull

# Filter accidents within the polygon (Leeds centre area)
accidents_within = gdf[gdf.geometry.within(polygon)]
print(f"\n✅ Number of accidents within the selected area: {len(accidents_within)}")

# Plot accidents on top of the road network
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the road network
ox.plot_graph(G, ax=ax, show=False, close=False, node_color='black', edge_color='gray')
# Plot accidents on top
accidents_within.plot(ax=ax, markersize=5, color='red', alpha=0.5, label='Accidents')

plt.title("Road Network and Accident Locations in Central Leeds (1 km²)")
plt.legend()
plt.show()



