import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import random

# --- Load road network ---
city_center = (53.8008, -1.5491)  # Leeds city centre
G = ox.graph_from_point(city_center, dist=6000, network_type='walk', simplify=True)
nodes, edges = ox.graph_to_gdfs(G)

# --- 1. Select 4 seed points ---
# Criteria: Evenly spread out locations far from known accident clusters (mock for now)
# These are manually chosen for diversity
seed_coords = [
    (53.835, -1.55),  # North Leeds
    (53.795, -1.5),   # East Leeds
    (53.76, -1.56),   # South Leeds
    (53.79, -1.61)    # West Leeds
]

seed_points = [Point(lon, lat) for lat, lon in seed_coords]
seeds_gdf = gpd.GeoDataFrame(geometry=seed_points, crs='EPSG:4326')

# --- 2. Generate Voronoi diagram ---
# Project to UTM for planar computations
seeds_proj = seeds_gdf.to_crs(epsg=27700)
points_proj = np.array([[p.x, p.y] for p in seeds_proj.geometry])
vor = Voronoi(points_proj)

# Plot Voronoi diagram
fig, ax = plt.subplots(figsize=(10, 10))
seeds_proj.plot(ax=ax, color='red')
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue')
plt.title("Voronoi Diagram for 4 Marathon Zones in Leeds")
plt.show()

# --- 3. Find loops of ~42km starting and ending at seed nodes ---
def find_circuit(G, center_node, target_length_km=42, tolerance=2):
    paths = []
    for _ in range(100):
        path = [center_node]
        total_length = 0
        current_node = center_node
        while total_length < (target_length_km * 1000):
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            try:
                edge_length = G[current_node][next_node][0]['length']
            except:
                edge_length = G[current_node][next_node]['length']
            total_length += edge_length
            path.append(next_node)
            current_node = next_node
            if current_node == center_node and abs(total_length - target_length_km * 1000) < tolerance * 1000:
                paths.append((path, total_length))
                break
    return paths

# Find closest nodes to seed points
seed_node_ids = [ox.nearest_nodes(G, point.x, point.y) for point in seeds_gdf.geometry]

fig, ax = plt.subplots(figsize=(12, 12))
nodes.plot(ax=ax, markersize=1, color='gray')
edges.plot(ax=ax, linewidth=0.5, color='lightgray')
colors = ['red', 'green', 'blue', 'purple']

for i, node_id in enumerate(seed_node_ids):
    circuits = find_circuit(G, node_id)
    if circuits:
        path, length = circuits[0]
        ox.plot_graph_route(G, path, route_color=colors[i], route_linewidth=2, node_size=0, show=False, close=False, ax=ax)
        print(f"✅ Cell {i+1}: Found path of {length/1000:.2f} km")
    else:
        print(f"❌ Cell {i+1}: No path found close to 42 km")

plt.title("Marathon Loops within Voronoi Cells")
plt.show()

