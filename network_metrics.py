import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys

# Import Task A results
sys.path.append("D:/CW Newtrok/")
from network_construction import networks

# Function to compute network metrics
def compute_network_metrics(G):
    """
    Compute key network metrics.
    """
    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean([deg for _, deg in G.degree()]),
        "clustering_coefficient": nx.average_clustering(G),
        "avg_path_length": nx.average_shortest_path_length(G) if nx.is_connected(G) else None
    }
    return metrics

# Function to plot degree distribution
def plot_degree_distribution(G, title):
    """
    Plot the degree distribution of the network.
    """
    degrees = [deg for _, deg in G.degree()]
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(f"Degree Distribution: {title}")
    plt.yscale("log")
    plt.show()

# Function to compare with a random network
def compare_with_random_network(G, name):
    """
    Compare the real network with a random Erdős–Rényi network.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    p = (2 * num_edges) / (num_nodes * (num_nodes - 1))  # Probability for ER model
    
    random_G = nx.erdos_renyi_graph(num_nodes, p)
    
    comparison = {
        "real_avg_clustering": nx.average_clustering(G),
        "random_avg_clustering": nx.average_clustering(random_G),
        "real_avg_path_length": nx.average_shortest_path_length(G) if nx.is_connected(G) else None,
        "random_avg_path_length": nx.average_shortest_path_length(random_G) if nx.is_connected(random_G) else None,
    }
    print(f"Comparison for {name}:", comparison)

# Compute metrics, plot distributions, and compare networks
for name, G in networks.items():
    print(f"Metrics for {name}: {compute_network_metrics(G)}")
    plot_degree_distribution(G, name)
    compare_with_random_network(G, name)
