# network_construction.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# File paths for the 3 datasets
data_files = {
    "ADMINISTRATORS": "C:/Users/86153/Desktop/ADMINISTRATORS.csv",
    "BOT_REQUESTS": "C:/Users/86153/Desktop/BOT_REQUESTS.csv",
    "REQUEST_FOR_DELETION": "C:/Users/86153/Desktop/REQUEST_FOR_DELETION.csv"
}

# Load datasets
datasets = {name: pd.read_csv(path) for name, path in data_files.items()}

# Build the Wikidata editor network
def build_network(df):
    G = nx.Graph()
    for (_, group) in df.groupby(["page_name", "thread_subject"]):
        users = group["username"].unique()
        for i, user1 in enumerate(users):
            for user2 in users[i + 1:]:
                G.add_edge(user1, user2)
    return G

# Construct networks
networks = {name: build_network(df) for name, df in datasets.items()}

# Print basic stats
network_stats = {
    name: {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G)
    }
    for name, G in networks.items()
}
print(network_stats)

# Optional visualization (sampled)
def plot_network(G, title, sample_size=200):
    plt.figure(figsize=(10, 7))
    subgraph = G.subgraph(list(G.nodes)[:sample_size])
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw(subgraph, pos, node_size=20, edge_color="gray", alpha=0.6)
    plt.title(title)
    plt.show()

for name, G in networks.items():
    plot_network(G, f"{name} Network (Sampled)")

# Save graphs as .pkl for later use
with open("admin_graph.pkl", "wb") as f:
    pickle.dump(networks["ADMINISTRATORS"], f)
with open("bot_graph.pkl", "wb") as f:
    pickle.dump(networks["BOT_REQUESTS"], f)
with open("deletion_graph.pkl", "wb") as f:
    pickle.dump(networks["REQUEST_FOR_DELETION"], f)

print("Graphs have been saved successfully.")
