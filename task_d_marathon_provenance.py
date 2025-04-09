
# Task D - Provenance Modeling, PageRank, and Knowledge Graph Embeddings

import networkx as nx
import matplotlib.pyplot as plt

# --- Step 1: Build PROV-style Network for Marathons in Leeds ---

G = nx.DiGraph()

# Define Entities, Activities, and Agents
entities = ['MarathonRoute', 'VolunteerList', 'TrafficReport']
activities = ['OrganiseMarathon', 'EvaluateSafety', 'AssignVolunteers']
agents = ['MayorOffice', 'SafetyOfficer', 'VolunteerTeam']

for e in entities:
    G.add_node(e, type='entity')
for a in activities:
    G.add_node(a, type='activity')
for ag in agents:
    G.add_node(ag, type='agent')

# Add PROV relationships
G.add_edge('MayorOffice', 'OrganiseMarathon', label='wasAssociatedWith')
G.add_edge('OrganiseMarathon', 'MarathonRoute', label='used')
G.add_edge('OrganiseMarathon', 'VolunteerList', label='generated')
G.add_edge('SafetyOfficer', 'EvaluateSafety', label='wasAssociatedWith')
G.add_edge('EvaluateSafety', 'TrafficReport', label='used')
G.add_edge('VolunteerTeam', 'AssignVolunteers', label='wasAssociatedWith')
G.add_edge('AssignVolunteers', 'VolunteerList', label='used')

# Visualization
pos = nx.spring_layout(G, seed=42)
node_colors = ['skyblue' if G.nodes[n]['type'] == 'entity' else 'lightgreen' if G.nodes[n]['type'] == 'activity' else 'salmon' for n in G.nodes()]
labels = nx.get_edge_attributes(G, 'label')

plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, font_size=10, arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='gray')
plt.title("W3C PROV Model for Leeds Marathon")
plt.show()

# --- Step 2: Compute PageRank on the PROV Network ---

pagerank = nx.pagerank(G, alpha=0.85)

print("\n🔎 PageRank results:")
for node, value in sorted(pagerank.items(), key=lambda x: x[1], reverse=True):
    print(f"{node}: {value:.4f}")

# --- Step 3: Train TransE on CoDExMedium and Visualise Embeddings ---

# Run this only if pykeen and torch are installed.
try:
    from pykeen.datasets import CoDExMedium
    from pykeen.pipeline import pipeline
    from sklearn.decomposition import PCA
    import numpy as np

    result = pipeline(
        dataset=CoDExMedium,
        model='TransE',
        training_kwargs=dict(num_epochs=100),
    )

    entity_embeddings = result.model.entity_representations[0]().detach().numpy()
    entities = list(result.training.entity_labeling.label_to_id.keys())

    # Reduce dimensions for plotting
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(entity_embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
    plt.title("TransE Embeddings (2D PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

except ImportError:
    print("\n⚠️ PyKEEN or torch not installed. Please install them with `pip install pykeen torch` to run TransE embedding.")
