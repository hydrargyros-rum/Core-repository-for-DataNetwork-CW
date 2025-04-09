import networkx as nx
import pickle
import random

# Load networks from pickle files
with open("admin_graph.pkl", "rb") as f:
    admin_graph = pickle.load(f)
with open("bot_graph.pkl", "rb") as f:
    bot_graph = pickle.load(f)
with open("deletion_graph.pkl", "rb") as f:
    deletion_graph = pickle.load(f)

# Organize graphs into a dictionary
networks = {
    "ADMINISTRATORS": admin_graph,
    "BOT_REQUESTS": bot_graph,
    "REQUEST_FOR_DELETION": deletion_graph
}

# Function to simulate trolling spread using SI model
def simulate_trolling_spread(G, initial_trolls, steps=5):
    """
    Simulates the spread of controversial discussions (trolling) in the network using an SI model.
    Infected nodes try to infect their neighbors at each step with a fixed probability.
    """
    infected = set(initial_trolls)
    for _ in range(steps):
        new_infected = set()
        for troll in infected:
            neighbors = list(G.neighbors(troll))
            for neighbor in neighbors:
                if neighbor not in infected and random.random() < 0.5:  # 50% infection probability
                    new_infected.add(neighbor)
        infected.update(new_infected)
    return infected

# Function to prioritize checking editors
def prioritize_editors(G, identified_trolls):
    """
    Returns a priority list of editors to check based on their proximity to already infected editors.
    Editors with more connections to known trolls are ranked higher.
    """
    priority_list = {}
    for troll in identified_trolls:
        for neighbor in G.neighbors(troll):
            if neighbor not in identified_trolls:
                priority_list[neighbor] = priority_list.get(neighbor, 0) + 1
    return sorted(priority_list.items(), key=lambda x: x[1], reverse=True)

# Set seed for reproducibility
random.seed(42)

# Run simulation for each network
for name, G in networks.items():
    if len(G.nodes) == 0:
        print(f"{name}: Graph is empty, skipping.")
        continue

    initial_trolls = random.sample(list(G.nodes), min(2, len(G.nodes)))
    spread = simulate_trolling_spread(G, initial_trolls)
    priority = prioritize_editors(G, spread)

    print(f"\n--- {name} ---")
    print(f"Initial trolls: {initial_trolls}")
    print(f"Spread after 5 steps: {len(spread)} users infected.")
    print("Top 10 editors to check (priority list):")
    for editor, score in priority[:10]:
        print(f"  {editor}: score {score}")

