import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms.community import greedy_modularity_communities


# kya ho rha h.

# Parameters
n_values = range(50, 151, 50)  # Class sizes to test
p_false_attendance = 0.1  # Probability of colluding students marked as present
p_collusion = 0.2  # Fraction of students colluding

# Define k and m values
k_values = range(2, 15)  # Confirmation count
m_values = range(5, 30, 5)  # Roll-call sample size
simulation_rounds = 20  # Number of rounds for averaging

# Initialize results DataFrame
results = pd.DataFrame(columns=['n', 'k', 'm', 'false_positives_avg', 'penalties_avg'])

def initialize_graph(n):
    """Initialize the graph with students and set up collusion."""
    G = nx.erdos_renyi_graph(n, 0.05)  # Random graph structure
    
    # Assign attendance and collusion properties
    for i in G.nodes:
        G.nodes[i]['attendance'] = True
        G.nodes[i]['collusion'] = False
    colluding_students = random.sample(list(G.nodes), int(n * p_collusion))
    for student in colluding_students:
        G.nodes[student]['collusion'] = True
        G.nodes[student]['attendance'] = False if np.random.rand() < p_false_attendance else True
    return G

# Part 1: Network Structure Analysis
def analyze_network_structure(G):
    """Analyze network metrics: modularity, average path length, clustering coefficient."""
    
    # Community detection for modularity
    communities = list(greedy_modularity_communities(G))
    modularity = nx.algorithms.community.modularity(G, communities)
    
    # Average path length (only on connected components)
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        avg_path_length = np.mean([nx.average_shortest_path_length(G.subgraph(c)) for c in nx.connected_components(G)])

    # Clustering coefficient
    clustering_coefficient = nx.average_clustering(G)
    
    print(f"Modularity: {modularity}")
    print(f"Average Path Length: {avg_path_length}")
    print(f"Clustering Coefficient: {clustering_coefficient}")

    return modularity, avg_path_length, clustering_coefficient

# Part 2: Community Stability Analysis
def analyze_community_stability(G, prev_communities=None):
    """Track community stability by comparing community structures over rounds."""
    
    # Detect communities with modularity-based approach
    communities = list(greedy_modularity_communities(G))
    community_mapping = {node: idx for idx, community in enumerate(communities) for node in community}
    
    # Stability tracking: if prev_communities provided, compare with current communities
    stability_score = None
    if prev_communities is not None:
        common_nodes = set(community_mapping.keys()).intersection(set(prev_communities.keys()))
        matching_nodes = sum(1 for node in common_nodes if prev_communities[node] == community_mapping[node])
        stability_score = matching_nodes / len(common_nodes) if common_nodes else 1.0
        print(f"Community Stability Score: {stability_score}")
    
    return communities, community_mapping, stability_score

# Part 3: Visualization
def visualize_network(G, community_mapping):
    """Visualize the network structure and highlight communities."""
    
    pos = nx.spring_layout(G)  # Spring layout for visualization
    plt.figure(figsize=(10, 10))
    
    # Color nodes by community
    communities = set(community_mapping.values())
    colors = sns.color_palette("hsv", len(communities))
    for idx, color in zip(communities, colors):
        nodes_in_community = [node for node in G.nodes if community_mapping[node] == idx]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_community, node_color=[color], label=f"Community {idx}")
    
    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.legend(scatterpoints=1)
    plt.title("Network Structure and Communities")
    plt.show()

# Simulation loop with network analysis and visualization
for n in n_values:
    for k in k_values:
        for m in m_values:
            total_false_positives = 0
            total_penalties = 0
            total_false_voters = 0
            total_false_presents_detected = 0
            total_false_voters_caught = 0

            # Track previous communities for stability analysis
            prev_community_mapping = None

            for round_num in range(simulation_rounds):
                G = initialize_graph(n)
                
                # Perform network analysis
                modularity, avg_path_length, clustering_coefficient = analyze_network_structure(G)
                
                # Analyze community stability
                communities, community_mapping, stability_score = analyze_community_stability(G, prev_community_mapping)
                prev_community_mapping = community_mapping  # Update previous communities
                
                # Visualize the network with communities
                if round_num == 0:  # Only visualize for the first round as a sample
                    visualize_network(G, community_mapping)

                # Continue with roll-call, confirmations, etc., as in the previous simulation code
                # (omitted here for brevity)
