import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms.community import greedy_modularity_communities

# Parameters
n_values = range(50, 151, 50)  # Class sizes to test
p_false_attendance = 0.1  # Probability of colluding students marked as present
p_collusion = 0.2  # Fraction of students colluding

# Define k and m values
k_values = range(2, 15)  # Confirmation count
m_values = range(5, 30, 5)  # Roll-call sample size
simulation_rounds = 20  # Number of rounds for averaging

# Store results
results = pd.DataFrame(columns=['n', 'k', 'm', 'false_positives_avg', 'penalties_avg'])

def initialize_graph_with_communities(n):
    """Initialize the graph with community structure and centrality measures."""
    G = nx.erdos_renyi_graph(n, 0.05)  # Random graph
    
    # Detect communities
    communities = list(greedy_modularity_communities(G))
    for i, community in enumerate(communities):
        for node in community:
            G.nodes[node]['community'] = i  # Assign community ID
    
    # Calculate centrality
    centrality = nx.degree_centrality(G)
    for node, cent_value in centrality.items():
        G.nodes[node]['centrality'] = cent_value  # Add centrality to node attributes
    
    # Set attendance, collusion, and false_votes flags
    colluding_students = random.sample(list(G.nodes), int(n * p_collusion))
    for student in G.nodes:
        G.nodes[student]['collusion'] = student in colluding_students
        G.nodes[student]['attendance'] = False if G.nodes[student]['collusion'] and np.random.rand() < p_false_attendance else True
        G.nodes[student]['false_votes'] = 0  # Initialize false votes for tracking
    
    return G



def generate_community_based_confirmations(G, k):
    """Generate confirmation lists based on community structure and centrality."""
    confirmations = {student: [] for student in G.nodes}
    for student in G.nodes:
        community = G.nodes[student]['community']
        # Select potential confirmers within the same community, excluding the student
        potential_confirmers = [n for n in G.nodes if G.nodes[n]['community'] == community and n != student]
        
        # If there are not enough confirmers in the community, expand to the whole network
        if len(potential_confirmers) < k:
            potential_confirmers = [n for n in G.nodes if n != student]
            centralities = [G.nodes[n]['centrality'] for n in potential_confirmers]
        else:
            # Use community-based confirmers if sufficient in number
            centralities = [G.nodes[n]['centrality'] for n in potential_confirmers]
        
        # Select confirmers based on centrality, with fallback if confirmers are still insufficient
        if len(potential_confirmers) > 0:
            confirmers = random.choices(potential_confirmers, weights=centralities, k=min(k, len(potential_confirmers)))
        else:
            confirmers = []
        
        confirmations[student] = confirmers
    return confirmations


def perform_roll_call(G, confirmations, m):
    """Perform roll-call and calculate false positives, penalties, and false voters."""
    roll_call = random.sample(list(G.nodes), m)
    false_positive = 0
    penalties = {}
    false_voters_count = 0  # Count of false voters caught in this roll call
    false_presents_detected = 0  # Initialize the counter for false presents detected

    for student in roll_call:
        actual_attendance = G.nodes[student]['attendance']
        
        # Detect if this student was falsely marked as present
        if not actual_attendance:
            G.nodes[student]['detected_as_false_present'] = True
            false_presents_detected += 1  # Increment the count of false presents detected
            for peer in confirmations[student]:
                if G.nodes[peer]['attendance']:
                    false_positive += 1
                    penalties[peer] = penalties.get(peer, 0) + 1
                    if G.nodes[peer]['false_votes'] > 0:
                        false_voters_count += 1  # Increment only if this voter is "caught" in roll call
    
    return false_positive, penalties, false_voters_count, false_presents_detected

# Rest of your simulation process

for n in n_values:
    for k in k_values:
        for m in m_values:
            total_false_positives = 0
            total_penalties = 0
            total_false_voters = 0
            total_false_presents_detected = 0
            total_false_voters_caught = 0

            for round_num in range(simulation_rounds):
                G = initialize_graph_with_communities(n)
                confirmations = generate_community_based_confirmations(G, k)
                
                # Perform roll call and calculate metrics as before
                false_positive, penalties, false_voters_caught, false_presents_detected = perform_roll_call(G, confirmations, m)
                
                total_false_positives += false_positive
                total_penalties += sum(penalties.values())
                total_false_voters += sum([1 for peer in G.nodes if G.nodes[peer]['false_votes'] > 0])
                total_false_presents_detected += false_presents_detected
                total_false_voters_caught += false_voters_caught
            
            # Calculate averages for each metric
            avg_false_positives = total_false_positives / simulation_rounds
            avg_penalties = total_penalties / simulation_rounds
            avg_false_voters = total_false_voters / simulation_rounds
            avg_false_presents_detected = total_false_presents_detected / simulation_rounds
            avg_false_voters_caught = total_false_voters_caught / simulation_rounds
            
            # Store results
            results = pd.concat([results, pd.DataFrame({
                'n': [n],
                'k': [k],
                'm': [m],
                'false_positives_avg': [avg_false_positives],
                'penalties_avg': [avg_penalties],
                'false_voters_avg': [avg_false_voters],
                'false_presents_detected_avg': [avg_false_presents_detected],
                'false_voters_caught_avg': [avg_false_voters_caught]
            })], ignore_index=True)

# Visualization can be as before, but now the analysis includes community and centrality effects.
for n in n_values:
    subset = results[results['n'] == n]
    
    # Pivot data for each metric with respect to k and m
    false_positives_pivot = subset.pivot(index="k", columns="m", values="false_positives_avg")
    penalties_pivot = subset.pivot(index="k", columns="m", values="penalties_avg")
    false_voters_pivot = subset.pivot(index="k", columns="m", values="false_voters_avg")
    false_presents_detected_pivot = subset.pivot(index="k", columns="m", values="false_presents_detected_avg")
    false_voters_caught_pivot = subset.pivot(index="k", columns="m", values="false_voters_caught_avg")
    
    # Plot heatmap for false positives
    plt.figure(figsize=(10, 6))
    sns.heatmap(false_positives_pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Average False Positives'})
    plt.title(f"Heatmap of Average False Positives for Varying k and m (n={n})")
    plt.xlabel("m (Roll-call sample size)")
    plt.ylabel("k (Confirmation count)")
    plt.show()
    
    # Plot heatmap for penalties
    plt.figure(figsize=(10, 6))
    sns.heatmap(penalties_pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Average Penalties'})
    plt.title(f"Heatmap of Average Penalties for Varying k and m (n={n})")
    plt.xlabel("m (Roll-call sample size)")
    plt.ylabel("k (Confirmation count)")
    plt.show()
    
    # Plot heatmap for false voters detected
    plt.figure(figsize=(10, 6))
    sns.heatmap(false_voters_pivot, annot=True, fmt=".2f", cmap="Blues", cbar_kws={'label': 'Average False Voters Detected'})
    plt.title(f"Heatmap of Average False Voters Detected for Varying k and m (n={n})")
    plt.xlabel("m (Roll-call sample size)")
    plt.ylabel("k (Confirmation count)")
    plt.show()
    
    # Plot heatmap for false presents detected
    plt.figure(figsize=(10, 6))
    sns.heatmap(false_presents_detected_pivot, annot=True, fmt=".2f", cmap="Purples", cbar_kws={'label': 'Average False Presents Detected'})
    plt.title(f"Heatmap of Average False Presents Detected for Varying k and m (n={n})")
    plt.xlabel("m (Roll-call sample size)")
    plt.ylabel("k (Confirmation count)")
    plt.show()
    
    # Plot heatmap for false voters caught
    plt.figure(figsize=(10, 6))
    sns.heatmap(false_voters_caught_pivot, annot=True, fmt=".2f", cmap="Oranges", cbar_kws={'label': 'Average False Voters Caught'})
    plt.title(f"Heatmap of Average False Voters Caught for Varying k and m (n={n})")
    plt.xlabel("m (Roll-call sample size)")
    plt.ylabel("k (Confirmation count)")
    plt.show()
