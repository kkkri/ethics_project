import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
n_values = range(50, 151, 50)  # Testing for 50, 100, 150 students
p_false_attendance = 0.1  # Probability of colluding students being marked as present despite being absent
p_collusion = 0.2  # Fraction of students colluding

# Set up a larger range of values for k and m to test
k_values = range(2, 15)  # Varying k from 2 to 14
m_values = range(5, 30, 5)  # Varying m from 5 to 25 in steps of 5
simulation_rounds = 20  # Increased number of rounds for averaging

# Initialize a DataFrame to store results
results = pd.DataFrame(columns=['n', 'k', 'm', 'false_positives_avg', 'penalties_avg'])

def initialize_graph(n):
    """Initialize the graph with students and set up collusion."""
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, attendance=True)  # Start with everyone marked as present
    colluding_students = random.sample(list(G.nodes), int(n * p_collusion))
    for student in colluding_students:
        G.nodes[student]['collusion'] = True
        G.nodes[student]['attendance'] = False if np.random.rand() < p_false_attendance else True
    return G

def generate_confirmations(G, k):
    """Generate confirmation networks for each student."""
    confirmations = {student: [] for student in G.nodes}
    for student in G.nodes:
        peers = random.sample(list(G.nodes), k)
        confirmations[student] = peers
    return confirmations

def perform_roll_call(G, confirmations, m):
    """Perform roll-call and calculate false positives and penalties."""
    roll_call = random.sample(list(G.nodes), m)
    false_positive = 0
    penalties = {}

    for student in roll_call:
        actual_attendance = G.nodes[student]['attendance']
        for peer in confirmations[student]:
            if not actual_attendance and G.nodes[peer]['attendance']:
                false_positive += 1
                penalties[peer] = penalties.get(peer, 0) + 1
    return false_positive, penalties

# Run the simulations
for n in n_values:
    for k in k_values:
        for m in m_values:
            total_false_positives = 0
            total_penalties = 0

            for _ in range(simulation_rounds):
                G = initialize_graph(n)
                confirmations = generate_confirmations(G, k)
                false_positive, penalties = perform_roll_call(G, confirmations, m)
                total_false_positives += false_positive
                total_penalties += sum(penalties.values())
            
            # Calculate averages for this n, k, m configuration
            avg_false_positives = total_false_positives / simulation_rounds
            avg_penalties = total_penalties / simulation_rounds

            # Store the result
            results = results._append({'n': n, 'k': k, 'm': m, 
                                       'false_positives_avg': avg_false_positives, 
                                       'penalties_avg': avg_penalties}, ignore_index=True)

# Visualization
for n in n_values:
    subset = results[results['n'] == n]
    
    # Pivot data for each n value
    false_positives_pivot = subset.pivot(index="k", columns="m", values="false_positives_avg")
    penalties_pivot = subset.pivot(index="k", columns="m", values="penalties_avg")
    
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
