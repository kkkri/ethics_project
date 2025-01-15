import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters for the simulation
n = 100  # Class size
p_false_attendance = 0.1  # Probability of colluding students marked as present
p_collusion = 0.2  # Fraction of students colluding
k_values = range(2, 15)  # Grid search values for k
m_values = range(5, 30, 5)  # Grid search values for m
simulation_rounds = 20  # Number of simulation rounds

# Initialize a DataFrame to store the results of the grid search
results = pd.DataFrame(columns=['k', 'm', 'false_positives_avg', 'penalties_avg', 'detection_accuracy'])

def initialize_graph(n):
    """Initialize the graph with students, set up collusion and attendance properties."""
    G = nx.erdos_renyi_graph(n, 0.05)  # Random graph for simulation purposes
    for i in G.nodes:
        G.nodes[i]['attendance'] = True
        G.nodes[i]['collusion'] = False
    colluding_students = random.sample(list(G.nodes), int(n * p_collusion))
    for student in colluding_students:
        G.nodes[student]['collusion'] = True
        G.nodes[student]['attendance'] = False if np.random.rand() < p_false_attendance else True
    return G

def perform_simulation(G, k, m):
    """Run simulation for specific values of k and m, tracking key metrics."""
    total_false_positives = 0
    total_penalties = 0
    detection_accuracy = 0
    colluding_students = [n for n in G.nodes if G.nodes[n]['collusion']]
    
    for _ in range(simulation_rounds):
        # Confirmation process: simple random selection of k peers
        confirmations = {student: random.sample(list(G.nodes), min(k, len(G.nodes) - 1)) for student in G.nodes}
        
        # Roll-call simulation
        roll_call = random.sample(list(G.nodes), m)
        false_positive = 0
        penalties = 0
        detected_colluders = set()
        
        for student in roll_call:
            actual_attendance = G.nodes[student]['attendance']
            if not actual_attendance:
                detected_colluders.add(student)
                for peer in confirmations[student]:
                    if G.nodes[peer]['attendance']:
                        false_positive += 1
                        penalties += 1

        total_false_positives += false_positive
        total_penalties += penalties
        detection_accuracy += len(detected_colluders.intersection(colluding_students)) / len(colluding_students)
    
    # Compute averages for each metric
    false_positives_avg = total_false_positives / simulation_rounds
    penalties_avg = total_penalties / simulation_rounds
    detection_accuracy_avg = detection_accuracy / simulation_rounds

    return false_positives_avg, penalties_avg, detection_accuracy_avg

# Run grid search for each combination of k and m
for k in k_values:
    for m in m_values:
        G = initialize_graph(n)
        false_positives_avg, penalties_avg, detection_accuracy_avg = perform_simulation(G, k, m)
        results = results._append({
            'k': k,
            'm': m,
            'false_positives_avg': false_positives_avg,
            'penalties_avg': penalties_avg,
            'detection_accuracy': detection_accuracy_avg
        }, ignore_index=True)

# Visualization of results
def plot_heatmaps(results, metric, title, cmap):
    """Plot heatmap for a given metric from the grid search results."""
    pivot_table = results.pivot(index='k', columns='m', values=metric)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, cbar_kws={'label': metric})
    plt.title(title)
    plt.xlabel("Roll-call sample size (m)")
    plt.ylabel("Confirmation count (k)")
    plt.show()

# Plot heatmaps for the main metrics
plot_heatmaps(results, 'false_positives_avg', "Average False Positives for k and m", "YlGnBu")
plot_heatmaps(results, 'penalties_avg', "Average Penalties for k and m", "YlOrRd")
plot_heatmaps(results, 'detection_accuracy', "Detection Accuracy for k and m", "Blues")
