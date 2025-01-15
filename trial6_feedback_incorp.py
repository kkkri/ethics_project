import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
n_values = range(50, 151, 50)  # Testing for 50, 100, 150 students
p_false_attendance = 0.1  # Probability of colluding students being marked as present despite being absent

# Set up a larger range of values for k and m to test
k_values = range(2, 15)  # Varying k from 2 to 14
m_values = range(5, 30, 5)  # Varying m from 5 to 25 in steps of 5
simulation_rounds = 20  # Increased number of rounds for averaging

def initialize_graph(n, p_collusion):
    """Initialize the graph with students and set up collusion."""
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, attendance=True, false_votes=0, false_confirmations=0, detected_as_false_present=False)
    
    # Set up colluding students
    colluding_students = random.sample(list(G.nodes), int(n * p_collusion))
    for student in colluding_students:
        G.nodes[student]['collusion'] = True
        G.nodes[student]['attendance'] = False if np.random.rand() < p_false_attendance else True
    
    return G

def generate_confirmations(G, k):
    confirmations = {student: [] for student in G.nodes}
    for student in G.nodes:
        peers = random.sample(list(G.nodes), k)
        confirmations[student] = peers
        
        for peer in peers:
            if not G.nodes[student]['attendance'] and G.nodes[peer].get('collusion', False):
                G.nodes[peer]['false_votes'] += 1
    
    return confirmations

def perform_roll_call(G, confirmations, m):
    roll_call = random.sample(list(G.nodes), m)
    false_positive = 0
    penalties = {}
    false_voters_count = 0
    false_presents_detected = 0

    for student in roll_call:
        actual_attendance = G.nodes[student]['attendance']
        
        if not actual_attendance:
            G.nodes[student]['detected_as_false_present'] = True
            false_presents_detected += 1
            for peer in confirmations[student]:
                if G.nodes[peer]['attendance']:
                    false_positive += 1
                    penalties[peer] = penalties.get(peer, 0) + 1
                    if G.nodes[peer]['false_votes'] > 0:
                        false_voters_count += 1
    
    return false_positive, penalties, false_voters_count, false_presents_detected

def initialize_colluding_groups(G, num_groups=5):
    colluders = [node for node in G.nodes if G.nodes[node].get('collusion', False)]
    group_size = max(1, len(colluders) // num_groups)
    colluding_groups = [colluders[i:i + group_size] for i in range(0, len(colluders), group_size)]
    
    return colluding_groups

def rotate_group_attendance(G, colluding_groups, round_num):
    for group in colluding_groups:
        group_size = len(group)
        
        num_absent = max(1, group_size // 3)
        absent_indices = [(round_num + i) % group_size for i in range(num_absent)]
        
        for i, student in enumerate(group):
            G.nodes[student]['attendance'] = i not in absent_indices

def generate_group_confirmations(G, colluding_groups, k):
    confirmations = {student: [] for student in G.nodes}
    for group in colluding_groups:
        present_members = [s for s in group if G.nodes[s]['attendance']]
        absent_members = [s for s in group if not G.nodes[s]['attendance']]
        
        for absent in absent_members:
            confirming_peers = random.sample(present_members, min(k, len(present_members)))
            confirmations[absent] = confirming_peers
            for peer in confirming_peers:
                G.nodes[peer]['false_votes'] += 1
                G.nodes[absent]['false_confirmations'] += 1
    
    for student in G.nodes:
        if not G.nodes[student].get('collusion', False):
            confirmations[student] = random.sample(list(G.nodes), k)
    
    return confirmations

def find_optimal_parameters(n, k_range, m_range, collusion_range, simulation_rounds=20):
    optimal_results = pd.DataFrame(columns=[
        'k', 'm', 'p_collusion', 'false_positives_avg', 'penalties_avg', 
        'false_voters_avg', 'false_presents_detected_avg', 'false_voters_caught_avg'
    ])
    
    for p_collusion in collusion_range:
        for k in k_range:
            for m in m_range:
                total_false_positives = 0
                total_penalties = 0
                total_false_voters = 0
                total_false_presents_detected = 0
                total_false_voters_caught = 0

                for round_num in range(simulation_rounds):
                    G = initialize_graph(n, p_collusion)
                    colluding_groups = initialize_colluding_groups(G)
                    rotate_group_attendance(G, colluding_groups, round_num)

                    confirmations = generate_group_confirmations(G, colluding_groups, k)
                    false_positive, penalties, false_voters_caught, false_presents_detected = perform_roll_call(G, confirmations, m)

                    total_false_positives += false_positive
                    total_penalties += sum(penalties.values())
                    total_false_voters += sum([1 for peer in G.nodes if G.nodes[peer]['false_votes'] > 0])
                    total_false_presents_detected += false_presents_detected
                    total_false_voters_caught += false_voters_caught
                
                avg_false_positives = total_false_positives / simulation_rounds
                avg_penalties = total_penalties / simulation_rounds
                avg_false_voters = total_false_voters / simulation_rounds
                avg_false_presents_detected = total_false_presents_detected / simulation_rounds
                avg_false_voters_caught = total_false_voters_caught / simulation_rounds

                optimal_results = pd.concat([optimal_results, pd.DataFrame({
                    'k': [k],
                    'm': [m],
                    'p_collusion': [p_collusion],
                    'false_positives_avg': [avg_false_positives],
                    'penalties_avg': [avg_penalties],
                    'false_voters_avg': [avg_false_voters],
                    'false_presents_detected_avg': [avg_false_presents_detected],
                    'false_voters_caught_avg': [avg_false_voters_caught]
                })], ignore_index=True)
    
    return optimal_results

k_range = range(2, 15)
m_range = range(5, 30, 5)
collusion_range = np.linspace(0.1, 0.5, 5)
n = 100
optimal_results = find_optimal_parameters(n, k_range, m_range, collusion_range)

# Sort and display top configurations
optimal_results_sorted = optimal_results.sort_values(by=['false_positives_avg', 'false_voters_caught_avg'], ascending=[True, False])
top_configs = optimal_results_sorted.head(5)
print("Top 5 Configurations:")
print(top_configs)

# Scatter plot for trade-offs
plt.figure(figsize=(10, 6))
plt.scatter(optimal_results['false_positives_avg'], optimal_results['false_voters_caught_avg'], alpha=0.5, label='All configurations')
plt.scatter(top_configs['false_positives_avg'], top_configs['false_voters_caught_avg'], color='red', label='Top Configurations')
plt.xlabel('Average False Positives')
plt.ylabel('Average False Voters Caught')
plt.title('Trade-off between False Positives and False Voters Caught')
plt.legend()
plt.show()

# Heatmap for penalties in top configurations
top_configs_pivot = top_configs.pivot(index="k", columns="m", values="penalties_avg")
plt.figure(figsize=(8, 6))
sns.heatmap(top_configs_pivot, annot=True, cmap="YlOrRd", fmt=".2f", cbar_kws={'label': 'Penalties Avg'})
plt.title("Penalties Avg for Top Configurations")
plt.xlabel("m (Roll-call sample size)")
plt.ylabel("k (Confirmation count)")
plt.show()

# Visualization for optimal results based on collusion values
for p_collusion in collusion_range:
    subset = optimal_results[optimal_results['p_collusion'] == p_collusion]
    false_positives_pivot = subset.pivot(index="k", columns="m", values="false_positives_avg")
    penalties_pivot = subset.pivot(index="k", columns="m", values="penalties_avg")
    false_voters_pivot = subset.pivot(index="k", columns="m", values="false_voters_avg")
    false_presents_detected_pivot = subset.pivot(index="k", columns="m", values="false_presents_detected_avg")
    false_voters_caught_pivot = subset.pivot(index="k", columns="m", values="false_voters_caught_avg")
    
    # Plotting heatmaps for each metric
    metrics = {
        "Average False Positives": false_positives_pivot,
        "Average Penalties": penalties_pivot,
        "Average False Voters Detected": false_voters_pivot,
        "Average False Presents Detected": false_presents_detected_pivot,
        "Average False Voters Caught": false_voters_caught_pivot,
    }
    for title, data in metrics.items():
        plt.figure(figsize=(10, 6))
        sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': title})
        plt.title(f"{title} for Varying k and m (p_collusion={p_collusion})")
        plt.xlabel("m (Roll-call sample size)")
        plt.ylabel("k (Confirmation count)")
        plt.show()
