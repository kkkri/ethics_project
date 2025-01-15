import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Initialize the graph with students, set up collusion, and assign trust scores."""
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, attendance=True, false_votes=0, false_confirmations=0, detected_as_false_present=False, trust_score=1.0)
    
    # Set up colluding students
    colluding_students = random.sample(list(G.nodes), int(n * p_collusion))
    for student in colluding_students:
        G.nodes[student]['collusion'] = True
        G.nodes[student]['attendance'] = False if np.random.rand() < p_false_attendance else True
    return G

def adjust_k_based_on_trust(G, base_k, student):
    """Adjust the number of confirmations needed for a student based on their trust score."""
    trust_score = G.nodes[student]['trust_score']
    adjusted_k = max(base_k, int(base_k * (1.5 - trust_score)))  # Higher k if trust score is lower
    return adjusted_k

def initialize_dynamic_colluding_groups(G, num_groups=5):
    """Initialize dynamic colluding groups with leaders."""
    colluders = [node for node in G.nodes if G.nodes[node].get('collusion', False)]
    group_size = max(1, len(colluders) // num_groups)
    
    # Create initial groups and assign a leader to each group
    colluding_groups = [colluders[i:i + group_size] for i in range(0, len(colluders), group_size)]
    group_leaders = {group[0]: group for group in colluding_groups if group}  # First student as leader
    
    # Add leader information
    for leader in group_leaders:
        G.nodes[leader]['leader'] = True
    
    return colluding_groups, group_leaders

def evolve_colluding_groups(G, colluding_groups, group_leaders):
    """Evolve colluding groups by merging or splitting."""
    new_groups = []
    group_change_probability = 0.3  # Chance of a group merging or splitting

    for group in colluding_groups:
        if len(group) < 2 or np.random.rand() > group_change_probability:
            new_groups.append(group)
            continue

        if np.random.rand() < 0.5:
            # Split group into two
            split_point = len(group) // 2
            new_groups.append(group[:split_point])
            new_groups.append(group[split_point:])
        else:
            # Merge with another random group
            other_group = random.choice(colluding_groups)
            merged_group = list(set(group + other_group))
            new_groups.append(merged_group)

    # Update leaders for new groups
    group_leaders.clear()
    for group in new_groups:
        leader = random.choice(group)
        group_leaders[leader] = group
        G.nodes[leader]['leader'] = True
    
    return new_groups

def rotate_group_attendance(G, colluding_groups, group_leaders, round_num):
    """Rotate attendance for colluding groups based on leader influence."""
    for group in colluding_groups:
        leader = next((s for s in group if G.nodes[s].get('leader', False)), None)
        group_size = len(group)
        
        if leader:
            # Leader influences attendance: if leader is absent, more members are likely to be absent
            absent_ratio = 0.5 if not G.nodes[leader]['attendance'] else 0.3
        else:
            absent_ratio = 0.3  # Default absence rate if no leader
        
        num_absent = max(1, int(absent_ratio * group_size))
        absent_indices = random.sample(range(group_size), num_absent)
        
        for i, student in enumerate(group):
            G.nodes[student]['attendance'] = i not in absent_indices

def generate_group_confirmations(G, colluding_groups, base_k):
    """Generate confirmations within groups, with required k based on trust scores."""
    confirmations = {student: [] for student in G.nodes}
    for group in colluding_groups:
        present_members = [s for s in group if G.nodes[s]['attendance']]
        absent_members = [s for s in group if not G.nodes[s]['attendance']]
        
        for absent in absent_members:
            # Adjust required k based on the trust score of the absent student
            adjusted_k = adjust_k_based_on_trust(G, base_k, absent)
            confirming_peers = random.sample(present_members, min(adjusted_k, len(present_members)))
            confirmations[absent] = confirming_peers
            
            for peer in confirming_peers:
                G.nodes[peer]['false_votes'] += 1
                G.nodes[absent]['false_confirmations'] += 1

    # For non-colluding students, assign random confirmations with adjusted k based on trust
    for student in G.nodes:
        if not G.nodes[student].get('collusion', False):
            adjusted_k = adjust_k_based_on_trust(G, base_k, student)
            confirmations[student] = random.sample(list(G.nodes), min(adjusted_k, len(G.nodes)))
    return confirmations

def update_trust_scores(G, penalties):
    """Update trust scores based on penalties."""
    for student, penalty_count in penalties.items():
        # Reduce trust score for students with penalties, minimum trust score is 0.1
        G.nodes[student]['trust_score'] = max(0.1, G.nodes[student]['trust_score'] - penalty_count * 0.1)

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

# Simulation loop with dynamic colluding groups and trust score adaptation
for n in n_values:
    for k in k_values:
        for m in m_values:
            total_false_positives = 0
            total_penalties = 0
            total_false_voters = 0
            total_false_presents_detected = 0
            total_false_voters_caught = 0

            for round_num in range(simulation_rounds):
                G = initialize_graph(n)
                colluding_groups, group_leaders = initialize_dynamic_colluding_groups(G)
                colluding_groups = evolve_colluding_groups(G, colluding_groups, group_leaders)
                
                rotate_group_attendance(G, colluding_groups, group_leaders, round_num)
                confirmations = generate_group_confirmations(G, colluding_groups, k)
                
                # Perform roll call and calculate metrics
                false_positive, penalties, false_voters_caught, false_presents_detected = perform_roll_call(G, confirmations, m)
                
                # Update trust scores based on penalties
                update_trust_scores(G, penalties)
                
                total_false_positives += false_positive
                total_penalties += sum(penalties.values())
                total_false_voters += sum([1 for peer in G.nodes if G.nodes[peer]['false_votes'] > 0])
                total_false_presents_detected += false_presents_detected
                total_false_voters_caught += false_voters_caught
            
            # Calculate averages for this configuration
            avg_false_positives = total_false_positives / simulation_rounds
            avg_penalties = total_penalties / simulation_rounds
            avg_false_voters = total_false_voters / simulation_rounds
            avg_false_presents_detected = total_false_presents_detected / simulation_rounds
            avg_false_voters_caught = total_false_voters_caught / simulation_rounds
            
            # Store results
            results = results._append({
                'n': n, 'k': k, 'm': m, 
                'false_positives_avg': avg_false_positives,
                'penalties_avg': avg_penalties,
                'false_voters_avg': avg_false_voters,
                'false_presents_detected_avg': avg_false_presents_detected,
                'false_voters_caught_avg': avg_false_voters_caught
            }, ignore_index=True)

# Visualization remains the same
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
