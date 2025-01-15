import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
n_values = range(50, 150, 50)  # Testing for 50, 100, 150 students
p_false_attendance = 0.1  # Probability of colluding students being marked as present despite being absent
p_collusion = 0.2  # Fraction of students colluding

# Set up a larger range of values for k and m to test
k_values = range(10, 25)  # Varying k from 2 to 14
m_values = range(25, 50, 5)  # Varying m from 5 to 25 in steps of 5
simulation_rounds = 20  # Increased number of rounds for averaging

# Initialize a DataFrame to store results
results = pd.DataFrame(columns=['n', 'k', 'm', 'false_positives_avg', 'penalties_avg'])


def initialize_graph(n):
    """Initialize the graph with students and set up collusion."""
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, attendance=True, false_votes=0, false_confirmations=0, detected_as_false_present=False)
    
    # Set up colluding students
    colluding_students = random.sample(list(G.nodes), int(n * p_collusion))
    for student in colluding_students:
        G.nodes[student]['collusion'] = True
        # False attendance with some probability for colluding students
        G.nodes[student]['attendance'] = False if np.random.rand() < p_false_attendance else True
    
    return G




def generate_confirmations(G, k):
    """Generate confirmation networks for each student with false vote tracking."""
    confirmations = {student: [] for student in G.nodes}
    for student in G.nodes:
        peers = random.sample(list(G.nodes), k)
        confirmations[student] = peers
        
        # Track false votes for colluding students
        for peer in peers:
            if not G.nodes[student]['attendance'] and G.nodes[peer].get('collusion', False):
                G.nodes[peer]['false_votes'] += 1
    
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




def initialize_colluding_groups(G, num_groups=5):
    """Divide colluding students into fixed groups for rotation-based false voting."""
    colluders = [node for node in G.nodes if G.nodes[node].get('collusion', False)]
    group_size = max(1, len(colluders) // num_groups)
    colluding_groups = [colluders[i:i + group_size] for i in range(0, len(colluders), group_size)]
    
    # Each round, rotate attendance within each group
    return colluding_groups


def rotate_group_attendance(G, colluding_groups, round_num):
    """Update attendance for colluding groups by rotating which group members are absent."""
    for group in colluding_groups:
        group_size = len(group)
        
        # Calculate how many members will be absent in this round (e.g., 1/3 of the group)
        num_absent = max(1, group_size // 3)
        absent_indices = [(round_num + i) % group_size for i in range(num_absent)]
        
        for i, student in enumerate(group):
            # Mark some members as absent and the others as present
            G.nodes[student]['attendance'] = i not in absent_indices


def generate_group_confirmations(G, colluding_groups, k):
    """Generate confirmations where present group members vouch for absent members."""
    confirmations = {student: [] for student in G.nodes}
    for group in colluding_groups:
        present_members = [s for s in group if G.nodes[s]['attendance']]
        absent_members = [s for s in group if not G.nodes[s]['attendance']]
        
        for absent in absent_members:
            # Only present members within the group confirm the attendance of absent members
            confirming_peers = random.sample(present_members, min(k, len(present_members)))
            confirmations[absent] = confirming_peers
            
            # Track false confirmations for absent members
            for peer in confirming_peers:
                G.nodes[peer]['false_votes'] += 1
                G.nodes[absent]['false_confirmations'] += 1
    
    # For non-colluding students, assign random confirmations
    for student in G.nodes:
        if not G.nodes[student].get('collusion', False):
            confirmations[student] = random.sample(list(G.nodes), k)
    
    return confirmations





def initialize_colluding_groups_with_rotation(G, num_groups=5):
    """Divide colluding students into fixed groups with rotation-based false attendance."""
    colluders = [node for node in G.nodes if G.nodes[node].get('collusion', False)]
    group_size = max(1, len(colluders) // num_groups)
    colluding_groups = [colluders[i:i + group_size] for i in range(0, len(colluders), group_size)]
    return colluding_groups

def rotate_group_attendance_with_turns(G, colluding_groups, round_num):
    """Rotate attendance with predetermined group turns for marking false attendance."""
    for group_num, group in enumerate(colluding_groups):
        group_size = len(group)
        
        # Only one group will mark false attendance in each round
        mark_false_attendance = (round_num % len(colluding_groups) == group_num)
        
        for i, student in enumerate(group):
            # If this group is responsible for marking false attendance this round
            G.nodes[student]['attendance'] = not mark_false_attendance
            if mark_false_attendance:
                G.nodes[student]['attendance'] = False  # This student will be marked as absent

def generate_group_confirmations_within_rotation(G, colluding_groups, k):
    """Generate confirmations where group members vouch for each other, and only present members confirm absentees."""
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

    # For non-colluding students, assign random confirmations
    for student in G.nodes:
        if not G.nodes[student].get('collusion', False):
            confirmations[student] = random.sample(list(G.nodes), k)
    
    return confirmations

# Run simulation using this new collusion behavior
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
                colluding_groups = initialize_colluding_groups_with_rotation(G)
                rotate_group_attendance_with_turns(G, colluding_groups, round_num)

                confirmations = generate_group_confirmations_within_rotation(G, colluding_groups, k)
                false_positive, penalties, false_voters_caught, false_presents_detected = perform_roll_call(G, confirmations, m)
                
                total_false_positives += false_positive
                total_penalties += sum(penalties.values())
                total_false_voters += sum([1 for peer in G.nodes if G.nodes[peer]['false_votes'] > 0])
                total_false_presents_detected += false_presents_detected
                total_false_voters_caught += false_voters_caught
            
            # Calculate averages for this n, k, m configuration
            avg_false_positives = total_false_positives / simulation_rounds
            avg_penalties = total_penalties / simulation_rounds
            avg_false_voters = total_false_voters / simulation_rounds
            avg_false_presents_detected = total_false_presents_detected / simulation_rounds
            avg_false_voters_caught = total_false_voters_caught / simulation_rounds

            # Store the result
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

# Visualization code remains the same as previously provided





# Visualization for each value of n
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

