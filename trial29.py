import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

# Parameters
n = 50  # Total students
k = 2   # Minimum peer confirmations required
m = 5   # Number of random audits
iterations = 50  # Simulation iterations
false_attendance_rate = 0.2  # Probability of false attendance
collusion_rate = 0.3  # Probability that a group is colluding
dynamic_change_prob = 0.1  # Probability of a student changing group
penalty_weight = 0.8  # Weight reduction for penalized students

# Analyze communities for suspicious behavior
def analyze_communities(communities, graph):
    suspicious_communities = []
    for community in communities:
        subgraph = graph.subgraph(community)
        density = nx.density(subgraph)  # Internal edge density
        avg_weight = np.mean([data['weight'] for _, _, data in subgraph.edges(data=True)]) if subgraph.number_of_edges() > 0 else 0
        if density > 0.1 and avg_weight > 1.5:  # Relaxed thresholds for better sensitivity
            suspicious_communities.append(community)
    return suspicious_communities

# Construct a peer confirmation graph
def build_graph(attendance, penalties):
    G = nx.DiGraph()
    for day, logs in attendance.items():
        for student, peers in logs.items():
            for peer in peers:
                weight = penalties.get(student, 1)
                if G.has_edge(student, peer):
                    G[student][peer]['weight'] += weight
                else:
                    G.add_edge(student, peer, weight=weight)
    return G

# Detect communities using Louvain method
def detect_communities(graph):
    if len(graph.edges) == 0:
        return []  # Return an empty list if the graph has no edges
    undirected_graph = graph.to_undirected()
    communities = greedy_modularity_communities(undirected_graph)
    return [list(community) for community in communities]

# Random audit within suspicious communities
def audit_communities(suspicious_communities, colluding_groups, m):
    if len(suspicious_communities) == 0:  # Avoid empty suspicious communities
        return 0, 0

    audited_communities = random.sample(suspicious_communities, min(m, len(suspicious_communities)))
    true_detected = 0
    false_detected = 0
    for community in audited_communities:
        for student in community:
            if any(student in group for group in colluding_groups):
                true_detected += 1
            else:
                false_detected += 1
    return true_detected, false_detected

# Generate dynamic attendance logs with penalties
def generate_dynamic_attendance_with_penalties(n, k, days=10, collusion_rate=0.3, dynamic_change_prob=0.1, penalties=None):
    colluding_groups = [set(random.sample(range(n), k)) for _ in range(int(n * collusion_rate))]
    attendance = {}

    for day in range(days):
        daily_logs = {}
        for group in colluding_groups:
            # Dynamic behavior: Add/remove members from groups
            if random.random() < dynamic_change_prob:
                if len(group) > 1:
                    group.remove(random.choice(list(group)))  # Remove one member
                group.add(random.randint(0, n - 1))  # Add a new random member
        
        for student in range(n):
            if penalties and penalties.get(student, 1) < 0.5:  # Skip heavily penalized students
                continue
            if any(student in group for group in colluding_groups):  # Colluding group behavior
                peers = random.sample(list(colluding_groups[0]), min(k, len(colluding_groups[0])))  # Share confirmations
            else:
                peers = random.sample([p for p in range(n) if p != student], k)  # Legitimate attendance
            daily_logs[student] = peers
        attendance[day] = daily_logs
    
    return attendance, colluding_groups

# Apply penalties to detected colluders
def apply_penalties(suspicious_communities, penalties):
    for community in suspicious_communities:
        for student in community:
            penalties[student] = penalties.get(student, 1) * penalty_weight  # Reduce weight
    return penalties

# Simulation with penalties
def evaluate_with_penalties(n, k_values, m_values, iterations, days=10):
    tdr_matrix = np.zeros((len(k_values), len(m_values)))
    fpr_matrix = np.zeros((len(k_values), len(m_values)))

    penalties = {}  # Track penalties for students

    for i, k in enumerate(k_values):
        for j, m in enumerate(m_values):
            true_detected = 0
            false_detected = 0
            for _ in range(iterations):
                attendance, colluding_groups = generate_dynamic_attendance_with_penalties(n, k, days, penalties=penalties)
                graph = build_graph(attendance, penalties)
                communities = detect_communities(graph)
                suspicious_communities = analyze_communities(communities, graph)
                if not suspicious_communities:
                    suspicious_communities = [random.sample(range(n), min(m, n))]  # Fallback
                
                # Audit and apply penalties
                true, false = audit_communities(suspicious_communities, colluding_groups, m)
                true_detected += true
                false_detected += false
                penalties = apply_penalties(suspicious_communities, penalties)  # Update penalties
            
            total_colluding_students = sum(len(group) for group in colluding_groups) * iterations
            tdr_matrix[i, j] = true_detected / total_colluding_students if total_colluding_students > 0 else 0
            fpr_matrix[i, j] = false_detected / (n * iterations) if n * iterations > 0 else 0

            print(f"k={k}, m={m}: TDR={tdr_matrix[i, j]:.2f}, FPR={fpr_matrix[i, j]:.2f}")
    return tdr_matrix, fpr_matrix

# Run simulation
k_values = [2, 3, 4, 5, 6]
m_values = [5, 10, 15, 20, 25]
tdr_matrix, fpr_matrix = evaluate_with_penalties(n, k_values, m_values, iterations)

# Visualize results with TDR values on heatmap
plt.figure(figsize=(10, 8))
plt.imshow(tdr_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
for i in range(len(k_values)):
    for j in range(len(m_values)):
        plt.text(j, i, f"{tdr_matrix[i, j]:.2f}", ha='center', va='center', color='white')
plt.colorbar(label='True Detection Rate (TDR)')
plt.xticks(ticks=np.arange(len(m_values)), labels=m_values)
plt.yticks(ticks=np.arange(len(k_values)), labels=k_values)
plt.xlabel("Number of Audits (m)")
plt.ylabel("Peer Confirmations (k)")
plt.title("True Detection Rate (TDR) with Penalty Feedback (n=50)")
plt.show()

# Visualize results with FPR values on heatmap
plt.figure(figsize=(10, 8))
plt.imshow(fpr_matrix, cmap='plasma', interpolation='nearest', aspect='auto')
for i in range(len(k_values)):
    for j in range(len(m_values)):
        plt.text(j, i, f"{fpr_matrix[i, j]:.2f}", ha='center', va='center', color='white')
plt.colorbar(label='False Positive Rate (FPR)')
plt.xticks(ticks=np.arange(len(m_values)), labels=m_values)
plt.yticks(ticks=np.arange(len(k_values)), labels=k_values)
plt.xlabel("Number of Audits (m)")
plt.ylabel("Peer Confirmations (k)")
plt.title("False Positive Rate (FPR) with Penalty Feedback (n=50)")
plt.show()

