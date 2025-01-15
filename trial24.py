import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

# Parameters
n = 200  # Total students
k = 3   # Minimum peer confirmations required
m = 5   # Number of random audits
iterations = 50  # Simulation iterations
false_attendance_rate = 0.2  # Probability of false attendance
collusion_rate = 0.2  # Probability that a group is colluding

# Generate attendance logs
def generate_attendance(n, k, days=10, collusion_rate=0.2):
    colluding_groups = [set(random.sample(range(n), k)) for _ in range(int(n * collusion_rate))]
    attendance = {}
    for day in range(days):
        daily_logs = {}
        for student in range(n):
            if any(student in group for group in colluding_groups):  # Colluding group behavior
                peers = random.sample(list(colluding_groups[0]), k)  # Share confirmations within the group
            else:
                peers = random.sample([p for p in range(n) if p != student], k)  # Legitimate attendance
            daily_logs[student] = peers
        attendance[day] = daily_logs
    return attendance, colluding_groups

# Construct a peer confirmation graph
def build_graph(attendance):
    G = nx.DiGraph()
    for day, logs in attendance.items():
        for student, peers in logs.items():
            for peer in peers:
                if G.has_edge(student, peer):
                    G[student][peer]['weight'] += 1
                else:
                    G.add_edge(student, peer, weight=1)
    return G

# Detect communities using Louvain method
def detect_communities(graph):
    # Convert to undirected graph for community detection
    undirected_graph = graph.to_undirected()
    communities = greedy_modularity_communities(undirected_graph)
    return [list(community) for community in communities]

# Analyze communities for suspicious behavior
def analyze_communities(communities, graph):
    suspicious_communities = []
    for community in communities:
        subgraph = graph.subgraph(community)
        density = nx.density(subgraph)  # Internal edge density
        if density > 0.2:  # Adjusted threshold for suspicion
            suspicious_communities.append(community)
    return suspicious_communities

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

# Evaluate detection rate vs false positives with community analysis
def evaluate_with_communities(attendance, colluding_groups, k_values, m_values, iterations=50):
    tdr_matrix = np.zeros((len(k_values), len(m_values)))
    fpr_matrix = np.zeros((len(k_values), len(m_values)))

    total_colluding_students = sum(len(group) for group in colluding_groups) * iterations

    for i, k in enumerate(k_values):
        for j, m in enumerate(m_values):
            true_detected = 0
            false_detected = 0
            for _ in range(iterations):
                graph = build_graph(attendance)
                communities = detect_communities(graph)
                if not communities:
                    print(f"No communities detected for k={k}, m={m}")
                suspicious_communities = analyze_communities(communities, graph)
                true, false = audit_communities(suspicious_communities, colluding_groups, m)
                true_detected += true
                false_detected += false
            tdr_matrix[i, j] = true_detected / total_colluding_students if total_colluding_students > 0 else 0
            fpr_matrix[i, j] = false_detected / (n * iterations) if n * iterations > 0 else 0

            print(f"k={k}, m={m}: TDR={tdr_matrix[i, j]:.2f}, FPR={fpr_matrix[i, j]:.2f}")
    return tdr_matrix, fpr_matrix

# Simulation
k_values = [5, 10, 15, 20, 25, 30]
m_values = [10, 20, 30, 40, 50]
attendance_logs, colluding_groups = generate_attendance(n, k)
tdr_matrix, fpr_matrix = evaluate_with_communities(attendance_logs, colluding_groups, k_values, m_values, iterations)

# Heatmap Visualization for TDR
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
plt.title("True Detection Rate (TDR) with Louvain Community Detection")
plt.show()

# Heatmap Visualization for FPR
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
plt.title("False Positive Rate (FPR) with Louvain Community Detection")
plt.show()
