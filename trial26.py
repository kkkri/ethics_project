import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

# Parameters
n = 200  # Total students
k = 3   # Minimum peer confirmations required
m = 10  # Number of random audits
iterations = 50  # Simulation iterations
false_attendance_rate = 0.2  # Probability of false attendance
collusion_rate = 0.2  # Probability that a group is colluding
dynamic_change_prob = 0.2  # Probability of a student changing group

# Generate dynamic attendance logs
# def generate_dynamic_attendance(n, k, days=10, collusion_rate=0.2, dynamic_change_prob=0.2):
#     colluding_groups = [set(random.sample(range(n), k)) for _ in range(int(n * collusion_rate))]
#     attendance = {}

#     for day in range(days):
#         daily_logs = {}
#         for group in colluding_groups:
#             # Dynamic behavior: Randomly add/remove members from groups
#             if random.random() < dynamic_change_prob:
#                 group_member = random.choice(list(group))
#                 group.remove(group_member) if len(group) > 1 else None
#                 group.add(random.randint(0, n - 1))
        
#         for student in range(n):
#             if any(student in group for group in colluding_groups):  # Colluding group behavior
#                 peers = random.sample(list(colluding_groups[0]), k)  # Share confirmations within the group
#             else:
#                 peers = random.sample([p for p in range(n) if p != student], k)  # Legitimate attendance
#             daily_logs[student] = peers
#         attendance[day] = daily_logs
    
#     return attendance, colluding_groups

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
    undirected_graph = graph.to_undirected()
    communities = greedy_modularity_communities(undirected_graph)
    return [list(community) for community in communities]

# Analyze communities for suspicious behavior
# def analyze_communities(communities, graph):
#     suspicious_communities = []
#     for community in communities:
#         subgraph = graph.subgraph(community)
#         density = nx.density(subgraph)  # Internal edge density
#         avg_weight = np.mean([data['weight'] for _, _, data in subgraph.edges(data=True)]) if subgraph.number_of_edges() > 0 else 0
#         if density > 0.2 and avg_weight > 2:  # Adjusted thresholds for suspicion
#             suspicious_communities.append(community)
#     return suspicious_communities

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

# Evaluate detection rate vs false positives
# def evaluate_dynamic_detection(attendance, colluding_groups, k_values, m_values, iterations=50):
#     tdr_matrix = np.zeros((len(k_values), len(m_values)))
#     fpr_matrix = np.zeros((len(k_values), len(m_values)))

#     total_colluding_students = sum(len(group) for group in colluding_groups) * iterations

#     for i, k in enumerate(k_values):
#         for j, m in enumerate(m_values):
#             true_detected = 0
#             false_detected = 0
#             for _ in range(iterations):
#                 graph = build_graph(attendance)
#                 communities = detect_communities(graph)
#                 suspicious_communities = analyze_communities(communities, graph)
#                 true, false = audit_communities(suspicious_communities, colluding_groups, m)
#                 true_detected += true
#                 false_detected += false
#             tdr_matrix[i, j] = true_detected / total_colluding_students if total_colluding_students > 0 else 0
#             fpr_matrix[i, j] = false_detected / (n * iterations) if n * iterations > 0 else 0

#             print(f"k={k}, m={m}: TDR={tdr_matrix[i, j]:.2f}, FPR={fpr_matrix[i, j]:.2f}")
#     return tdr_matrix, fpr_matrix




# Simulation
# k_values = [5, 10, 15, 20, 25, 30]
# m_values = [10, 20, 30, 40, 50]
# attendance_logs, colluding_groups = generate_dynamic_attendance(n, k)
# tdr_matrix, fpr_matrix = evaluate_dynamic_detection(attendance_logs, colluding_groups, k_values, m_values, iterations)

# # Heatmap Visualization for TDR
# plt.figure(figsize=(10, 8))
# plt.imshow(tdr_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
# for i in range(len(k_values)):
#     for j in range(len(m_values)):
#         plt.text(j, i, f"{tdr_matrix[i, j]:.2f}", ha='center', va='center', color='white')
# plt.colorbar(label='True Detection Rate (TDR)')
# plt.xticks(ticks=np.arange(len(m_values)), labels=m_values)
# plt.yticks(ticks=np.arange(len(k_values)), labels=k_values)
# plt.xlabel("Number of Audits (m)")
# plt.ylabel("Peer Confirmations (k)")
# plt.title("True Detection Rate (TDR) with Dynamic Behavior")
# plt.show()

# # Heatmap Visualization for FPR
# plt.figure(figsize=(10, 8))
# plt.imshow(fpr_matrix, cmap='plasma', interpolation='nearest', aspect='auto')
# for i in range(len(k_values)):
#     for j in range(len(m_values)):
#         plt.text(j, i, f"{fpr_matrix[i, j]:.2f}", ha='center', va='center', color='white')
# plt.colorbar(label='False Positive Rate (FPR)')
# plt.xticks(ticks=np.arange(len(m_values)), labels=m_values)
# plt.yticks(ticks=np.arange(len(k_values)), labels=k_values)
# plt.xlabel("Number of Audits (m)")
# plt.ylabel("Peer Confirmations (k)")
# plt.title("False Positive Rate (FPR) with Dynamic Behavior")
# plt.show()




def generate_dynamic_attendance(n, k, days=10, collusion_rate=0.2, dynamic_change_prob=0.1):
    colluding_groups = [set(random.sample(range(n), k)) for _ in range(int(n * collusion_rate))]
    attendance = {}

    for day in range(days):
        daily_logs = {}
        for group in colluding_groups:
            # Limit dynamic behavior to a small fraction of group members
            if random.random() < dynamic_change_prob:
                if len(group) > 1:
                    group.remove(random.choice(list(group)))  # Remove one member
                group.add(random.randint(0, n - 1))  # Add a new random member
        
        for student in range(n):
            if any(student in group for group in colluding_groups):  # Colluding group behavior
                peers = random.sample(list(colluding_groups[0]), min(k, len(colluding_groups[0])))  # Share confirmations
            else:
                peers = random.sample([p for p in range(n) if p != student], k)  # Legitimate attendance
            daily_logs[student] = peers
        attendance[day] = daily_logs
    
    return attendance, colluding_groups

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

# Evaluate detection rate vs false positives
def evaluate_dynamic_detection(attendance, colluding_groups, k_values, m_values, iterations=50):
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
                suspicious_communities = analyze_communities(communities, graph)
                # Fallback: Randomly audit if no suspicious communities are found
                if not suspicious_communities:
                    suspicious_communities = [random.sample(range(n), min(m, n))]
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
attendance_logs, colluding_groups = generate_dynamic_attendance(n, k)
tdr_matrix, fpr_matrix = evaluate_dynamic_detection(attendance_logs, colluding_groups, k_values, m_values, iterations)

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
plt.title("True Detection Rate (TDR) with Dynamic Behavior")
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
plt.title("False Positive Rate (FPR) with Dynamic Behavior")
plt.show()
