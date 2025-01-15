import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Parameters
n = 200  # Total students
k = 3   # Minimum peer confirmations required
m = 5   # Number of random audits
iterations = 50  # Simulation iterations
false_attendance_rate = 0.2  # Probability of false attendance
collusion_rate = 0.2  # Probability that a group is colluding

# Generate random attendance logs
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

# Detect collusion based on repetition patterns
def detect_collusion(attendance):
    peer_confirmations = defaultdict(list)
    for day, logs in attendance.items():
        for student, peers in logs.items():
            for peer in peers:
                peer_confirmations[peer].append(student)

    # Collusion suspicion: number of unique confirmations a peer has given
    suspicion = {peer: len(set(confirmations)) for peer, confirmations in peer_confirmations.items()}
    return suspicion

# Random audit
def random_audit(attendance, m, false_attendance_rate=0.2):
    students = list(attendance[0].keys())
    audited = random.sample(students, m)
    false_positives = []
    for student in audited:
        if random.random() < false_attendance_rate:
            false_positives.append(student)
    return audited, false_positives

# Evaluate detection rate vs false positives
def evaluate_detection(attendance, colluding_groups, k_values, m_values, iterations=50):
    tdr_matrix = np.zeros((len(k_values), len(m_values)))
    fpr_matrix = np.zeros((len(k_values), len(m_values)))

    total_colluding_students = sum(len(group) for group in colluding_groups) * iterations  # Normalize by total colluders

    for i, k in enumerate(k_values):
        for j, m in enumerate(m_values):
            true_detected = 0
            false_detected = 0
            for _ in range(iterations):
                suspicion = detect_collusion(attendance)
                audited, false_positives = random_audit(attendance, m)
                
                # Check for collusion among audited students
                for student in audited:
                    if any(student in group for group in colluding_groups):
                        true_detected += 1
                    elif student in false_positives:
                        false_detected += 1
            tdr_matrix[i, j] = true_detected / total_colluding_students  # Correct normalization for TDR
            fpr_matrix[i, j] = false_detected / (n * iterations)  # Normalize by total students across iterations

            # Print TDR and FPR for each combination
            print(f"k={k}, m={m}: TDR={tdr_matrix[i, j]:.2f}, FPR={fpr_matrix[i, j]:.2f}")
    return tdr_matrix, fpr_matrix

# Simulation
k_values = [5, 10, 15, 20, 25, 30]  # Adjusted for n=200
m_values = [10, 20, 30, 40, 50]  # Adjusted for n=200
attendance_logs, colluding_groups = generate_attendance(n, k)
tdr_matrix, fpr_matrix = evaluate_detection(attendance_logs, colluding_groups, k_values, m_values, iterations)

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
plt.title("True Detection Rate (TDR) Heatmap (n=200)")
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
plt.title("False Positive Rate (FPR) Heatmap (n=200)")
plt.show()
