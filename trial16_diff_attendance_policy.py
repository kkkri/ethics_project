# import networkx as nx
# import numpy as np
# import pandas as pd
# import random
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Parameters for simulation
# n = 100  # Class size
# p_false_attendance = 0.1  # Probability of colluding students marked as present
# p_collusion = 0.2  # Fraction of students colluding
# k = 5  # Confirmation count
# m = 10  # Roll-call sample size
# simulation_rounds = 20  # Number of rounds for policy and influence testing

# # Initialize DataFrames for policy and social influence experiments
# policy_results = pd.DataFrame(columns=['Policy', 'false_positives_avg', 'penalties_avg', 'detection_accuracy'])
# influence_results = pd.DataFrame(columns=['Influence Model', 'false_positives_avg', 'penalties_avg', 'detection_accuracy'])

# def initialize_graph(n):
#     """Initialize the graph with students, set up collusion and attendance properties."""
#     G = nx.erdos_renyi_graph(n, 0.05)  # Random graph for simulation purposes
#     for i in G.nodes:
#         G.nodes[i]['attendance'] = True
#         G.nodes[i]['collusion'] = False
#     colluding_students = random.sample(list(G.nodes), int(n * p_collusion))
#     for student in colluding_students:
#         G.nodes[student]['collusion'] = True
#         G.nodes[student]['attendance'] = False if np.random.rand() < p_false_attendance else True
#     return G

# def perform_simulation(G, policy_type, influence_type):
#     """Run the simulation based on attendance policy and social influence models."""
#     total_false_positives = 0
#     total_penalties = 0
#     detection_accuracy_sum = 0
#     colluding_students = [n for n in G.nodes if G.nodes[n]['collusion']]
    
#     for _ in range(simulation_rounds):
#         # Roll-call always happens for simplicity
#         if policy_type == "Scheduled Confirmation Days":
#             confirmation_days = [5, 10, 15]
#             if random.choice(range(1, 21)) not in confirmation_days:
#                 continue  # Skip some rounds for scheduled days
#         elif policy_type == "Surprise Roll-Call" and np.random.rand() > 0.5:
#             continue  # Only call roll on 50% of rounds for surprise

#         # Influence Models
#         if influence_type == "Peer Pressure":
#             for student in colluding_students:
#                 if np.random.rand() < 0.7:  # Higher probability for peer influence
#                     G.nodes[student]['attendance'] = False
#         elif influence_type == "Fear of Penalties":
#             for student in colluding_students:
#                 if np.random.rand() < 0.5:  # Higher probability to avoid penalties
#                     G.nodes[student]['attendance'] = True

#         # Confirmation process: random selection of k peers
#         confirmations = {student: random.sample(list(G.nodes), min(k, len(G.nodes) - 1)) for student in G.nodes}
        
#         # Roll-call simulation
#         roll_call = random.sample(list(G.nodes), m)
#         false_positive = 0
#         penalties = 0
#         detected_colluders = set()
        
#         for student in roll_call:
#             actual_attendance = G.nodes[student]['attendance']
#             if not actual_attendance:
#                 detected_colluders.add(student)
#                 for peer in confirmations[student]:
#                     if G.nodes[peer]['attendance']:
#                         false_positive += 1
#                         penalties += 1

#         total_false_positives += false_positive
#         total_penalties += penalties
#         if len(colluding_students) > 0:
#             detection_accuracy_sum += len(detected_colluders.intersection(colluding_students)) / len(colluding_students)
    
#     # Compute averages for each metric
#     false_positives_avg = total_false_positives / simulation_rounds
#     penalties_avg = total_penalties / simulation_rounds
#     detection_accuracy_avg = detection_accuracy_sum / simulation_rounds

#     return false_positives_avg, penalties_avg, detection_accuracy_avg

# # Run simulations for different policies
# policies = ["Scheduled Confirmation Days", "Surprise Roll-Call"]
# influences = ["Peer Pressure", "Fear of Penalties"]

# for policy in policies:
#     G = initialize_graph(n)
#     false_positives_avg, penalties_avg, detection_accuracy_avg = perform_simulation(G, policy, None)
#     policy_results = policy_results._append({
#         'Policy': policy,
#         'false_positives_avg': false_positives_avg,
#         'penalties_avg': penalties_avg,
#         'detection_accuracy': detection_accuracy_avg
#     }, ignore_index=True)

# # Run simulations for different social influence models
# for influence in influences:
#     G = initialize_graph(n)
#     false_positives_avg, penalties_avg, detection_accuracy_avg = perform_simulation(G, "Surprise Roll-Call", influence)
#     influence_results = influence_results._append({
#         'Influence Model': influence,
#         'false_positives_avg': false_positives_avg,
#         'penalties_avg': penalties_avg,
#         'detection_accuracy': detection_accuracy_avg
#     }, ignore_index=True)

# # Visualization of policy results
# def plot_policy_results(policy_results, metric, title, cmap):
#     """Plot results of attendance policies for different metrics."""
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x='Policy', y=metric, data=policy_results, palette=cmap)
#     plt.title(f"{title} by Attendance Policy")
#     plt.ylabel(metric)
#     plt.show()

# # Visualization of social influence results
# def plot_influence_results(influence_results, metric, title, cmap):
#     """Plot results of social influence models for different metrics."""
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x='Influence Model', y=metric, data=influence_results, palette=cmap)
#     plt.title(f"{title} by Social Influence Model")
#     plt.ylabel(metric)
#     plt.show()

# # Plot the metrics for both attendance policies and social influence experiments
# plot_policy_results(policy_results, 'false_positives_avg', "False Positives", "Blues")
# plot_policy_results(policy_results, 'penalties_avg', "Penalties", "Reds")
# plot_policy_results(policy_results, 'detection_accuracy', "Detection Accuracy", "Greens")

# plot_influence_results(influence_results, 'false_positives_avg', "False Positives", "Blues")
# plot_influence_results(influence_results, 'penalties_avg', "Penalties", "Reds")
# plot_influence_results(influence_results, 'detection_accuracy', "Detection Accuracy", "Greens")
















# trial 2

# import networkx as nx
# import numpy as np
# import pandas as pd
# import random
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Parameters for simulation
# n = 100  # Class size
# p_false_attendance = 0.1  # Probability of colluding students marked as present
# p_collusion = 0.2  # Fraction of students colluding
# k = 5  # Confirmation count
# m = 10  # Roll-call sample size
# simulation_rounds = 20  # Number of rounds for policy and influence testing

# # Initialize DataFrames for policy and social influence experiments
# policy_results = pd.DataFrame(columns=['Policy', 'false_positives_avg', 'penalties_avg', 'detection_accuracy'])
# influence_results = pd.DataFrame(columns=['Influence Model', 'false_positives_avg', 'penalties_avg', 'detection_accuracy'])

# def initialize_graph(n):
#     """Initialize the graph with students, set up collusion and attendance properties."""
#     G = nx.erdos_renyi_graph(n, 0.05)  # Random graph for simulation purposes
#     for i in G.nodes:
#         G.nodes[i]['attendance'] = True
#         G.nodes[i]['collusion'] = False
#     colluding_students = random.sample(list(G.nodes), int(n * p_collusion))
#     for student in colluding_students:
#         G.nodes[student]['collusion'] = True
#         G.nodes[student]['attendance'] = False if np.random.rand() < p_false_attendance else True
#     return G

# def perform_simulation(G, policy_type, influence_type):
#     """Run the simulation based on attendance policy and social influence models."""
#     total_false_positives = 0
#     total_penalties = 0
#     detection_accuracy_sum = 0
#     colluding_students = [n for n in G.nodes if G.nodes[n]['collusion']]
    
#     for _ in range(simulation_rounds):
#         # Roll-call always happens for simplicity
#         if policy_type == "Scheduled Confirmation Days":
#             confirmation_days = [5, 10, 15]
#             if random.choice(range(1, 21)) not in confirmation_days:
#                 continue  # Skip some rounds for scheduled days
#         elif policy_type == "Surprise Roll-Call" and np.random.rand() > 0.5:
#             continue  # Only call roll on 50% of rounds for surprise

#         # Influence Models
#         if influence_type == "Peer Pressure":
#             for student in colluding_students:
#                 if np.random.rand() < 0.7:  # Higher probability for peer influence
#                     G.nodes[student]['attendance'] = False
#         elif influence_type == "Fear of Penalties":
#             for student in colluding_students:
#                 if np.random.rand() < 0.3:  # Lower probability to switch attendance
#                     G.nodes[student]['attendance'] = True
#                 else:
#                     # Small chance they still avoid attending, simulating partial fear effect
#                     G.nodes[student]['attendance'] = False

#         # Confirmation process: random selection of k peers
#         confirmations = {student: random.sample(list(G.nodes), min(k, len(G.nodes) - 1)) for student in G.nodes}
        
#         # Roll-call simulation
#         roll_call = random.sample(list(G.nodes), m)
#         false_positive = 0
#         penalties = 0
#         detected_colluders = set()
        
#         for student in roll_call:
#             actual_attendance = G.nodes[student]['attendance']
#             if not actual_attendance:
#                 detected_colluders.add(student)
#                 for peer in confirmations[student]:
#                     if G.nodes[peer]['attendance']:
#                         false_positive += 1
#                         penalties += 1

#         total_false_positives += false_positive
#         total_penalties += penalties
#         if len(colluding_students) > 0:
#             detection_accuracy_sum += len(detected_colluders.intersection(colluding_students)) / len(colluding_students)
    
#     # Compute averages for each metric
#     false_positives_avg = total_false_positives / simulation_rounds
#     penalties_avg = total_penalties / simulation_rounds
#     detection_accuracy_avg = detection_accuracy_sum / simulation_rounds

#     return false_positives_avg, penalties_avg, detection_accuracy_avg

# # Run simulations for different policies
# policies = ["Scheduled Confirmation Days", "Surprise Roll-Call"]
# influences = ["Peer Pressure", "Fear of Penalties"]

# for policy in policies:
#     G = initialize_graph(n)
#     false_positives_avg, penalties_avg, detection_accuracy_avg = perform_simulation(G, policy, None)
#     policy_results = policy_results._append({
#         'Policy': policy,
#         'false_positives_avg': false_positives_avg,
#         'penalties_avg': penalties_avg,
#         'detection_accuracy': detection_accuracy_avg
#     }, ignore_index=True)

# # Run simulations for different social influence models
# for influence in influences:
#     G = initialize_graph(n)
#     false_positives_avg, penalties_avg, detection_accuracy_avg = perform_simulation(G, "Surprise Roll-Call", influence)
#     influence_results = influence_results._append({
#         'Influence Model': influence,
#         'false_positives_avg': false_positives_avg,
#         'penalties_avg': penalties_avg,
#         'detection_accuracy': detection_accuracy_avg
#     }, ignore_index=True)

# # Visualization of policy results
# def plot_policy_results(policy_results, metric, title, cmap):
#     """Plot results of attendance policies for different metrics."""
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x='Policy', y=metric, data=policy_results, palette=cmap)
#     plt.title(f"{title} by Attendance Policy")
#     plt.ylabel(metric)
#     plt.show()

# # Visualization of social influence results
# def plot_influence_results(influence_results, metric, title, cmap):
#     """Plot results of social influence models for different metrics."""
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x='Influence Model', y=metric, data=influence_results, palette=cmap)
#     plt.title(f"{title} by Social Influence Model")
#     plt.ylabel(metric)
#     plt.show()

# # Plot the metrics for both attendance policies and social influence experiments
# plot_policy_results(policy_results, 'false_positives_avg', "False Positives", "Blues")
# plot_policy_results(policy_results, 'penalties_avg', "Penalties", "Reds")
# plot_policy_results(policy_results, 'detection_accuracy', "Detection Accuracy", "Greens")

# plot_influence_results(influence_results, 'false_positives_avg', "False Positives", "Blues")
# plot_influence_results(influence_results, 'penalties_avg', "Penalties", "Reds")
# plot_influence_results(influence_results, 'detection_accuracy', "Detection Accuracy", "Greens")




















#trial 3

import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters for simulation
n = 100  # Class size
p_false_attendance = 0.1  # Probability of colluding students marked as present
p_collusion = 0.2  # Fraction of students colluding
k = 5  # Confirmation count
m = 10  # Roll-call sample size
simulation_rounds = 20  # Number of rounds for policy and influence testing

# Initialize DataFrames for policy and social influence experiments
policy_results = pd.DataFrame(columns=['Policy', 'false_positives_avg', 'penalties_avg', 'detection_accuracy'])
influence_results = pd.DataFrame(columns=['Influence Model', 'false_positives_avg', 'penalties_avg', 'detection_accuracy'])

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

def perform_simulation(G, policy_type, influence_type):
    """Run the simulation based on attendance policy and social influence models."""
    total_false_positives = 0
    total_penalties = 0
    detection_accuracy_sum = 0
    colluding_students = [n for n in G.nodes if G.nodes[n]['collusion']]
    
    for round_num in range(simulation_rounds):
        # Determine whether to conduct roll-call based on policy
        if policy_type == "Scheduled Confirmation Days":
            # Conduct roll-call on specific scheduled days (every 4 rounds in this example)
            if round_num % 4 != 0:
                continue  # Skip if it's not a scheduled round
        elif policy_type == "Surprise Roll-Call":
            # Conduct roll-call with a lower probability (only 30% chance)
            if np.random.rand() > 0.3:
                continue  # Skip roll-call for this round

        # Influence Models
        if influence_type == "Peer Pressure":
            for student in colluding_students:
                if np.random.rand() < 0.7:  # Higher probability for peer influence
                    G.nodes[student]['attendance'] = False
        elif influence_type == "Fear of Penalties":
            for student in colluding_students:
                if np.random.rand() < 0.3:  # Lower probability to switch attendance
                    G.nodes[student]['attendance'] = True
                else:
                    # Small chance they still avoid attending, simulating partial fear effect
                    G.nodes[student]['attendance'] = False

        # Confirmation process: random selection of k peers
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
        if len(colluding_students) > 0:
            detection_accuracy_sum += len(detected_colluders.intersection(colluding_students)) / len(colluding_students)
    
    # Compute averages for each metric
    false_positives_avg = total_false_positives / simulation_rounds
    penalties_avg = total_penalties / simulation_rounds
    detection_accuracy_avg = detection_accuracy_sum / simulation_rounds

    return false_positives_avg, penalties_avg, detection_accuracy_avg

# Run simulations for different policies
policies = ["Scheduled Confirmation Days", "Surprise Roll-Call"]
influences = ["Peer Pressure", "Fear of Penalties"]

for policy in policies:
    G = initialize_graph(n)
    false_positives_avg, penalties_avg, detection_accuracy_avg = perform_simulation(G, policy, None)
    policy_results = policy_results._append({
        'Policy': policy,
        'false_positives_avg': false_positives_avg,
        'penalties_avg': penalties_avg,
        'detection_accuracy': detection_accuracy_avg
    }, ignore_index=True)

# Run simulations for different social influence models
for influence in influences:
    G = initialize_graph(n)
    false_positives_avg, penalties_avg, detection_accuracy_avg = perform_simulation(G, "Surprise Roll-Call", influence)
    influence_results = influence_results._append({
        'Influence Model': influence,
        'false_positives_avg': false_positives_avg,
        'penalties_avg': penalties_avg,
        'detection_accuracy': detection_accuracy_avg
    }, ignore_index=True)

# Visualization of policy results
def plot_policy_results(policy_results, metric, title, cmap):
    """Plot results of attendance policies for different metrics."""
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Policy', y=metric, data=policy_results, palette=cmap)
    plt.title(f"{title} by Attendance Policy")
    plt.ylabel(metric)
    plt.show()

# Visualization of social influence results
def plot_influence_results(influence_results, metric, title, cmap):
    """Plot results of social influence models for different metrics."""
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Influence Model', y=metric, data=influence_results, palette=cmap)
    plt.title(f"{title} by Social Influence Model")
    plt.ylabel(metric)
    plt.show()

# Plot the metrics for both attendance policies and social influence experiments
plot_policy_results(policy_results, 'false_positives_avg', "False Positives", "Blues")
plot_policy_results(policy_results, 'penalties_avg', "Penalties", "Reds")
plot_policy_results(policy_results, 'detection_accuracy', "Detection Accuracy", "Greens")

plot_influence_results(influence_results, 'false_positives_avg', "False Positives", "Blues")
plot_influence_results(influence_results, 'penalties_avg', "Penalties", "Reds")
plot_influence_results(influence_results, 'detection_accuracy', "Detection Accuracy", "Greens")




# trial 4


# import networkx as nx
# import numpy as np
# import pandas as pd
# import random
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Parameters for simulation
# n = 100  # Class size
# p_false_attendance = 0.1  # Probability of colluding students marked as present
# p_collusion = 0.2  # Fraction of students colluding
# k = 5  # Confirmation count
# m = 10  # Roll-call sample size
# simulation_rounds = 20  # Number of rounds for policy and influence testing

# # Initialize DataFrames for policy and social influence experiments
# policy_results = pd.DataFrame(columns=['Policy', 'false_positives_avg', 'penalties_avg', 'detection_accuracy'])
# influence_results = pd.DataFrame(columns=['Influence Model', 'false_positives_avg', 'penalties_avg', 'detection_accuracy'])

# def initialize_graph(n):
#     """Initialize the graph with students, set up collusion and attendance properties."""
#     G = nx.erdos_renyi_graph(n, 0.05)  # Random graph for simulation purposes
#     for i in G.nodes:
#         G.nodes[i]['attendance'] = True
#         G.nodes[i]['collusion'] = False
#     colluding_students = random.sample(list(G.nodes), int(n * p_collusion))
#     for student in colluding_students:
#         G.nodes[student]['collusion'] = True
#         G.nodes[student]['attendance'] = False if np.random.rand() < p_false_attendance else True
#     return G

# def perform_simulation(G, policy_type, influence_type):
#     """Run the simulation based on attendance policy and social influence models."""
#     total_false_positives = 0
#     total_penalties = 0
#     detection_accuracy_sum = 0
#     roll_call_count = 0  # Count actual roll-calls to calculate accurate averages
#     colluding_students = [n for n in G.nodes if G.nodes[n]['collusion']]
    
#     for round_num in range(simulation_rounds):
#         # Determine whether to conduct roll-call based on policy
#         if policy_type == "Scheduled Confirmation Days":
#             # Conduct roll-call on specific scheduled days (every 4 rounds in this example)
#             if round_num % 4 != 0:
#                 continue  # Skip if it's not a scheduled round
#         elif policy_type == "Surprise Roll-Call":
#             # Increase probability to 50% for more frequent roll-calls
#             if np.random.rand() > 0.5:
#                 continue  # Skip roll-call for this round

#         roll_call_count += 1  # Increment count of rounds with roll-call

#         # Influence Models
#         if influence_type == "Peer Pressure":
#             for student in colluding_students:
#                 if np.random.rand() < 0.7:  # Higher probability for peer influence
#                     G.nodes[student]['attendance'] = False
#         elif influence_type == "Fear of Penalties":
#             for student in colluding_students:
#                 if np.random.rand() < 0.3:  # Lower probability to switch attendance
#                     G.nodes[student]['attendance'] = True
#                 else:
#                     # Small chance they still avoid attending, simulating partial fear effect
#                     G.nodes[student]['attendance'] = False

#         # Confirmation process: random selection of k peers
#         confirmations = {student: random.sample(list(G.nodes), min(k, len(G.nodes) - 1)) for student in G.nodes}
        
#         # Roll-call simulation
#         roll_call = random.sample(list(G.nodes), m)
#         false_positive = 0
#         penalties = 0
#         detected_colluders = set()
        
#         for student in roll_call:
#             actual_attendance = G.nodes[student]['attendance']
#             if not actual_attendance:
#                 detected_colluders.add(student)
#                 for peer in confirmations[student]:
#                     if G.nodes[peer]['attendance']:
#                         false_positive += 1
#                         penalties += 1

#         total_false_positives += false_positive
#         total_penalties += penalties
#         if len(colluding_students) > 0:
#             detection_accuracy_sum += len(detected_colluders.intersection(colluding_students)) / len(colluding_students)
    
#     # Compute averages for each metric, dividing by actual roll-call rounds
#     if roll_call_count > 0:
#         false_positives_avg = total_false_positives / roll_call_count
#         penalties_avg = total_penalties / roll_call_count
#         detection_accuracy_avg = detection_accuracy_sum / roll_call_count
#     else:
#         false_positives_avg, penalties_avg, detection_accuracy_avg = 0, 0, 0

#     return false_positives_avg, penalties_avg, detection_accuracy_avg

# # Run simulations for different policies
# policies = ["Scheduled Confirmation Days", "Surprise Roll-Call"]
# influences = ["Peer Pressure", "Fear of Penalties"]

# for policy in policies:
#     G = initialize_graph(n)
#     false_positives_avg, penalties_avg, detection_accuracy_avg = perform_simulation(G, policy, None)
#     policy_results = policy_results._append({
#         'Policy': policy,
#         'false_positives_avg': false_positives_avg,
#         'penalties_avg': penalties_avg,
#         'detection_accuracy': detection_accuracy_avg
#     }, ignore_index=True)

# # Run simulations for different social influence models
# for influence in influences:
#     G = initialize_graph(n)
#     false_positives_avg, penalties_avg, detection_accuracy_avg = perform_simulation(G, "Surprise Roll-Call", influence)
#     influence_results = influence_results._append({
#         'Influence Model': influence,
#         'false_positives_avg': false_positives_avg,
#         'penalties_avg': penalties_avg,
#         'detection_accuracy': detection_accuracy_avg
#     }, ignore_index=True)

# # Visualization of policy results
# def plot_policy_results(policy_results, metric, title, cmap):
#     """Plot results of attendance policies for different metrics."""
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x='Policy', y=metric, data=policy_results, palette=cmap)
#     plt.title(f"{title} by Attendance Policy")
#     plt.ylabel(metric)
#     plt.show()

# # Visualization of social influence results
# def plot_influence_results(influence_results, metric, title, cmap):
#     """Plot results of social influence models for different metrics."""
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x='Influence Model', y=metric, data=influence_results, palette=cmap)
#     plt.title(f"{title} by Social Influence Model")
#     plt.ylabel(metric)
#     plt.show()

# # Plot the metrics for both attendance policies and social influence experiments
# plot_policy_results(policy_results, 'false_positives_avg', "False Positives", "Blues")
# plot_policy_results(policy_results, 'penalties_avg', "Penalties", "Reds")
# plot_policy_results(policy_results, 'detection_accuracy', "Detection Accuracy", "Greens")

# plot_influence_results(influence_results, 'false_positives_avg', "False Positives", "Blues")
# plot_influence_results(influence_results, 'penalties_avg', "Penalties", "Reds")
# plot_influence_results(influence_results, 'detection_accuracy', "Detection Accuracy", "Greens")

