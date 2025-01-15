import streamlit as st
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

# Initialize DataFrames for tracking metrics per round
simulation_data = []

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

def run_simulation_round(n, k, m, round_num):
    """Run one round of simulation and collect metrics."""
    G = initialize_graph(n)  # Initialize a new graph for each round to reset attendance statuses
    
    # Roll-call probability
    roll_call = np.random.rand() < 0.5
    false_positives = 0
    penalties = 0
    detected_colluders = set()
    
    if roll_call:
        # Confirmation process: random selection of k peers
        confirmations = {student: random.sample(list(G.nodes), min(k, len(G.nodes) - 1)) for student in G.nodes}
        
        # Roll-call
        roll_call_students = random.sample(list(G.nodes), m)
        for student in roll_call_students:
            if not G.nodes[student]['attendance']:
                detected_colluders.add(student)
                for peer in confirmations[student]:
                    if G.nodes[peer]['attendance']:
                        false_positives += 1
                        penalties += 1

    # Calculate detection accuracy
    colluding_students = [n for n in G.nodes if G.nodes[n]['collusion']]
    detection_accuracy = len(detected_colluders.intersection(colluding_students)) / len(colluding_students) if colluding_students else 0
    
    # Collect data for this round
    round_data = {
        "round": round_num,
        "false_positives": false_positives,
        "penalties": penalties,
        "detection_accuracy": detection_accuracy
    }
    simulation_data.append(round_data)
    return round_data, G  # Return graph G for visualization

# Streamlit UI
st.title("Real-Time Simulation Dashboard for Attendance Confirmation")

# User input for simulation parameters
n = st.sidebar.slider("Class size (n)", 50, 200, 100)
k = st.sidebar.slider("Confirmation count (k)", 1, 10, 5)
m = st.sidebar.slider("Roll-call sample size (m)", 1, 20, 10)
simulation_rounds = st.sidebar.slider("Number of simulation rounds", 10, 100, 20)

# Run and display simulation data
st.subheader("Simulation Results")

if st.button("Run Simulation"):
    st.write("Running simulation...")
    simulation_data.clear()  # Clear previous data

    for round_num in range(simulation_rounds):
        round_data, G = run_simulation_round(n, k, m, round_num + 1)
        
        # Displaying metrics dynamically as each round completes
        st.write(f"**Round {round_data['round']}**:")
        st.write(f"- False Positives: {round_data['false_positives']}")
        st.write(f"- Penalties: {round_data['penalties']}")
        st.write(f"- Detection Accuracy: {round_data['detection_accuracy']:.2%}")

        # Display graph structure for each round
        fig, ax = plt.subplots()
        color_map = ["green" if G.nodes[node]['attendance'] else "red" for node in G]
        nx.draw(G, node_color=color_map, node_size=50, ax=ax)
        st.pyplot(fig)

# Display summary metrics after all rounds
if simulation_data:
    # Convert simulation data to DataFrame
    df_simulation = pd.DataFrame(simulation_data)
    
    # Display line charts for each metric
    st.line_chart(df_simulation.set_index("round")[["false_positives", "penalties", "detection_accuracy"]])

    # Playback simulation data
    st.subheader("Playback Simulation Timeline")
    playback_round = st.slider("Select round to playback", 1, simulation_rounds, 1)
    selected_round_data = df_simulation[df_simulation["round"] == playback_round]
    st.write(f"**Round {playback_round}** Playback:")
    st.write(f"- False Positives: {selected_round_data['false_positives'].values[0]}")
    st.write(f"- Penalties: {selected_round_data['penalties'].values[0]}")
    st.write(f"- Detection Accuracy: {selected_round_data['detection_accuracy'].values[0]:.2%}")
    
    # Plot network graph with node colors based on attendance status
    fig, ax = plt.subplots()
    color_map = ["green" if G.nodes[node]['attendance'] else "red" for node in G]
    nx.draw(G, node_color=color_map, node_size=50, ax=ax)
    st.pyplot(fig)

# Display data table of all rounds
st.subheader("Simulation Data Table")
st.dataframe(df_simulation)
