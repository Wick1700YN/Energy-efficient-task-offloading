import numpy as np
from constants import *
from utilities import predict_linear_path, calculate_Tdmax, calculate_sojourn_times
from energy_computations import compute_exec_time_and_energy
from matrix_construction import (compute_n1k, create_ec_matrices, compute_ej, compute_n3,
                                 create_matrices, create_M1_matrices, create_M2_matrices,
                                 create_M5_matrices, create_M3_dash_matrices)
from sdp_solver import setup_sdp_problem


def stochastic_mapping(G, L, Tdmax, nT, n0, nTs):
    FTk_values = []  # To store FTk values for for all tasks
    num_elements_per_task = 1 + I + M + I * M
    best_total_FTk = float('inf')
    best_alpha = None
    if np.linalg.matrix_rank(G) == 1:
        best_alpha = G[-1, :num_elements_per_task * K]
    else:
        for l in range(L):
            alpha_l = G[-1, :num_elements_per_task * K]
            total_FTk = 0
            FTk_values_l = []  # FTk values for this iteration
            feasible_alpha = np.copy(alpha_l)
            for k in range (K):
                is_feasible = False
                while not is_feasible:
                    start_idx = num_elements_per_task * k
                    end_idx = start_idx + num_elements_per_task
                    segment = feasible_alpha[start_idx:end_idx]
                    exec_time_segment = nT[start_idx:end_idx]
                    sojourn_time_segment = nTs[start_idx:end_idx]
                    # Perform probability-based stochastic mapping
                    # Calculate unique values and their counts
                    unique, counts = np.unique(segment, return_counts=True)
                    # Calculate probabilities as unique values multiplied by their counts
                    probabilities = unique * counts
                    # Find the maximum probability value
                    max_probability = np.max(probabilities)
                    # Identify indices in 'probabilities' that have this max value
                    max_indices = np.where(probabilities == max_probability)[0]
                    # Randomly select one of these indices if there are ties
                    selected_index = np.random.choice(max_indices)
                    # Find the unique value corresponding to this selected index
                    selected_unique_value = unique[selected_index]
                    # Initialize the updated segment with zeros
                    segment_updated = np.zeros_like(segment)
                    # Find all occurrences of the selected unique value
                    occurrence_indices = np.where(segment == selected_unique_value)[0]
                    # Randomly select one of these occurrences
                    random_occurrence_index = np.random.choice(occurrence_indices)
                    # Set this randomly selected occurrence to 1
                    segment_updated[random_occurrence_index] = 1
                    # Update the feasible_alpha with the updated segment
                    feasible_alpha[start_idx:end_idx] = segment_updated
                    # Calculate execution time and energy for the current task
                    Texe = segment_updated * exec_time_segment
                    task_exec_time = np.max(Texe)            
                    Tsk = segment_updated * sojourn_time_segment
                    task_sojourn_time = np.max(Tsk) 
                    # Update total finish time and energy
                    if k == 0:
                        FTk = task_exec_time
                    else:
                        FTk += task_exec_time
                    if FTk <= task_sojourn_time:
                        is_feasible = True
                        FTk_values_l.append(FTk) 
                        total_FTk = FTk_values_l[-1]                 
                        if total_FTk > Tdmax:
                            break
                if not is_feasible or total_FTk > Tdmax:
                    break  # If not feasible or total exceeds Tdmax, start a new iteration of L
            if is_feasible and total_FTk <= Tdmax and total_FTk < best_total_FTk:
                best_alpha = np.copy(feasible_alpha)
                best_total_FTk = total_FTk
                FTk_values = FTk_values_l.copy()        
        return best_alpha, FTk_values, best_total_FTk


def calculate_minimum_energy(best_alpha, n0):    
    # Extract the relevant segment from n0 for all tasks and decisions
    energy_costs_segment = n0[:len(best_alpha)]
    # Calculate the total energy cost by multiplying best_alpha with n0
    min_energy_cost = np.sum(best_alpha * energy_costs_segment)
    return min_energy_cost