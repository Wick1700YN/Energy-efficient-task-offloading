import numpy as np
from constants import *


# Functions for Path Prediction, Sojourn Time Calculation.
def predict_linear_path(start, direction, speed, time_steps):
    return np.array([start + speed * t * direction for t in time_steps])


def calculate_Tdmax(b_k, c_k, f_UE_k, K):
    Tdmax = 0
    for k in range(K):
        T_local = b_k * c_k / f_UE_k  # Calculate local execution time
        Tdmax += T_local  # Sum up the local execution times
    return Tdmax


# Sojourn Time Calculation
def calculate_sojourn_times(path1, path2, communication_range, time_steps):
    distances = np.linalg.norm(path1 - path2, axis=1)
    return np.sum(distances < communication_range) * (time_steps[1] - time_steps[0])
