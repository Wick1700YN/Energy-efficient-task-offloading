import time
import numpy as np
from constants import *
from utilities import predict_linear_path, calculate_Tdmax, calculate_sojourn_times
from energy_computations import compute_exec_time_and_energy
from matrix_construction import (compute_n1k, create_ec_matrices, compute_ej, compute_n3,
                                 create_matrices, create_M1_matrices, create_M2_matrices,
                                 create_M5_matrices, create_M3_dash_matrices)
from sdp_solver import setup_sdp_problem
from stochastic_mapping import stochastic_mapping, calculate_minimum_energy


def main():
    # Define parameters for path prediction
    start_time = time.time()  # Start timing
    UE_start, helpers_start, servers_start = (
        np.array([0, 0]),
        np.random.rand(I, 2),
        np.random.rand(M, 2),
    )
    UE_direction, helpers_direction = np.random.rand(2), np.random.rand(I, 2)
    UE_speed, helpers_speed = 1, 0.7
    time_steps = np.linspace(0, 100, 100)
    # Path Prediction
    UE_path = predict_linear_path(UE_start, UE_direction, UE_speed, time_steps)
    helpers_paths = np.array(
        [
            predict_linear_path(
                helpers_start[i], helpers_direction[i], helpers_speed, time_steps
            )
            for i in range(I)
        ]
    )
    # calculate Tdmax
    Tdmax = calculate_Tdmax(b_k, c_k, f_UE_k, K)
    # Sojourn Time Calculation
    sojourn_times_dict = {}
    nTs = []
    for k in range(K):
        nTs.append(Tdmax)
        # For each task, calculate sojourn times for each offloading option
        # UE and helpers
        for i in range(I):
            key = ("UE", "Helper", i)
            sojourn_times_dict[key] = calculate_sojourn_times(
                UE_path, helpers_paths[i], helper_communication_range, time_steps
            )
            nTs.append(sojourn_times_dict[("UE", "Helper", i)])
            # Append sojourn times for UE and servers
        for m in range(M):
            nTs.append(
                calculate_sojourn_times(
                    UE_path, servers_start[m], server_communication_range, time_steps
                )
            )
        # Servers and helpers
        for m in range(M):
            for i in range(I):
                key = ("Server", m, "Helper", i)
                sojourn_times_dict[key] = calculate_sojourn_times(
                    servers_start[m],
                    helpers_paths[i],
                    server_communication_range,
                    time_steps,
                )
        # Server via Helper Calculations
        for m in range(M):
            for i in range(I):
                sojourn_time_UE_helper = sojourn_times_dict[("UE", "Helper", i)]
                sojourn_time_server_helper = sojourn_times_dict[
                    ("Server", m, "Helper", i)
                ]
                sojourn_time_UE_server = calculate_sojourn_times(
                    UE_path, servers_start[m], server_communication_range, time_steps
                )
                min_sojourn_time = min(
                    sojourn_time_UE_helper,
                    sojourn_time_server_helper,
                    sojourn_time_UE_server,
                )
                nTs.append(min_sojourn_time)
    # Append additional elements at the end
    nTs.extend([1] * (K + 1))
    # Store in nTs
    nTs = np.array(nTs)
    print("nTs:", nTs)
    print(nTs.shape)
    # Compute Execution Time and Energy Consumption
    nT, n0, n2 = compute_exec_time_and_energy()
    n1k_matrices = compute_n1k(K, I, M)
    # Accessing individual M3_dash matrices
    for key, n1k in n1k_matrices.items():
        print(f"n1k_ for {key}: \n", n1k)
    ec_matrices = create_ec_matrices(K, I, M)
    for key, ec in ec_matrices.items():
        print(f"ec for {key}: \n", ec)
        print(ec.shape)
    ej_matrices = compute_ej(K, I, M)
    for key, ej in ej_matrices.items():
        print(f"ej for {key}: \n", ej)
        print(ej.shape)
    n3_matrices = compute_n3(K, I, M)
    for key, n3 in n3_matrices.items():
        print(f"n3 for {key}: \n", n3)
        print(n3.shape)
        print(np.diag(n3))
    M0, M3, M4 = create_matrices(n0, n2)
    M1_matrices = create_M1_matrices(ej_matrices)
    # Accessing individual M1 matrices
    for key, M1 in M1_matrices.items():
        print(f"M1 for {key}: \n", M1)
        print(M1.shape) 
    M2_matrices = create_M2_matrices(n1k_matrices, a)
    # Accessing individual M2 matrices
    for key, M2 in M2_matrices.items():
        print(f"M2 for {key}: \n", M2)
        print(M2.shape)
    M5_matrices = create_M5_matrices(n1k_matrices, nT, nTs)
    # Accessing individual M5 matrices
    for key, M5 in M5_matrices.items():
        print(f"M5 for {key}: \n", M5)
        print(M5.shape)
    M3_dash_matrices = create_M3_dash_matrices(n3_matrices, nT, ec_matrices)
    # Accessing individual M3_dash matrices
    for key, M3_dash in M3_dash_matrices.items():
        print(f"M3_dash for {key}: \n", M3_dash)
        print(M3_dash.shape)
    G = setup_sdp_problem(
        M0, M1_matrices, M2_matrices, M3, M3_dash_matrices, M4, M5_matrices, Tdmax, a, K
    )
    print("G:", G)
    end_time = time.time()  # End timing
    compilation_time = end_time - start_time  # Calculate compilation time
    best_alpha, FTk_values, best_total_FTk = stochastic_mapping(G, L, Tdmax, nT, n0, nTs)
    print("best_alpha",best_alpha)
    print("best_total_FTk", best_total_FTk)
    print("FTk_values", FTk_values)
    print("Tdmax:", Tdmax)
    min_energy_cost = calculate_minimum_energy(best_alpha, n0)
    print("min_energy_cost:", min_energy_cost)
    print("compilation_time :", compilation_time)
if __name__ == "__main__":
    main()
