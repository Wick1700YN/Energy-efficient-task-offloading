import numpy as np
from constants import *
from utilities import predict_linear_path, calculate_Tdmax, calculate_sojourn_times


def compute_exec_time_and_energy():
    nT = []  # Array to store execution time for each task
    n0 = []  # Array to store energy consumption for each task
    n2 = np.zeros((2 + I + M + I * M) * K + 1)
    # Loop over each task
    for k in range(K):
        # Local Execution Time and Energy
        local_exec_time = b_k * c_k / f_UE_k
        local_energy = e_UE * (f_UE_k**2) * b_k * c_k
        nT.append(local_exec_time)
        n0.append(local_energy)
        # Local execution for the last task
        # Set the execution times for the last task for each offloading method
        offset = (1 + I + M + I * M) * (K - 1)
        n2[offset] = local_exec_time
        # Check offloading to each helper
        for i in range(I):
            offload_time_to_helper = (
                b_k / (B * np.log2(1 + (P_tr_k * H_k) / sigma_squared))
                + b_k * c_k / f_UHi_k
            )
            offload_energy_to_helper = (
                P_tr_k * (b_k / (B * np.log2(1 + (P_tr_k * H_k) / sigma_squared)))
                + e_UHi * (f_UHi_k**2) * b_k * c_k
                + P_wait_k * (b_k * c_k / f_UHi_k)
            )
            nT.append(offload_time_to_helper)
            n0.append(offload_energy_to_helper)
            n2[offset + i + 1] = offload_time_to_helper
        # Check offloading to each server
        for m in range(M):
            # Data rate between UE and server m for task k
            r_Sm_k = B * np.log2(1 + (P_tr_k * H_k_Sm) / sigma_squared)
            # Execution Time for task k offloaded to server m
            offload_time_to_server = b_k / r_Sm_k + b_k * c_k / f_Sm_k
            # Energy Consumption for task k offloaded to server m
            offload_energy_to_server = (
                P_tr_k * (b_k / r_Sm_k) + P_wait_k * (b_k * c_k) / f_Sm_k
            )
            nT.append(offload_time_to_server)
            n0.append(offload_energy_to_server)
            n2[offset + I + m + 1] = offload_time_to_server
        # Check offloading to each server via each helper
        for m in range(M):
            for i in range(I):
                # Data rate between UE and helper i for task k
                r_UHi_k = B * np.log2(1 + (P_tr_k * H_k) / sigma_squared)
                # Data rate between helper i and server m for task k
                r_Sm_i_k = B * np.log2(1 + (P_tr_k * H_k_Sm) / sigma_squared)
                # Execution Time for task k offloaded from UE to helper i to server m
                offload_time_to_server_via_helper = (
                    b_k / r_UHi_k + b_k / r_Sm_i_k + b_k * c_k / f_Sm_k
                )
                # Energy Consumption for task k offloaded from UE to helper i to server m
                offload_energy_to_server_via_helper = P_tr_k * (
                    b_k / r_UHi_k
                ) + P_wait_k * ((b_k / r_Sm_i_k) + (b_k * c_k / f_Sm_k))
                nT.append(offload_time_to_server_via_helper)
                n0.append(offload_energy_to_server_via_helper)
                n2[offset + 1 + I + M + i * M + m] = offload_time_to_server_via_helper
    # Append additional elements to nT and n0
    nT.extend([1] * (K + 1))
    n0.extend([0] * (K + 1))
    # Set the second last element to 1, rest remain 0
    n2[-2] = 1
    # Convert to numpy arrays
    nT = np.array(nT)
    n0 = np.array(n0)
    n2 = np.array(n2)
    return nT, n0, n2
