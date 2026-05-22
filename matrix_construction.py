import numpy as np
from constants import *
from utilities import predict_linear_path, calculate_Tdmax, calculate_sojourn_times
from energy_computations import compute_exec_time_and_energy

def compute_n1k(K, I, M):
    n1k_matrices = {}
    size = (2 + I + M + I * M) * K + 1
    for k in range(1, K + 1):
        n1k = np.zeros(size)
        # Assuming 'e' is a base vector (with 1 at specific positions)
        indices = [
            (1 + I + M + I * M) * (k - 1) + 1 + i for i in range(1 + I + M + I * M)
        ]
        indices.append((1 + I + M + I * M) * K + k)
        for idx in indices:
            n1k[idx - 1] = 1  # Subtract 1 because numpy arrays are zero-indexed
        n1k_matrices[f"n1{k}"] = n1k
    return n1k_matrices


def create_ec_matrices(K, I, M):
    ec_matrices = {}
    size = (2 + I + M + I * M) * K + 1
    for k in range(1, K + 1):
        ec = np.zeros(size)
        c = (1 + I + M + I * M) * K + k
        ec[c - 1] = 1  # Subtract 1 because numpy arrays are zero-indexed
        ec_matrices[f"e{k}"] = ec
    return ec_matrices


def compute_ej(K, I, M):
    ej_matrices = {}
    ej_length = (2 + I + M + I * M) * K + 1
    for j in range(1, (1 + I + M + I * M) * K + 1):
        ej = np.zeros(ej_length)
        # Set only the jth entry to 1
        if j <= ej_length:
            ej[j - 1] = 1  # Subtract 1 because numpy arrays are zero-indexed
        ej_matrices[f"e{j}"] = ej
    return ej_matrices


def compute_n3(K, I, M):
    n3_matrices = {}
    n3_length = (2 + I + M + I * M) * K + 1
    for j in range(1, (1 + I + M + I * M) * K + 1):
        n3 = np.zeros(n3_length)
        indices = [
            (1 + I + M + I * M) * (j - 1) + 1 + i for i in range(1 + I + M + I * M)
        ]
        indices.append((1 + I + M + I * M) * K + j)
        for idx in indices:
            if idx < n3_length:
                n3[idx - 1] = 1  # Subtract 1 because numpy arrays are zero-indexed
                n3_matrices[f"n3{j}"] = n3
    return n3_matrices


def create_matrices(n0, n2):
    zero_block = np.zeros((a + 1, a + 1))  # Zero block of size (a+1) x (a+1)
    # Constructing M0
    M0_top = np.hstack([zero_block, 0.5 * n0[:, np.newaxis]])
    M0_bottom = np.hstack([0.5 * n0, 0])
    M0 = np.vstack([M0_top, M0_bottom])

    # Constructing M3
    length = (2 + I + M + I * M) * K + 1
    eb = np.zeros(length)
    eb[b - 1] = 1
    M3_top = np.hstack([zero_block, 0.5 * eb[:, np.newaxis]])
    M3_bottom = np.hstack([0.5 * eb, 0])
    M3 = np.vstack([M3_top, M3_bottom])
    # Constructing M4
    M4_top = np.hstack([zero_block, 0.5 * n2[:, np.newaxis]])
    M4_bottom = np.hstack([0.5 * n2, 0])
    M4 = np.vstack([M4_top, M4_bottom])
    return M0, M3, M4


def create_M1_matrices(ej_matrices):
    M1_matrices = {}
    for key, ej in ej_matrices.items():
        diag_ej = np.diag(ej) 
        M1_top = np.hstack([diag_ej, -0.5 * ej[:, np.newaxis]])
        M1_bottom = np.hstack([-0.5 * ej, 0])
        M1 = np.vstack([M1_top, M1_bottom])
        M1_matrices[key] = M1
    return M1_matrices


def create_M2_matrices(n1k_matrices, a):
    M2_matrices = {}
    for key, n1k in n1k_matrices.items():
        zero_block = np.zeros((a + 1, a + 1))
        M2_top = np.hstack([zero_block, 0.5 * n1k[:, np.newaxis]])
        M2_bottom = np.hstack([0.5 * n1k, 0])
        M2 = np.vstack([M2_top, M2_bottom])
        M2_matrices[key] = M2
    return M2_matrices


def create_M5_matrices(n1k_matrices, nT, nTs):
    M5_matrices = {}
    for key, n1k in n1k_matrices.items():
        zero_block = np.zeros((a + 1, a + 1))
        # Creating a diagonal matrix from n1k
        n1k_dash = np.diag(n1k)
        # Calculating the product [(nT - nTs)* n1k_dash]
        nT_nTs_product = 0.5 * np.dot((nT - nTs), n1k_dash)
        # Constructing M5 for the specific n1k
        M5_top = np.hstack(
            [zero_block, nT_nTs_product[:, np.newaxis]]
        )  # Ensuring it's a column vector
        M5_bottom = np.hstack(
            [nT_nTs_product, 0]
        )  
        M5 = np.vstack([M5_top, M5_bottom])
        M5_matrices[key] = M5
    return M5_matrices


def create_M3_dash_matrices(n3_matrices, nT, ec_matrices):
    M3_dash_matrices = {}
    for n3_key, n3 in n3_matrices.items():
        n3_dash = np.diag(n3)
        for ec_key, ec in ec_matrices.items():
            ec_column = 0.5 * ec
            # Constructing M3_dash
            nT_n3_dash_block = -0.5 * np.dot(nT, n3_dash)  
            M3_dash_top = np.hstack(
                [np.zeros((a, a)), nT_n3_dash_block[:-1][:, np.newaxis], ec_column[:, np.newaxis][:-1]]
            )
            M3_dash_middle = np.hstack(
                [
                    nT_n3_dash_block[np.newaxis, :][:, :-1],
                    np.array([[nT_n3_dash_block[-1]]]),
                    np.array([[0]]),
                ]
            )
            M3_dash_bottom = np.hstack(
                [0.5 * ec[np.newaxis, :][:, :-1], np.array([[0]]), np.array([[0]])]
            )
            M3_dash = np.vstack([M3_dash_top, M3_dash_middle, M3_dash_bottom])
            combined_key = f"{ec_key}_{n3_key}"
            M3_dash_matrices[combined_key] = M3_dash
    return M3_dash_matrices
