import cvxpy as cp
from constants import *
from matrix_construction import (compute_n1k, create_ec_matrices, compute_ej, compute_n3,
                                 create_matrices, create_M1_matrices, create_M2_matrices,
                                 create_M5_matrices, create_M3_dash_matrices)


def setup_sdp_problem(
    M0, M1_matrices, M2_matrices, M3, M3_dash_matrices, M4, M5_matrices, Tdmax, a, K
):
    # Define the SDP variable
    G = cp.Variable((a + 2, a + 2), PSD=True)
    # Objective
    objective = cp.Minimize(cp.trace(M0 @ G))
    # Constraints
    constraints = [G >> 0]
    # Add constraints for M1, M2, M3, M3_dash, M4, and M5 matrices
    for key in M1_matrices:
        constraints.append(cp.trace(M1_matrices[key] @ G) == 0)
    for key in M2_matrices:
        constraints.append(cp.trace(M2_matrices[key] @ G) == 1)
    for k in range(1, K + 1):
        if k == 1:  # No preceding tasks for the first task
            constraints.append(cp.trace(M3 @ G) == 0)
        else:
            for key in M3_dash_matrices:
                constraints.append(cp.trace(M3_dash_matrices[key] @ G) >= 0)
    constraints.append(cp.trace(M4 @ G) <= Tdmax)
    for key in M5_matrices:
        constraints.append(cp.trace(M5_matrices[key] @ G) <= 0)
    constraints += [
        G[a, a] == 1,
        G[a, a + 1] == 1,
        G[a + 1, a] == 1,
        G[a + 1, a + 1] == 1,
    ]
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=True)
    return G.value