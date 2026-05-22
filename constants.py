# Constants and Parameters Initialization
K, I, M = 5, 3, 2  # Number of tasks, helpers, and servers
B = 5e6  # Bandwidth
# b_k = np.random.uniform(200e3, 400e3, K)  # Data sizes
b_k = 350e3
# c_k = np.random.uniform(30, 50, K)  # Computational complexities
c_k = 80
P_tr_k, P_wait_k = 0.2, 0.05  # Transmit power, Waiting power
f_UE_k, f_UHi_k, f_Sm_k = 0.1e9, 0.5e9, 2e9  # Computing frequencies
e_UE, e_UHi = 1e-25, 0.8e-27  # Energy coefficients
sigma_squared = 1e-9  # Noise Power
H_k, H_k_Sm = 1e-7, 1e-8   # Channel gains
size = (1 + I + M + I * M) * K
L = 1000
a = (2 + I + M + I * M) * K
b = (1 + I + M + I * M) * K + 1
# Constants and Parameters Initialization
helper_communication_range = 50
server_communication_range = 400