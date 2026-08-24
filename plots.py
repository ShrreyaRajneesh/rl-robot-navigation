import numpy as np
import matplotlib.pyplot as plt

# Data
latent_dims = np.array([2, 4, 6, 8, 10])

success_rate = np.array([0.84, 0.73, 0.74, 0.64, 0.67])
mean_mse = np.array([56.585674, 58.478617, 61.673791, 51.395185, 55.178122])
std_mse = np.array([20.448122, 24.743159, 26.159463, 23.492559, 25.412161])
path_eff = np.array([0.506, 0.619, 0.897, 0.805, 1.073])

# ---- Plot 1: Success Rate ----
plt.figure()
plt.plot(latent_dims, success_rate, marker='o')
plt.xlabel("Latent Dimension")
plt.ylabel("Success Rate")
plt.title("Success Rate vs Latent Dimension")
plt.grid(True)
plt.show()

# ---- Plot 2: Mean MSE with Std ----
plt.figure()
plt.errorbar(latent_dims, mean_mse, yerr=std_mse, marker='o', capsize=5)
plt.xlabel("Latent Dimension")
plt.ylabel("Mean Position MSE")
plt.title("MSE vs Latent Dimension")
plt.grid(True)
plt.show()

# ---- Plot 3: Path Efficiency ----
plt.figure()
plt.plot(latent_dims, path_eff, marker='o')
plt.xlabel("Latent Dimension")
plt.ylabel("Path Efficiency")
plt.title("Path Efficiency vs Latent Dimension")
plt.grid(True)
plt.show()
