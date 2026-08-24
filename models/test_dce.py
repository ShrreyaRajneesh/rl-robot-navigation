# models/test_dce.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dce_vae_8 import DCEVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# =====================================================
# LOAD DATASET
# =====================================================
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dce_dataset.npy"
)

print("Loading dataset:", DATA_PATH)
data = np.load(DATA_PATH).astype(np.float32)

N, D = data.shape
print(f"Dataset loaded: {N} samples, dim={D}")

# Split data
sensor_data = data[:, :-1]          # <-- only sensor vector
collision_labels = data[:, -1]      # <-- NOT used for VAE test

tensor_data = torch.tensor(sensor_data).to(device)  # shape (N, 24)

# =====================================================
# LOAD MODEL
# =====================================================
latent_dim = 8
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "dce_vae_latent8.pt"
)

print("Loading model:", MODEL_PATH)


model = DCEVAE(input_dim=30, latent_dim=latent_dim, use_gaussian_weighting=True).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✓ Model loaded")

# Pick ONE sample for detailed diagnostics
x_single = tensor_data[:1]

with torch.no_grad():
    recon_s, mu_s, logvar_s, _ = model(x_single)

weights = model.last_weights[0].numpy()
mu_g = model.last_mu[0].item()
sigma_g = model.last_sigma[0].item()

x_np_s = x_single.cpu().numpy()[0]
recon_np_s = recon_s.cpu().numpy()[0]

plt.figure(figsize=(8,4))
plt.plot(weights, marker='o')
plt.axvline(mu_g, color='r', linestyle='--', label=f"μ = {mu_g:.1f}")
plt.title(f"Gaussian Weights over 30D (σ = {sigma_g:.1f})")
plt.xlabel("Input dimension")
plt.ylabel("Weight")
plt.legend()
plt.grid()
plt.show()

per_dim_error = (x_np_s - recon_np_s) ** 2

plt.figure(figsize=(8,4))
plt.plot(per_dim_error, label="Reconstruction error")
plt.plot(weights / weights.max(), '--', label="Gaussian weight (normalized)")
plt.xlabel("Input dimension")
plt.ylabel("Value")
plt.title("Reconstruction Error vs Gaussian Weight")
plt.legend()
plt.grid()
plt.show()

# Latent sensitivity test
with torch.no_grad():
    z_base, _, _, _ = model.encode(x_single)

latent_sensitivity = []
eps = 0.05

for i in range(30):
    x_pert = x_single.clone()
    x_pert[0, i] += eps

    with torch.no_grad():
        z_pert, _, _, _ = model.encode(x_pert)

    dz = torch.norm(z_pert - z_base).item()
    latent_sensitivity.append(dz)

latent_sensitivity = np.array(latent_sensitivity)

plt.figure(figsize=(8,4))
plt.plot(latent_sensitivity, label="Latent sensitivity ||Δz||")
plt.plot(weights / weights.max(), '--', label="Gaussian weight (normalized)")
plt.xlabel("Input dimension")
plt.ylabel("Magnitude")
plt.title("Latent Sensitivity vs Gaussian Weight")
plt.legend()
plt.grid()
plt.show()

# =====================================================
# TEST: RECONSTRUCTION
# =====================================================
with torch.no_grad():
    x = tensor_data
    recon, mu, logvar, col_logit = model(x)

x_np = x.cpu().numpy()
recon_np = recon.cpu().numpy()
mu_np = mu.cpu().numpy()

errors = np.mean((x_np - recon_np)**2, axis=1)

print(f"Mean MSE = {errors.mean():.6f}")
print(f"Std  MSE = {errors.std():.6f}")
print(f"Min  MSE = {errors.min():.6f}")
print(f"Max  MSE = {errors.max():.6f}")

plt.figure(figsize=(6,4))
plt.hist(errors, bins=40)
plt.title("Reconstruction Error Distribution")
plt.show()

# =====================================================
# RECONSTRUCTION PLOTS
# =====================================================
def plot_recon(idx):
    plt.figure(figsize=(10,4))
    plt.plot(x_np[idx], 'o-', label="Original")
    plt.plot(recon_np[idx], 'x-', label="Reconstructed")
    plt.legend()
    plt.grid()
    plt.show()

plot_recon(10)
plot_recon(50)
plot_recon(200)

# =====================================================
# PCA VISUALIZATION
# =====================================================
print("Computing PCA...")
latent_2d = PCA(n_components=2).fit_transform(mu_np)

plt.figure(figsize=(6,6))
plt.scatter(latent_2d[:,0], latent_2d[:,1], s=3, alpha=0.5)
plt.title("Latent Space (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()

print("\n✓ DCE Test Completed")
