import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dcevae6 import DCEVAE
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load data
# ----------------------------
data = np.load("dce_dataset.npy").astype(np.float32)
x = torch.tensor(data[:, :-1]).to(device)  # (N, 30)

latent_dim = 6
# -------------------------------------------------
# Model paths
# -------------------------------------------------
MODEL_PATH_GAUSS = os.path.join(
    os.path.dirname(__file__),
    "dce_vae_latent6_new_10.pt"
)

MODEL_PATH_NO_GAUSS = os.path.join(
    os.path.dirname(__file__),
    "dce_vae_latent6_no_gauss_new_10.pt"
)

print("Loading Gaussian model:", MODEL_PATH_GAUSS)
print("Loading No-Gaussian model:", MODEL_PATH_NO_GAUSS)

# -------------------------------------------------
# Gaussian-trained model
# -------------------------------------------------
model_g = DCEVAE(
    input_dim=30,
    latent_dim=latent_dim,
    use_gaussian_weighting=True
).to(device)

state_gauss = torch.load(MODEL_PATH_GAUSS, map_location=device)
model_g.load_state_dict(state_gauss)
model_g.eval()

# -------------------------------------------------
# No-Gaussian-trained model
# -------------------------------------------------
model_no = DCEVAE(
    input_dim=30,
    latent_dim=latent_dim,
    use_gaussian_weighting=False
).to(device)

state_plain = torch.load(MODEL_PATH_NO_GAUSS, map_location=device)
model_no.load_state_dict(state_plain)
model_no.eval()



# ----------------------------
# Forward pass
# ----------------------------
with torch.no_grad():
    recon_no, mu_no, _, _ = model_no(x)
    recon_g,  mu_g,  _, _ = model_g(x)

xp = x.cpu().numpy()
r_no = recon_no.cpu().numpy()
r_g  = recon_g.cpu().numpy()
mu_no = mu_no.cpu().numpy()
mu_g  = mu_g.cpu().numpy()

# =====================================================
# PLOT 1 — RECONSTRUCTION ERROR PER DIMENSION
# =====================================================
err_no = np.mean((xp - r_no)**2, axis=0)
err_g  = np.mean((xp - r_g )**2, axis=0)
print("Max abs diff:", np.max(np.abs(err_no - err_g)))


plt.figure(figsize=(10,4))
plt.plot(err_no, 'o-', label="No Gaussian")
plt.plot(err_g,  'x--', label="Gaussian")


plt.title("Per-dimension Reconstruction Error")
plt.xlabel("Input dimension (0–29)")
plt.ylabel("MSE")
plt.legend()
plt.grid()
plt.show()

# =====================================================
# PLOT 2 — LATENT SENSITIVITY
# =====================================================
def latent_sensitivity(model, x, eps=1e-2):
    base_mu = model.encode(x)[1]
    sens = []

    for i in range(x.shape[1]):
        x_pert = x.clone()
        x_pert[:, i] += eps
        mu_pert = model.encode(x_pert)[1]
        dz = torch.norm(mu_pert - base_mu, dim=1).mean()
        sens.append(dz.item())
    return np.array(sens)

sens_no = latent_sensitivity(model_no, x[:512])
sens_g  = latent_sensitivity(model_g,  x[:512])

plt.figure(figsize=(10,4))
plt.plot(sens_no, label="No Gaussian", linewidth=2)
plt.plot(sens_g,  label="Gaussian-trained", linewidth=2)
plt.title("Latent Sensitivity vs Input Dimension")
plt.xlabel("Input dimension")
plt.ylabel("||Δz||")
plt.legend()
plt.grid()
plt.show()

# =====================================================
# PLOT 3 — PCA COMPARISON
# =====================================================
pca = PCA(n_components=2)
z_no = pca.fit_transform(mu_no)
z_g  = pca.fit_transform(mu_g)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(z_no[:,0], z_no[:,1], s=2, alpha=0.4)
plt.title("Latent PCA — No Gaussian")
plt.grid()

plt.subplot(1,2,2)
plt.scatter(z_g[:,0], z_g[:,1], s=2, alpha=0.4)
plt.title("Latent PCA — Gaussian-trained")
plt.grid()

plt.show()

plt.figure()
plt.plot(err_no - err_g)
plt.title("Difference: NoGaussian − Gaussian")
plt.grid()
plt.show()

# =====================================================
# PLOT 4 — GROUND TRUTH VS RECONSTRUCTION
# =====================================================
def plot_reconstruction(idx):
    plt.figure(figsize=(10,4))

    plt.plot(xp[idx], 'o-', label="Ground Truth", linewidth=2)
    plt.plot(r_no[idx], 'x--', label="No Gaussian", linewidth=2)
    plt.plot(r_g[idx],  's--', label="Gaussian", linewidth=2)

    plt.title(f"Reconstruction Comparison (Sample {idx})")
    plt.xlabel("Input dimension (beam index)")
    plt.ylabel("Depth value")
    plt.legend()
    plt.grid()
    plt.show()


# visualize a few examples
plot_reconstruction(10)
plot_reconstruction(200)
plot_reconstruction(1000)