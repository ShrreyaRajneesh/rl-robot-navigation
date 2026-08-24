# models/train_dce.py
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dcevae6 import DCEVAE, vae_loss
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training DCE-VAE on:", device)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dce_dataset.npy"
)

print("Loading dataset:", DATA_PATH)
data = np.load(DATA_PATH).astype(np.float32)   # shape (N, 25)

sensor_vecs = data[:, :30]       # beams + depth row
collision_labels = data[:, 30]   # last element

print("Sensor vec shape:", sensor_vecs.shape)
print("Collision label shape:", collision_labels.shape)

# ---------------------------------------------------
# DATASET + LOADER
# ---------------------------------------------------
x_tensor = torch.tensor(sensor_vecs, dtype=torch.float32)
y_tensor = torch.tensor(collision_labels, dtype=torch.float32)

dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------
input_dim = sensor_vecs.shape[1]   # = 24
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=6)
args = parser.parse_args()

latent_dim = args.latent_dim
print("Training DCE with latent_dim =", latent_dim)
model = DCEVAE(input_dim=input_dim, latent_dim=latent_dim, use_gaussian_weighting = True).to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
bce = torch.nn.BCEWithLogitsLoss()

epochs = 40

# ---------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------
for ep in range(1, epochs + 1):
    total_loss = total_recon = total_kl = total_ce = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        recon, mu, logvar, col_logit = model(x)

        loss, recon_loss, kl_loss, ce_loss = vae_loss(
            model,
            recon,
            x,
            mu,
            logvar,
            col_logit,
            y,
            bce,
            ep
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_ce += ce_loss.item()

    print(
        f"Epoch {ep}/{epochs} | Loss={total_loss:.4f} | Recon={total_recon:.4f} | "
        f"KL={total_kl:.4f} | CE={total_ce:.4f}"
    )

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------
SAVE_PATH = os.path.join(
    os.path.dirname(__file__),
    f"dce_vae_latent{latent_dim}_new_10.pt"
)

torch.save(model.state_dict(), SAVE_PATH)
print("\n✓ Saved new DCE-VAE:", SAVE_PATH)
