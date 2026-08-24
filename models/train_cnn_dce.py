import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# DEVICE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ==========================================
# DATASET
# ==========================================
class DepthDataset(Dataset):
    def __init__(self, folder):
        print("Loading from:", folder)

        self.files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".npy")   # ✅ FIX HERE
        ])

        print("Total files found:", len(self.files))

        if len(self.files) == 0:
            raise ValueError("No .npy files found!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        while True:  # keep trying until valid sample found
            try:
                path = self.files[idx]

                img = np.load(path, allow_pickle=True)

                # --------- HARD CHECKS ---------
                if img is None:
                    raise ValueError("Loaded None")

                if not isinstance(img, np.ndarray):
                    raise ValueError("Not numpy array")

                if img.size == 0:
                    raise ValueError("Empty array")

                # OPTIONAL shape check (VERY IMPORTANT)
                # Fix shape issues
                if img.shape == (64, 96):
                    img = img.T   # transpose to (96,64)

                elif img.shape != (96, 64):
                    raise ValueError(f"Unexpected shape {img.shape}")
                # --------- PROCESS ---------
                img = img.astype(np.float32)

                # normalize safely
                img = img.astype(np.float32)

                # normalize safely
                img = np.clip(img, 0, 20.0) / 20.0

                # 👉 compute collision BEFORE torch conversion
                threshold = 0.1

                # count how many pixels are "too close"
                close_pixels = np.sum(img < threshold)

                # ratio of close pixels
                ratio = close_pixels / img.size

                collision = 1.0 if ratio > 0.02 else 0.0   # 2% threshold

                img = torch.tensor(img).unsqueeze(0)
                y = torch.tensor([collision], dtype=torch.float32)
                #print("collision ratio:", ratio)
                return img, y

            except Exception as e:
                print(f"⚠️ Skipping bad file: {self.files[idx]} | {e}")

                # pick another random index
                idx = np.random.randint(0, len(self.files))

   

 

# ==========================================
# 2D GAUSSIAN WEIGHTING
# ==========================================
def gaussian_2d(H, W, sigma=15):
    y = torch.arange(H).float()
    x = torch.arange(W).float()

    yy, xx = torch.meshgrid(y, x, indexing='ij')

    cy, cx = H//2, W//2

    weights = torch.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*sigma**2))
    weights = weights / weights.max()

    return weights


# ==========================================
# MODEL
# ==========================================
class CNN_DCEVAE(nn.Module):
    def __init__(self, latent_dim=6, use_gaussian=True):
        super().__init__()

        self.use_gaussian = use_gaussian

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 96x64 → 48x32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 48x32 → 24x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 24x16 → 12x8
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64*12*8, latent_dim)
        self.fc_logvar = nn.Linear(64*12*8, latent_dim)

        # collision head
        self.col_head = nn.Linear(latent_dim, 1)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 64*12*8)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.use_gaussian:
            B, _, H, W = x.shape
            weights = gaussian_2d(H, W).to(x.device)
            x = x * weights

        h = self.enc(x)
        h = h.view(x.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        col_logit = self.col_head(z)

        h = self.fc_dec(z)
        h = h.view(x.size(0), 64, 12, 8)

        recon = self.dec(h)

        return recon, mu, logvar, col_logit
        #return recon, mu, logvar

# ==========================================
# LOSS
# ==========================================
def loss_fn(recon, x, mu, logvar, col_logit, y, ep):

    recon_loss = F.l1_loss(recon, x)

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    kl_loss = kl_loss / x.numel()
    ce_loss = F.binary_cross_entropy_with_logits(
    col_logit.view(-1),
    y.view(-1),
    reduction='mean'
)

    beta = min(1.0, ep / 10)

    loss = recon_loss + beta * kl_loss + 0.1 * ce_loss

    return loss, recon_loss, kl_loss, ce_loss


# ==========================================
# TRAIN
# ==========================================
BASE_DIR = "/home/shrreya07/rl_robot"
DATA_DIR = os.path.join(BASE_DIR, "datasets/depth_vae_data/train")

dataset = DepthDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = CNN_DCEVAE(latent_dim=6, use_gaussian=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 20

for ep in range(1, epochs+1):
    total_loss = total_recon = total_kl = total_ce = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        recon, mu, logvar, col_logit = model(x)
        #recon, mu, logvar = model(x)
        loss, r, k, c = loss_fn(
            recon, x, mu, logvar, col_logit, y, ep
        )
        #loss, r, k = loss_fn(recon, x, mu, logvar, ep)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += r.item()
        total_kl += k.item()
        total_ce += c.item()

    print(f"Epoch {ep} | Loss={total_loss:.3f} | Recon={total_recon:.3f} | KL={total_kl:.3f} | CE={total_ce:.3f}")


# ==========================================
# SAVE
# ==========================================
torch.save(model.state_dict(), "cnn_dcevae.pt")
print("Model saved")