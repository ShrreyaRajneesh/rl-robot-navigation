import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_cnn_dce import CNN_DCEVAE   # reuse model

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/home/shrreya07/rl_robot/datasets/depth_vae_data/train"
MODEL_PATH = "cnn_dcevae.pt"

# =========================
# LOAD MODEL
# =========================
model = CNN_DCEVAE(latent_dim=6).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded")

# =========================
# HELPER: LOAD ONE SAMPLE
# =========================
def load_sample(path):
    img = np.load(path)

    # fix shape
    if img.shape == (64, 96):
        img = img.T

    img = np.clip(img, 0, 20.0) / 20.0

    # collision label (same logic as training)
    threshold = 0.1
    ratio = np.sum(img < threshold) / img.size
    collision = 1.0 if ratio > 0.02 else 0.0

    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

    return tensor.to(DEVICE), img, collision

# =========================
# TEST LOOP
# =========================
files = sorted(os.listdir(DATA_DIR))

for i in range(5):   # test 5 samples
    path = os.path.join(DATA_DIR, files[np.random.randint(len(files))])

    x, img_np, y_true = load_sample(path)

    with torch.no_grad():
        recon, mu, logvar, col_logit = model(x)

        pred = torch.sigmoid(col_logit).item()

    print(f"\nFile: {os.path.basename(path)}")
    print(f"GT Collision: {y_true}")
    print(f"Pred Collision: {pred:.3f}")

    # =========================
    # PLOT
    # =========================
    recon_np = recon.squeeze().cpu().numpy()

    plt.figure(figsize=(8,3))

    plt.subplot(1,2,1)
    plt.title("Input Depth")
    plt.imshow(img_np, cmap='gray')

    plt.subplot(1,2,2)
    plt.title("Reconstruction")
    plt.imshow(recon_np, cmap='gray')

    plt.show()