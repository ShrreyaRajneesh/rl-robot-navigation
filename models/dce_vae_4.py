# dce_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCEVAE(nn.Module):
    def __init__(self, input_dim=24, latent_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, 128)
        self.enc_fc2 = nn.Linear(128, 64)
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.col_head = nn.Linear(latent_dim, 1)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 64)
        self.dec_fc2 = nn.Linear(64, 128)
        self.dec_out = nn.Linear(128, input_dim)
        self.recon = nn.Linear(64, input_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))

        mu = self.mu(h)
        logvar = self.logvar(h)

        # sample z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # collision classifier using latent z
        col_logit = self.col_head(z)

        return z, mu, logvar, col_logit


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        out = torch.sigmoid(self.dec_out(h))  # normalized output
        return out

    def forward(self, x):
        z, mu, logvar, col_logit = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar, col_logit



# -------------------------
# LOSS FUNCTION
# -------------------------
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon_loss + kl_loss
    return loss, recon_loss, kl_loss
