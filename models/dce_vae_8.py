# dce_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCEVAE(nn.Module):
    def __init__(self, input_dim=30, latent_dim=8, use_gaussian_weighting=False):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_gaussian_weighting = use_gaussian_weighting
        self.sigma_min = 1.5
        self.sigma_max = 5.0

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
        
    def apply_gaussian_weighting(self, x):
        B, D = x.shape
        device = x.device

        mu = torch.full((B,), D//2, device=device).float()
        sigma = torch.empty(B, device=device).uniform_(self.sigma_min, self.sigma_max)

        idx = torch.arange(D, device=device).unsqueeze(0)

        weights = torch.exp(-0.5 * ((idx - mu.unsqueeze(1)) / sigma.unsqueeze(1)) ** 2)
        weights = weights / weights.max(dim=1, keepdim=True)[0]
        self.last_mu = mu.detach()
        self.last_sigma = sigma.detach()
        self.last_weights = weights.detach()
        if not hasattr(self, "_debug_done"):
            self._debug_done = True
            print("Gaussian μ:", mu[0].item())
            print("Gaussian σ:", sigma[0].item())
            print("Weights:", weights[0].cpu().numpy())
        return x * weights

    def encode(self, x):
        if self.use_gaussian_weighting:
            x = self.apply_gaussian_weighting(x)

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
# -------------------------
# LOSS FUNCTION
# -------------------------
def vae_loss(model, recon_x, x, mu, logvar, col_logit, y, bce, ep):
    
    # Reconstruction loss
    if model.use_gaussian_weighting:
        weights = model.last_weights.to(x.device)
        recon_loss = torch.mean(3.0*weights * (recon_x - x) ** 2)
    else:
        recon_loss = F.mse_loss(recon_x, x)

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Collision classification loss
    ce_loss = bce(col_logit, y.unsqueeze(1))

    # Total loss
    beta = min(1.0, ep / 10)   # KL warmup over first 10 epochs
    loss = 5.0 * recon_loss + beta * kl_loss + ce_loss

    return loss, recon_loss, kl_loss, ce_loss



