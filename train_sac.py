# train_sac.py
# train_sac.py
import gymnasium as gym
from stable_baselines3 import SAC
#from stable_baselines3.common.vec_env import DummyVecEnv
from envs.cyl_depth_env import CylDepthEnv
from stable_baselines3.common.env_util import make_vec_env 
import torch

# Create the environment
def make_env():
    return CylDepthEnv(render=False)

#env = DummyVecEnv([make_env])  # wrap in DummyVecEnv for SB3 compatibility
env = make_vec_env(lambda: CylDepthEnv(render=False), n_envs=1)

print("CUDA available:", torch.cuda.is_available())
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")


# Create the model
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    learning_starts=30000,
    policy_kwargs=dict(
        net_arch=[256, 256]       
    ),
    verbose=1,
    tensorboard_log="./sac_cyl_depth_tb/",
    device="cuda" if torch.cuda.is_available() else "cpu"

)

# Train
model.learn(total_timesteps=1000000)

# Save the model
model.save("sac_cyl_depth")
env.close()
