# collect_dce_data.py
import sys
import os
os.environ["USE_DCE"] = "0"
import pybullet as p

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import numpy as np
import torch
from tqdm import tqdm
from envs.cyl_depth_env import CylDepthEnv


def normalize_beams(beams, max_range=5.0):
    return np.clip(beams / max_range, 0.0, 1.0)


def normalize_depth(depth_row, near=0.1, far=20.0):
    return np.clip((depth_row - near) / (far - near), 0.0, 1.0)


def collect_dataset(
    episodes=100,
    steps_per_episode=200,
    save_path="dce_dataset.npy"
):

    env = CylDepthEnv(render=False)
    dataset = []

    for ep in range(episodes):
        print(f"Episode {ep+1}/{episodes}")

        obs, _ = env.reset()
        
        for t in range(steps_per_episode):
            if t % 50 == 0:
                print(f"  Step {t}/{steps_per_episode}")

            # random continuous action
            action = env.action_space.sample()

            # step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # compute min distance to any obstacle
            

            

            # extract raw beams + depth
            beams = env._get_depth_scan()       
            assert beams.shape[0] == 14  # (8,)
            depth_row = env.get_1d_camera_depth(16)     # (16,)
            #depth_row = np.zeros(16, dtype=np.float32)

            # normalize
            beams_n = normalize_beams(beams)            # (8,)
            depth_n = normalize_depth(depth_row)        # (16,)

            pos, _ = p.getBasePositionAndOrientation(env.robot)
            robot_xy = np.array(pos[:2])

            min_obs_dist = 999
            for o in env.obstacles:
                op, _ = p.getBasePositionAndOrientation(o)
                d = np.linalg.norm(robot_xy - np.array(op[:2]))
                min_obs_dist = min(min_obs_dist, d)

                if min_obs_dist < 0.35:
                    collision = 1
                else:
                    collision = 0

            
            
            
            # concatenate into 24-d vector
            sensor_vec = np.concatenate([beams_n, depth_n])  # (30,)
            sample = np.concatenate([sensor_vec, [collision]])  # (31,)
            dataset.append(sample)


            if done:
                break

    dataset = np.array(dataset)
    np.save(save_path, dataset)
    print(f"\nSaved DCE dataset → {save_path}  shape={dataset.shape}")


if __name__ == "__main__":
    collect_dataset(
        episodes=300,              # increase if needed
        steps_per_episode=300,
        save_path="dce_dataset.npy"
    )
