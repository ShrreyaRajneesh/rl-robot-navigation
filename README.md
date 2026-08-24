# RL-Based Robot Navigation with DCE-VAE Latent Representations

A PyBullet-based mobile robot navigation project exploring **reinforcement learning for goal-directed navigation and obstacle avoidance**, together with **VAE-based latent representation learning** for compact sensory observations.

The project combines a simulated robot environment, depth/range sensing, DCE-VAE representation learning, and Soft Actor-Critic (SAC) policies to investigate how learned low-dimensional representations can be used for robot navigation.

---

## Project Overview

The objective is to train a mobile robot to navigate toward a target while avoiding obstacles using reinforcement learning.

The project investigates two connected problems:

1. **Representation Learning**
   Learning a compact latent representation of depth observations using a DCE-VAE.

2. **Robot Navigation with RL**
   Using the learned representation together with robot/goal state information as input to an RL navigation policy.

### Overall Pipeline

```text
              PyBullet Environment
                       │
                       ▼
              Depth / Range Sensors
                       │
                       ▼
                 DCE-VAE Encoder
                       │
                       ▼
                 Latent Space (z)
                       │
          ┌────────────┴────────────┐
          │                         │
   Goal / Robot State        Learned Representation
          │                         │
          └────────────┬────────────┘
                       ▼
                 SAC Policy
                       │
                       ▼
              Robot Velocity Action
                       │
                       ▼
             Navigation Environment
                       │
                       ▼
                    Reward
                       │
                       └──────► RL Training
```

---

## Simulation Environment

The robot is simulated in **PyBullet** and operates in environments containing obstacles and navigation targets.

The observation incorporates information describing:

* Robot state
* Relative position of the goal
* Goal direction
* Depth observations
* Range measurements / beam-based obstacle information

The environment provides continuous actions corresponding to robot motion, allowing the learned policy to control the robot while navigating through the environment.

---

## DCE-VAE Representation Learning

A major component of the project is learning a compact representation of the robot's sensory observations.

Instead of directly providing the complete depth representation to the RL policy, the observation can be passed through a VAE encoder:

```text
Depth Observation
       │
       ▼
   DCE-VAE Encoder
       │
       ▼
   Latent Distribution
       │
       ▼
      z / μ
       │
       ▼
Compact Environmental Representation
```

The VAE is trained to reconstruct the original depth observation while regularizing the latent distribution.

The objective follows the general VAE formulation:

[
\mathcal{L}
===========

\mathcal{L}*{reconstruction}
+
\beta \mathcal{L}*{KL}
]

where the reconstruction term encourages preservation of useful sensory information and the KL-divergence term regularizes the learned latent space.

---

## Latent-Space Experiments

Multiple latent dimensions were investigated to study the effect of representation size on the navigation problem.

Experiments include latent dimensions such as:

* 2
* 4
* 6
* 8
* 10

Additional experiments investigate variations of the VAE, including deterministic/no-Gaussian variants.

This allows comparison between:

```text
Higher-dimensional representation
            ↓
More information
            vs.
Lower-dimensional representation
            ↓
More compact state
```

The goal is to understand how much sensory information is required for effective navigation.

---

## Reinforcement Learning

The navigation policy is trained using **Soft Actor-Critic (SAC)**.

The RL observation combines navigation-relevant state information with the learned sensory representation:

[
s_t =
[
\text{robot state},
\text{goal state},
z_t
]
]

The policy outputs continuous motion commands for the robot.

The navigation objective is to:

* Reach the target
* Avoid collisions
* Maintain effective motion toward the goal
* Learn a robust navigation policy

Multiple SAC configurations were trained for different observation/latent representations.

---

## Experiments

The repository contains experiments corresponding to different latent-space configurations and navigation setups.

### Representation Experiments

| Experiment           | Latent Dimension | Representation                      |
| -------------------- | ---------------: | ----------------------------------- |
| DCE-VAE              |                2 | Compact latent representation       |
| DCE-VAE              |                4 | Low-dimensional representation      |
| DCE-VAE              |                6 | Intermediate representation         |
| DCE-VAE              |                8 | Higher-capacity representation      |
| DCE-VAE              |               10 | Higher-dimensional representation   |
| No-Gaussian variants |            6 / 8 | Deterministic latent representation |

### RL Experiments

Separate SAC training and evaluation scripts are provided for the different observation configurations.

The repository also contains TensorBoard logs from the training experiments.

---

## Demonstrations

### Robot Navigation

<!-- Add your navigation GIF/video here -->

![Robot Navigation](results/navigation.gif)

### Latent-Representation Navigation

<!-- Add your second demonstration here -->

![RL Navigation](results/latent_navigation.gif)

> Video demonstrations will be added to showcase the trained policies and navigation behaviour.

---

## Results

The repository contains scripts and experiment outputs for analysing:

* Training performance
* Navigation behaviour
* Depth reconstruction
* Latent representation quality
* Different latent dimensions
* RL policy performance

Example result plots can be placed here:

```text
results/
├── reconstruction/
├── training_curves/
├── latent_space/
└── navigation/
```

### Depth Reconstruction

```text
Input Depth                 Reconstruction

┌───────────────┐           ┌───────────────┐
│               │           │               │
│   DEPTH MAP   │    →      │  RECONSTRUCTED │
│               │           │    DEPTH MAP   │
└───────────────┘           └───────────────┘
```

The reconstruction experiments provide a way to evaluate whether the learned latent space preserves useful information from the original sensory input.

---

## Repository Structure

```text
rl-robot-navigation/
│
├── envs/
│   ├── cyl_depth_env.py
│   └── ...
│
├── models/
│   ├── dce_vae.py
│   ├── train_dce.py
│   ├── test_dce.py
│   ├── train_cnn_dce.py
│   ├── train_dce_no_gauss.py
│   └── trained model weights
│
├── scripts/
│   └── collect_dce_data.py
│
├── tartanair_tools/
│   └── TartanAir utilities
│
├── train_sac.py
├── train_sac_2.py
├── train_sac_4.py
├── train_sac_6.py
├── train_sac_8.py
├── train_sac_10.py
│
├── test_policy.py
├── test_policy2.py
├── test_policy4.py
├── test_policy6.py
├── test_policy8.py
├── test_policy10.py
│
├── viz_depth.py
├── plots.py
├── docker/
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone git@github.com:ShrreyaRajneesh/rl-robot-navigation.git
cd rl-robot-navigation
```

Create and activate the Python environment:

```bash
python3 -m venv rl_env
source rl_env/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

> `requirements.txt` will be added/updated according to the final environment used for the experiments.

---

## Running the Project

### Train the DCE-VAE

```bash
python models/train_dce.py
```

### Test the DCE-VAE

```bash
python models/test_dce.py
```

### Train the SAC policy

For example:

```bash
python train_sac.py
```

Different latent-dimensional experiments can be run using the corresponding training scripts:

```bash
python train_sac_2.py
python train_sac_4.py
python train_sac_6.py
python train_sac_8.py
python train_sac_10.py
```

### Evaluate a trained policy

```bash
python test_policy.py
```

---

## Key Research Questions

This project explores the following questions:

**1. Can sensory observations be compressed into a useful low-dimensional latent representation?**

**2. How does latent dimensionality affect the information retained by the representation?**

**3. Can a learned latent representation be effectively used by an RL navigation policy?**

**4. How does representation choice influence navigation performance and learning behaviour?**

**5. Can compact sensory representations reduce the dimensionality of the RL observation space without significantly degrading navigation?**

---

## 🛠️ Technologies

* **Python**
* **PyTorch**
* **PyBullet**
* **Stable-Baselines3 / SAC**
* **Variational Autoencoders**
* **Deep Learning**
* **Reinforcement Learning**
* **TensorBoard**
* **Docker**

---

## Project Highlights

* Developed a simulated mobile robot navigation environment in PyBullet.
* Implemented depth/range-based sensory observations for obstacle-aware navigation.
* Developed and evaluated DCE-VAE models for learning compact sensory representations.
* Investigated multiple latent dimensions and deterministic/no-Gaussian variants.
* Integrated learned latent representations into SAC-based navigation.
* Conducted comparative experiments across different representation configurations.
* Generated reconstruction, training and navigation analysis for evaluating the learned policies.

---

## Author

**Shrreya Rajneesh**
B.Tech — Aerospace Engineering

[GitHub](https://github.com/ShrreyaRajneesh)

---

## Note

This repository contains research and experimental code developed during the project. Some scripts correspond to intermediate experiments and different model configurations retained for comparison and reproducibility.
