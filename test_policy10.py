# test_policy.py
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import pybullet as p

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.cyl_depth_env_10 import CylDepthEnv

# Create a vectorized env with render=True so CylDepthEnv opens GUI itself.
env = DummyVecEnv([lambda: CylDepthEnv(render=True, fixed_curriculum=2)])
raw_env = env.envs[0]   # underlying env object


# Load trained model
model = SAC.load("sac_cyl_depth10")

# Reset the vectorized env (returns batch-like obs)
obs = env.reset()
print(obs.shape)


episode_metrics = []
NUM_EVAL_EPISODES = 100

# test_policy.py
def draw_obstacle(ax, body_id):
    # pose
    pos, orn = p.getBasePositionAndOrientation(body_id)
    x, y = pos[0], pos[1]
    yaw = p.getEulerFromQuaternion(orn)[2]

    # collision shape
    shape_data = p.getCollisionShapeData(body_id, -1)
    if not shape_data:
        return

    hx, hy, hz = shape_data[0][3]  # half extents

    width  = 2 * hx
    height = 2 * hy

    # classify obstacle type by aspect ratio
    aspect = max(width, height) / max(1e-3, min(width, height))

    if aspect > 3.0:
        # wall / occluder
        color = "purple"
        alpha = 0.9
        zorder = 3
    else:
        # cube
        color = "red"
        alpha = 0.8
        zorder = 2

    # Matplotlib rectangles rotate about lower-left corner,
    # so we shift to center before rotation
    rect = plt.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        angle=np.degrees(yaw),
        rotation_point="center",
        facecolor=color,
        edgecolor="black",
        linewidth=1.0,
        alpha=alpha,
        zorder=zorder
    )

    ax.add_patch(rect)


# get obstacle positions and goal once (will update if env resets)
def get_scene_info():
    obstacle_positions = []
    for o in raw_env.obstacles:
        op, _ = p.getBasePositionAndOrientation(o)
        obstacle_positions.append(op[:2])
    goal_pos = raw_env.goal.copy()
    return obstacle_positions, goal_pos

obstacle_positions, goal_pos = get_scene_info()

plt.ion()
fig, (ax_map, ax_depth) = plt.subplots(1, 2, figsize=(12, 6))
last_level = raw_env.curriculum_level


for ep in range(NUM_EVAL_EPISODES):
    goal_reached = False

    obs = env.reset()
    obstacle_positions, goal_pos = get_scene_info()

    done = False

    # -------------------------------
    # Episode buffers
    # -------------------------------
    robot_path = []
    pos_errors = []
    vel_history = []

    step = 0

    while not done:
        step += 1

        # -------------------------------
        # Policy action
        # -------------------------------
        action, _ = model.predict(obs, deterministic=True)

        obs, rewards, dones, infos = env.step(action)
        done = bool(dones[0])

        # -------------------------------
        # State tracking
        # -------------------------------
        pos_bullet, _ = p.getBasePositionAndOrientation(raw_env.robot)
        robot_xy = np.array(pos_bullet[:2])
        pos_error = np.linalg.norm(raw_env.goal - robot_xy)
        SUCCESS_DIST = 0.6
        if pos_error < SUCCESS_DIST:
            print(f"[LATCH] Goal reached at step {step}, dist={pos_error:.3f}")

            goal_reached = True
      
        
        
        
        
        robot_path.append(robot_xy)

        #pos_error = np.linalg.norm(raw_env.goal - robot_xy)
        pos_errors.append(pos_error)

        linear_vel, ang_vel = p.getBaseVelocity(raw_env.robot)
        vel_history.append(linear_vel[0])

        # -------------------------------
        # Visualization (MAP)
        # -------------------------------
        ax_map.clear()

        for body_id in raw_env.obstacles:
            draw_obstacle(ax_map, body_id)

        if len(robot_path) > 1:
            rp = np.array(robot_path)
            ax_map.plot(rp[:, 0], rp[:, 1], 'b-')

        ax_map.plot(robot_xy[0], robot_xy[1], 'bo')
        ax_map.plot(goal_pos[0], goal_pos[1], 'go', markersize=10)

        # actual velocity (green)
        vx, vy = linear_vel[0], linear_vel[1]
        ax_map.quiver(
            robot_xy[0], robot_xy[1],
            vx, vy,
            color='green',
            scale=5.0,
            width=0.01,
            alpha=0.9
        )
        # -------------------------------
# DEBUG PRINTS (throttled)
# -------------------------------
        if step % 20 == 0:   # print every 20 steps (important!)
            print(
                f"[Ep {ep+1} | Step {step}] "
                f"Action={action} | "
                f"vx={linear_vel[0]:.2f}, vy={linear_vel[1]:.2f} | "
                f"YawRate={ang_vel[2]:.2f} | "
                f"DistToGoal={pos_error:.2f}"
            )

        # goal direction (black)
        goal_vec = raw_env.goal - robot_xy
        goal_vec /= (np.linalg.norm(goal_vec) + 1e-6)

        ax_map.quiver(
            robot_xy[0], robot_xy[1],
            goal_vec[0], goal_vec[1],
            color='black',
            scale=5.0,
            width=0.01,
            alpha=0.6
        )

        params = raw_env._curriculum_params()
        limit = params["arena_half"]
        ax_map.set_xlim(-limit, limit)
        ax_map.set_ylim(-limit, limit)
        ax_map.set_aspect('equal')
        ax_map.set_title(f"Episode {ep+1}, Step {step}")

        # -------------------------------
        # Visualization (DEPTH)
        # -------------------------------
        ax_depth.clear()

        beams = raw_env._get_depth_scan(fov=raw_env.cam_fov)

        _, orn = p.getBasePositionAndOrientation(raw_env.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]

        angles = np.linspace(
            -math.radians(raw_env.cam_fov) / 2,
            math.radians(raw_env.cam_fov) / 2,
            raw_env.num_beams
        )

        for d, a in zip(beams, angles):
            theta = yaw + a
            xb = d * math.cos(theta)
            yb = d * math.sin(theta)
            ax_depth.plot([0, xb], [0, yb], color='cyan', linewidth=2)

        cam = raw_env.get_1d_camera_depth(width=raw_env.cam_depth_len)

        angles_cam = np.linspace(
            -math.radians(raw_env.cam_fov) / 2,
            math.radians(raw_env.cam_fov) / 2,
            raw_env.cam_depth_len
        )

        for d, a in zip(cam, angles_cam):
            theta = yaw + a
            xc = d * math.cos(theta)
            yc = d * math.sin(theta)
            ax_depth.plot([0, xc], [0, yc], color='orange', alpha=0.7)

        ax_depth.set_xlim(-10, 10)
        ax_depth.set_ylim(-10, 10)
        ax_depth.set_aspect('equal')
        ax_depth.grid()
        ax_depth.set_title("Depth sensors")

        plt.pause(0.001)

    # =================================================
    # Episode finished → compute metrics
    # =================================================
    pos_errors = np.array(pos_errors)
    vel_history = np.array(vel_history)
    robot_path = np.array(robot_path)

    mse_pos = np.mean(pos_errors ** 2)
    final_error = pos_errors[-1]

    if len(vel_history) > 1:
        mse_vel = np.mean(np.diff(vel_history) ** 2)
    else:
        mse_vel = 0.0

    if len(robot_path) > 1:
        path_length = np.sum(
            np.linalg.norm(np.diff(robot_path, axis=0), axis=1)
        )
        straight_dist = np.linalg.norm(robot_path[0] - raw_env.goal)
        efficiency = straight_dist / (path_length + 1e-6)
    else:
        efficiency = 0.0

   
    success = goal_reached

    episode_metrics.append({
        "success": success,
        "mse_pos": mse_pos,
        "mse_vel": mse_vel,
        "final_error": final_error,
        "efficiency": efficiency
    })

    print(
        f"[Episode {ep+1}] "
        f"Success={success} | "
        f"MSE={mse_pos:.3f} | "
        f"FinalErr={final_error:.3f} | "
        f"Eff={efficiency:.3f}"
    )

# =====================================================
# Aggregate evaluation
# =====================================================
success_rate = np.mean([m["success"] for m in episode_metrics])
mse_vals = np.array([m["mse_pos"] for m in episode_metrics])
eff_vals = np.array([m["efficiency"] for m in episode_metrics])

print("\n===== OVERALL EVALUATION =====")
print(f"Success rate        : {success_rate:.2f}")
print(f"Mean  Position MSE  : {np.mean(mse_vals):.6f}")
print(f"Std   Position MSE  : {np.std(mse_vals):.6f}")
print(f"Min   Position MSE  : {np.min(mse_vals):.6f}")
print(f"Max   Position MSE  : {np.max(mse_vals):.6f}")
print(f"Avg Path Efficiency: {np.mean(eff_vals):.3f}")
print("=============================\n")

env.close()
plt.ioff()

