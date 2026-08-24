# test.py

import os, sys

# -------------------------------------------
# ADD PROJECT ROOT TO PYTHONPATH
# -------------------------------------------
# test.py is located in: rl_robot/envs/test.py
# project root is:       rl_robot/
ROOT = os.path.dirname(os.path.dirname(__file__))  
sys.path.insert(0, ROOT)

# Now Python can find:
#   envs.cyl_depth_env
#   models.dce_vae
#   scripts.*
# -------------------------------------------

from envs.cyl_depth_env import CylDepthEnv
import numpy as np

env = CylDepthEnv(render=False)
obs, _ = env.reset()

print("\n=== OBSERVATION SHAPE CHECK ===")
print("obs.shape =", obs.shape)
print("env.observation_space =", env.observation_space.shape)

print("\n=== WHEEL JOINT CHECK ===")
print("wheel joints =", env.wheel_joints)

print("\n=== SINGLE STEP CHECK ===")
a = env.action_space.sample()
obs2, r, done, trunc, info = env.step(a)
print("step output:", obs2.shape, r, done, trunc)


def reset(self, seed=None, options=None):
        # ---- curriculum update (episode-level) ----
        if self.fixed_curriculum is not None:
            self.curriculum_level = self.fixed_curriculum

        if hasattr(self, "episode_success"):
            if self.episode_success:
                self.success_counter += 1
            else:
                self.success_counter = max(0, self.success_counter - 1)

            if self.success_counter >= 3:   # start small: 3 successes
                self.curriculum_level = min(self.curriculum_level + 1, 2)
                self.success_counter = 0
                print(f"[CURRICULUM] Advanced to level {self.curriculum_level}")

        # reset episode flag
        self.episode_success = False
        print(
            f"[RESET] level={self.curriculum_level}, "
            f"success_counter={self.success_counter}"
        )


        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Load plane
        #self.plane = p.loadURDF("plane.urdf")
        floor_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[20, 20, 0.1]   # 40m × 40m arena
        )
        floor_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[20, 20, 0.1],
            rgbaColor=[0.9, 0.9, 0.9, 1]
        )
        self.plane = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_col,
            baseVisualShapeIndex=floor_vis,
            basePosition=[0, 0, -0.1]
        )


        # Create obstacles
        self._create_obstacles()
        if not self.render:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        # 1) LOAD ROBOT (must come BEFORE wheel disabling)
        flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER
        self.robot = p.loadURDF("husky/husky.urdf", [0,0,0.25], flags=flags)


                # 2) Disable Husky wheel motors
                # --- after loading robot URDF (self.robot = p.loadURDF(...)) ---
        # find wheel joints and store indices & wheel params
        self.wheel_joints = []
        self.wheel_radius = 0.165   # Husky typical wheel radius (adjust if needed)
        self.wheel_separation = 0.34  # distance between left/right wheels (adjust)
        print("wheel_joints:", self.wheel_joints)
        for j in self.wheel_joints:
            print("joint", j, p.getJointInfo(self.robot, j)[1].decode())

        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            name = info[1].decode('utf-8').lower()
            # Husky wheel joint names may contain 'wheel' or 'front_left_wheel' etc.
            if 'wheel' in name:
                self.wheel_joints.append(j)
        print("Detected wheel_joints:", self.wheel_joints)
        for j in self.wheel_joints:
            print("joint", j, p.getJointInfo(self.robot, j)[1].decode())

        # ensure we have 4 wheels (Husky has 4)
        if len(self.wheel_joints) == 0:
            # fallback: assume joints 2..5 are wheels (only if you know URDF)
            fallback = list(range(2, 6))
            print("Warning: wheel_joints empty, using fallback indices:", fallback)
            self.wheel_joints = fallback

        # enable motors (velocity control) with nonzero max force
        self.wheel_max_force = 200.0
        for j in self.wheel_joints:
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=self.wheel_max_force
    )
        # 3) Set dynamics (base)
        p.changeDynamics(self.robot, -1,
                        lateralFriction=1.2,
                        spinningFriction=0.1,
                        rollingFriction=0.01,
                        restitution=0.0,
                        linearDamping=0.1,
                        angularDamping=0.1)

        # 4) Set dynamics for each link
        for i in range(p.getNumJoints(self.robot)):
            p.changeDynamics(self.robot, i,
                            lateralFriction=1.2,
                            spinningFriction=0.1,
                            rollingFriction=0.01,
                            restitution=0.0,
                            linearDamping=0.1,
                            angularDamping=0.1)

        # 5) Initialize internal variables
        self.step_count = 0
        #self.success_counter = 0
        self.total_path_length = 0.0
        self.prev_pos = np.array([0.0, 0.0])
        self.prev_yaw = 0.0

        self.current_lin_vel = np.array([0.0, 0.0])
        self.current_angular_vel = 0.0

        self.max_linear_acc = 2.0
        self.max_ang_acc = 4.0

        # 6) Sample goal
        params = self._curriculum_params()
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(params["goal_min"], params["goal_max"])
        self.goal = np.array([dist * np.cos(angle), dist * np.sin(angle)], dtype=np.float32)
        
        p.loadURDF("sphere2.urdf",
                [self.goal[0], self.goal[1], 0.1],
                globalScaling=0.2)

        self.prev_dist = np.linalg.norm(self.goal)

        obs = self._get_obs()
        return obs, {}
    
    
    
     def _compute_reward_done(self, x, y, yaw):
    
       
    # tuning params
        w_progress = 9.0
        w_vel_align = 1.5
        w_yaw = 0.5
        w_approach = 0.8
        obs_soft_thresh = 1.0
        obs_hard_thresh = 0.35
        w_obs_soft = 4.0
        w_obs_hard = 120.0
        near_bonus_dist = 1.8
        near_bonus_gain = 6.0
        success_dist = 0.5
        path_penalty_weight = [0.2, 0.4, 0.5][self.curriculum_level]  # start small, tune later


        pos = np.array([x, y])
        goal = np.array(self.goal)
        dist = np.linalg.norm(goal - pos)
        reward = 0.0
        done = False

        # progress reward (distance reduction)
        progress = (self.prev_dist - dist)
        progress_reward = w_progress * progress + 2

        # gaussian approach reward (smooth attractor)
        approach_reward = w_approach * math.exp(- (dist**2) / (2 * (0.6**2)))

        # heading alignment
        goal_dir = math.atan2(goal[1] - y, goal[0] - x)
        yaw_err = (goal_dir - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_reward = w_yaw * math.cos(yaw_err)

        # velocity alignment: project linear velocity onto direction to goal
        linear_vel, ang_vel = p.getBaseVelocity(self.robot)
        vel_vec = np.array([linear_vel[0], linear_vel[1]])
        # forward speed in robot frame
         # already computed and signed

        
        vel_toward_goal = 0.0
        if dist > 1e-6:
            vel_toward_goal = float(np.dot(vel_vec, (goal - pos) / dist))
        vel_align_reward = w_vel_align * vel_toward_goal
        v_forward = vel_toward_goal 
        # --- GOAL OCCLUSION CHECK ---
        ray_start = [x, y, 0.3]
        ray_end = [self.goal[0], self.goal[1], 0.3]

        
        hit_id, _, hit_frac, _, _ = p.rayTest(ray_start, ray_end)[0]

        goal_occluded = (hit_id != -1 and hit_frac < 0.98)




        # --- OCCLUSION-AWARE PENALTY ---
        if goal_occluded:
        # reward clearing the line of sight
            clearance = hit_frac - self.prev_hit_frac
            reward += 6.0 * clearance

            # penalize motion INTO the obstacle only
            if vel_toward_goal > 0:
                reward -= 2.0 * vel_toward_goal

       


       


        # obstacle distance: min distance to obstacles
        min_obs_dist = 1e6
        for o in self.obstacles:
            op, _ = p.getBasePositionAndOrientation(o)
            d = np.linalg.norm(pos - np.array(op[:2]))
            min_obs_dist = min(min_obs_dist, d)

        obs_penalty = 0.0
        reward = (
                progress_reward
                + approach_reward
                + yaw_reward
                + vel_align_reward
                + obs_penalty
            )
        # -------------------------------
        # Near-obstacle shaping (NON-terminal)
        # -------------------------------
        if obs_hard_thresh < min_obs_dist < obs_soft_thresh:
            proximity = (obs_soft_thresh - min_obs_dist) / (obs_soft_thresh - obs_hard_thresh)

            # repel from obstacle
            reward -= w_obs_soft * proximity
            
            yaw_rate = ang_vel[2]
            reward += 0.3 * proximity * abs(yaw_rate)


        # -------------------------------
        # Hard collision (TERMINAL)
        # -------------------------------
        if min_obs_dist <= obs_hard_thresh:
            reward -= w_obs_hard

            # collision-aware penalty
            col_prob = self.last_collision_prob
            reward -= 5.0 * col_prob

            self.prev_dist = dist
            return reward, True

       
        # near-goal bonus
        near_bonus = 0.0
        if dist < near_bonus_dist:
            near_bonus = near_bonus_gain * (near_bonus_dist - dist) / near_bonus_dist

        # assemble final reward
        reward = progress_reward + approach_reward + yaw_reward + vel_align_reward + obs_penalty + near_bonus
        reward -= 0.02  # small living penalty

        # anti-stuck: if moved little for many steps
        moved = np.linalg.norm(self.prev_pos - pos)
        if self.step_count > 20 and moved < 0.01:
            reward -= 0.5

        # discourage spinning in place
        if abs(ang_vel[2]) > 0.4 and abs(v_forward) < 0.05:
            reward -= 0.3

        self.prev_dist = dist
        self.prev_hit_frac = hit_frac


        # success
        if dist < success_dist:
            path_penalty = path_penalty_weight * self.total_path_length
            reward = reward + 100.0 - path_penalty
            return reward, True

        # timeout
        if self.step_count >= 1200:
            path_penalty = path_penalty_weight * self.total_path_length
            reward = reward - path_penalty - 10.0
            self.episode_success = False 
            return reward, True
        
        return reward, False