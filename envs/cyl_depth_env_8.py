import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math
import torch
from models.dce_vae_8 import DCEVAE
import os

class CylDepthEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self, render=False, fixed_curriculum = None):
        super().__init__()
        self.fixed_curriculum = fixed_curriculum
        self.render = render
        self.curriculum_level = 0
        self.success_counter = 0
        self.episode_success = True
        self.latent_dim = int(os.environ.get("DCE_LATENT_DIM", 8))

        if self.render:
            self.physicsClient = p.connect(p.GUI)
            # enable GUI & rendering
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setPhysicsEngineParameter(enableConeFriction=0)
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        # simulation timestep: controller runs at control_dt, physics uses smaller substep for stability
        self.control_dt = 1.0 / 30.0   # policy step interval (30 Hz)
        self.sim_substeps = 4          # physics substeps per control step
        self.dt = self.control_dt / self.sim_substeps
        p.setTimeStep(self.dt)


        # Observation: [x, y, v, yaw_rate] + depth beams
        #robot properties
        self.max_speed = 1.0         # m/s (scale action[0] to this)
        self.max_yaw_rate = 2.5  
        
        #sensor properties
        self.num_beams = 14
        self.max_beam_range = 8.0

        self.cam_near = 0.1
        self.cam_far = 30.0
        self.cam_fov = 60  # deg
        self.cam_depth_len = 16  # width of camera 1D slice

                # ================================
        # Load DCE-VAE (latent collision encoder)
        # ================================
        self.use_dce = os.environ.get("USE_DCE", "1") == "1"                   # switchable
        self.latent_dim = 8                    # must match trained model
        self.dce_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.use_dce:
            vae_path = f"models/dce_vae_latent{self.latent_dim}.pt"

            print(f"[DCE] Loading VAE from {vae_path}")

            self.dce_vae = DCEVAE(
                input_dim=self.num_beams + self.cam_depth_len,
                latent_dim=self.latent_dim,
                use_gaussian_weighting=True
            ).to(self.dce_device)

            self.dce_vae.load_state_dict(torch.load(vae_path, map_location=self.dce_device))
            self.dce_vae.eval()

        else:
            print("[DCE] Disabled (raw sensor data only)")
            self.dce_vae = None 
        
         
        # Build one example obs (guarantees shape match)
        # Note: use zeros / safe defaults that match exactly the _get_obs ordering below.
        if self.use_dce:
            latent_len = self.latent_dim
        else:
            # if no DCE we will fall back to raw sensor length (num_beams + cam_depth_len)
            latent_len = self.num_beams + self.cam_depth_len

        example_obs = np.concatenate([
            [0.0, 0.0],                       # dx, dy
            [0.0],                            # dist_to_goal
            [0.0, 0.0],                            # goal_angle_local
            [0.0, 0.0],                       # v_signed, yaw_rate
            np.zeros(latent_len, dtype=np.float32)  # latent or raw sensors
        ]).astype(np.float32)

        obs_dim = example_obs.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)



        # Action: [v, yaw_rate]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32),
                                       dtype=np.float32)

        self.reset()

    def _remove_obstacles(self):
            if not hasattr(self, "obstacles"):
                self.obstacles = []
                return

            for o in self.obstacles:
                if p.isConnected():
                    try:
                        p.removeBody(o)
                    except:
                        pass

            self.obstacles = []
            
    def _create_obstacles(self):
        params = self._curriculum_params()
        for o in self.obstacles:
            try:
                p.removeBody(o)
            except:
                pass
        self.obstacles = []
        arena = params["arena_half"]
        total_obs = params["num_obs"]
       
        if self.curriculum_level == 0:
                num_line_obs = 2
        elif self.curriculum_level == 1:
            self._add_curriculum1_occlusion()
            num_line_obs = params["num_obs"] // 3
        else:
            num_line_obs = params["num_obs"] // 2
        num_box_obs = total_obs - num_line_obs
        half_size = [0.15, 0.15, 0.25] # half extents
        line_length = 2.0
        line_thickness = 0.2
        line_height = 0.2
        line_half = [line_length / 2, line_thickness / 2, line_height / 2]
        def sample_xy():
            return np.random.uniform(-arena + 1.0, arena - 1.0)

        def valid_pose(x, y):
            if np.linalg.norm([x, y]) < 2.0:
                return False
            if np.linalg.norm([x - self.goal[0], y - self.goal[1]]) < 2.0:
                return False
            return True
        
   

        
        for _ in range(num_line_obs):
            for _ in range(50):  # retry
                x, y = sample_xy(), sample_xy()
                if not valid_pose(x, y):
                    continue

                yaw = np.random.uniform(0, np.pi)

                col = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=line_half
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=line_half,
                    rgbaColor=[0.7, 0.1, 0.1, 1]
                )

                wall = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[x, y, line_half[2]],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, yaw])
                )

                self.obstacles.append(wall)
                break

        for _ in range(params["num_obs"]):  # number of obstacles
            x = np.random.uniform(-params["arena_half"], params["arena_half"])
            y = np.random.uniform(-params["arena_half"], params["arena_half"])

            yaw = np.random.choice([0.0, np.pi / 2])
            quat = p.getQuaternionFromEuler([0, 0, yaw])
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size)
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_size, rgbaColor=[1, 0, 0, 1]
            )
            wall = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, line_height / 2],
                baseOrientation=quat
            )
            self.obstacles.append(wall)


    def _get_depth_scan(self, fov = 180):
        
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]

        half_fov = math.radians(fov) / 2
        angles = np.linspace(-half_fov, half_fov, self.num_beams)


        ray_from = []
        ray_to = []

        for a in angles:
            dx = np.cos(yaw + a)
            dy = np.sin(yaw + a)
            ray_from.append([pos[0], pos[1], pos[2] + 0.2])
            ray_to.append([
                pos[0] + self.max_beam_range * math.cos(yaw + a),
                pos[1] + self.max_beam_range * math.sin(yaw + a),
                pos[2] + 0.2
            ])

        results = p.rayTestBatch(ray_from, ray_to)
        dists = []
        for r in results:
            frac = r[2]
            d = frac * self.max_beam_range
            dists.append(float(d))

        return np.array(dists, dtype=np.float32)

    def _convert_depth_buffer(self, depth_buffer):
        return self.cam_near + depth_buffer * (self.cam_far - self.cam_near)
    def get_camera_images(self, width=64, height=64):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]
        robot_pos = np.array(pos)
        eye = robot_pos + np.array([0, 0, 0.5])
        pitch_down = -0.2
        target = eye + np.array([
            math.cos(yaw),
            math.sin(yaw),
            pitch_down
        ])
        up = [0, 0, 1]

        view = p.computeViewMatrix(eye, target, up)
        proj = p.computeProjectionMatrixFOV(self.cam_fov, width / height,
                                            self.cam_near, self.cam_far)

        img = p.getCameraImage(width, height, view, proj)
        rgb = np.reshape(img[2], (height, width, 4))[:, :, :3]
        depth_raw = np.reshape(img[3], (height, width))
        depth_m = self._convert_depth_buffer(depth_raw)
        seg = np.reshape(img[4], (height, width))
        return rgb, depth_m.astype(np.float32), seg

    def get_1d_camera_depth(self, width=16):
        FULL_W, FULL_H = 64, 64  # render high-res
        rgb, depth_m, seg = self.get_camera_images(FULL_W, FULL_H)

        row = FULL_H // 2
        depth_row = depth_m[row, :]
        depth_resized = np.interp(
            np.linspace(0, FULL_W - 1, width),
            np.arange(FULL_W),
            depth_row
        )

        return depth_resized.astype(np.float32)


    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        x, y, _ = pos
        yaw = p.getEulerFromQuaternion(orn)[2]

        # velocity from physics directly
        linear_vel, angular_vel = p.getBaseVelocity(self.robot)  # world frame
        vx_world, vy_world = linear_vel[0], linear_vel[1]

        # robot yaw (already computed earlier)
        c = math.cos(yaw); s = math.sin(yaw)
        # forward component in robot frame (signed)
        vx_local =  c*vx_world + s*vy_world
        v_signed = float(vx_local)  # can be negative (backwards)  # speed magnitude
        yaw_rate = angular_vel[2]

        # Goal in robot frame
        gx, gy = self.goal
        dx = gx - x
        dy = gy - y
        # distance + bearing to goal
        dist_to_goal = np.linalg.norm(self.goal - np.array([x, y]))

        goal_angle = math.atan2(dy, dx)
        goal_angle_local = goal_angle - yaw
        goal_angle_local = math.atan2(math.sin(goal_angle_local),
                                    math.cos(goal_angle_local))  # normalize

        

        # sensors
        beams = self._get_depth_scan(fov=self.cam_fov)          # (8,)
        depth_row = self.get_1d_camera_depth(self.cam_depth_len) # (16,)

        # normalize (same as dataset collection)
        beams_n = np.clip(beams / self.max_beam_range, 0.0, 1.0)
        depth_n = np.clip((depth_row - self.cam_near) /
                        (self.cam_far - self.cam_near), 0.0, 1.0)

        sensor_vec = np.concatenate([beams_n, depth_n]).astype(np.float32)   # (24,)

        # === DCE-VAE latent encoding ===
       # === DCE-VAE latent encoding ===
        if self.use_dce and self.dce_vae is not None:
            # convert to tensor with batch dim
            x_t = torch.tensor(sensor_vec, dtype=torch.float32, device=self.dce_device).unsqueeze(0)  # (1, D)
            with torch.no_grad():
                # call encode with the batched input if your model expects batch
                # assume encode returns (z, mu, logvar) OR (z, mu, logvar, col_logit) depending on model
                out = self.dce_vae.encode(x_t)   # keep flexible
                # normalize possibilities:
                if isinstance(out, tuple) or isinstance(out, list):
                    # common: z, mu, logvar, maybe col_logit
                    if len(out) == 4:
                        z_b, mu_b, logvar_b, col_logit_b = out
                        col_prob = torch.sigmoid(col_logit_b).cpu().numpy().squeeze().astype(float)
                    else:
                        z_b, mu_b, logvar_b = out
                        col_prob = 0.0
                    z = z_b
                else:
                    # if encode returns tensor z only (unlikely), fallback
                    z = out
                    col_prob = 0.0

            latent = z.squeeze(0).cpu().numpy()   # (latent_dim,)
            # store last collision probability for reward use
            self.last_collision_prob = float(col_prob)
        else:
            latent = sensor_vec  # fallback (raw sensors)
            self.last_collision_prob = 0.0

        self._last_dist_to_goal = dist_to_goal
        self._last_goal_angle = goal_angle_local
        self._last_min_beam = float(np.min(beams))

        sin_goal = math.sin(goal_angle_local)
        cos_goal = math.cos(goal_angle_local)
        # BUILD OBSERVATION: dx,dy, dist, angle, v, yaw_rate, latent
        obs = np.concatenate([
            [dx, dy],                 # 2
            [dist_to_goal],           # 1
            [sin_goal, cos_goal],       # 1
            [v_signed, yaw_rate],            # 2
            latent                    # latent_dim (e.g. 8)
        ]).astype(np.float32)

        return obs.astype(np.float32)


    def robot_pos(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        return pos[0], pos[1]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ===============================
        # Curriculum update
        # ===============================
        if self.fixed_curriculum is not None:
            self.curriculum_level = self.fixed_curriculum

        if hasattr(self, "episode_success"):
            if self.episode_success:
                self.success_counter += 1
            else:
                self.success_counter = max(0, self.success_counter - 1)

            if self.success_counter >= 3:
                self.curriculum_level = min(self.curriculum_level + 1, 2)
                self.success_counter = 0
                print(f"[CURRICULUM] Advanced to level {self.curriculum_level}")

        self.episode_success = False
        print(f"[RESET] level={self.curriculum_level}, success_counter={self.success_counter}")

        # ===============================
        # Physics reset
        # ===============================
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        params = self._curriculum_params()
        arena_half = params["arena_half"]

        # ===============================
        # Floor (curriculum-sized)
        # ===============================
        floor_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[arena_half, arena_half, 0.1]
        )
        floor_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[arena_half, arena_half, 0.1],
            rgbaColor=[0.9, 0.9, 0.9, 1]
        )
        self.plane = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_col,
            baseVisualShapeIndex=floor_vis,
            basePosition=[0, 0, -0.1]
        )

        # ===============================
        # Robot
        # ===============================
        flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER
        self.robot = p.loadURDF("husky/husky.urdf", [0, 0, 0.25], flags=flags)

        # Wheels
        self.wheel_joints = []
        for j in range(p.getNumJoints(self.robot)):
            name = p.getJointInfo(self.robot, j)[1].decode().lower()
            if "wheel" in name:
                self.wheel_joints.append(j)

        if len(self.wheel_joints) == 0:
            self.wheel_joints = list(range(2, 6))

        self.wheel_radius = 0.165
        self.wheel_separation = 0.34
        self.wheel_max_force = 200.0

        for j in self.wheel_joints:
            p.setJointMotorControl2(
                self.robot, j, p.VELOCITY_CONTROL,
                targetVelocity=0.0, force=self.wheel_max_force
            )

        # ===============================
        # Sample GOAL (BEFORE obstacles)
        # ===============================
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(params["goal_min"], params["goal_max"])
        self.goal = np.array([dist * np.cos(angle), dist * np.sin(angle)], dtype=np.float32)

        p.loadURDF(
            "sphere2.urdf",
            [self.goal[0], self.goal[1], 0.1],
            globalScaling=0.2
        )

        # ===============================
        # Obstacles (goal now exists!)
        # ===============================
        self._remove_obstacles()
        self._create_obstacles()

        # ===============================
        # State tracking
        # ===============================
        self.step_count = 0
        

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        self.prev_pos = np.array(pos[:2])
        self.prev_yaw = p.getEulerFromQuaternion(orn)[2]
        self.prev_dist = np.linalg.norm(self.prev_pos - self.goal)
        self.prev_hit_frac = 1.0   # assume fully occluded initially

        return self._get_obs(), {}






  
    
    
    def step(self, action):
        action = np.array(action).flatten()
        # desired commands
        # ---------------------------------
# ERROR-BASED ACTION FORMULATION
# ---------------------------------
        # current pose and yaw
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]
# current robot control (robot frame)
        linear_vel, ang_vel = p.getBaseVelocity(self.robot)
        v_current = (
            linear_vel[0] * math.cos(yaw) +
            linear_vel[1] * math.sin(yaw)
        )
        omega_current = ang_vel[2]
        

        # desired control from GOAL ERROR (nominal controller)
        k_v = 0.8
        k_w = 1.5
        v_des = k_v * self._last_dist_to_goal
        omega_des = k_w * self._last_goal_angle
        v_nom = np.clip(v_des, -self.max_speed, self.max_speed)
        omega_nom = np.clip(omega_des, -self.max_yaw_rate, self.max_yaw_rate)
        # policy outputs CONTROL ERROR
        delta_v = action[0] * 0.5        # m/s correction
        delta_omega = action[1] * 1.0    # rad/s correction

        # final control (apply error)
        v_body = np.clip(
            v_nom + delta_v,
            -self.max_speed,
            self.max_speed
        )
        yaw_rate_des = np.clip(
            omega_nom + delta_omega,
            -self.max_yaw_rate,
            self.max_yaw_rate
        )
        if self._last_dist_to_goal > 1.5:
            v_body = max(v_body, 0.15)

        #print(f"v_current={v_current:.2f}, delta_v={delta_v:.2f}, v_cmd={v_body:.2f}")

        
        # Slow down automatically if close to goal
        if np.linalg.norm(np.array([pos[0], pos[1]]) - self.goal) < 1.0:
            v_body *= 0.3      # reduce speed to 30%

                # convert commanded linear speed v_body and yaw_rate_des into wheel angular velocities
        # assume robot forward speed v_body (m/s) and yaw_rate_des (rad/s)
        v = v_body
        omega = yaw_rate_des

        # wheel kinematics for differential drive:
        # v = (r/2) * (omega_r + omega_l)
        # omega = (r / L) * (omega_r - omega_l)
        r = self.wheel_radius
        L = self.wheel_separation

        # compute wheel angular velocities (rad/s)
        # solve:
        # omega_r = (v/r) + (omega * L) / (2*r)
        # omega_l = (v/r) - (omega * L) / (2*r)
        omega_r = (v / r) + (omega * L) / (2.0 * r)
        omega_l = (v / r) - (omega * L) / (2.0 * r)

        # set motor targets for each wheel — Husky has 4 wheels, map left/right
        # Map depends on order; common mapping: [front_left, front_right, rear_left, rear_right]
        # Find left and right indices from wheel_joints list by name if needed; simplest: pair them
        if len(self.wheel_joints) >= 4:
            # try to sort by joint name to assign left/right properly (defensive)
            joints_info = [(j, p.getJointInfo(self.robot, j)[1].decode()) for j in self.wheel_joints]
            # crude mapping: assume joints_info order: [fl, fr, rl, rr] (verify with print once)
            j_fl, j_fr, j_rl, j_rr = self.wheel_joints[0], self.wheel_joints[1], self.wheel_joints[2], self.wheel_joints[3]
            # set velocities (targetVelocity expects rad/s)
            p.setJointMotorControl2(self.robot, j_fl, p.VELOCITY_CONTROL, targetVelocity=omega_l, force=self.wheel_max_force)
            p.setJointMotorControl2(self.robot, j_rl, p.VELOCITY_CONTROL, targetVelocity=omega_l, force=self.wheel_max_force)
            p.setJointMotorControl2(self.robot, j_fr, p.VELOCITY_CONTROL, targetVelocity=omega_r, force=self.wheel_max_force)
            p.setJointMotorControl2(self.robot, j_rr, p.VELOCITY_CONTROL, targetVelocity=omega_r, force=self.wheel_max_force)
        else:
            # if number of wheels unknown, set all wheels to average:
            for j in self.wheel_joints:
                p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL,
                                        targetVelocity=(omega_l + omega_r)/2.0, force=self.wheel_max_force)

        # step physics
        for _ in range(self.sim_substeps):
            p.stepSimulation()

        # ---------------------------------
        # Prevent robot from falling over
        # ---------------------------------
        
        '''upright_quat = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot,
                                  [pos[0], pos[1], pos[2]],
                                  upright_quat)'''
        # ---------------------------------
        # PATH LENGTH ACCUMULATION
        # ---------------------------------
        current_pos = np.array([pos[0], pos[1]])
        step_dist = np.linalg.norm(current_pos - self.prev_pos)
        
        #print(f"Path length: {self.total_path_length:.2f}")


        # afterwards handle rest: reward, obs, done as before
        # compute reward & termination
        reward, done = self._compute_reward_done(pos[0], pos[1], yaw)
        terminated = bool(done)
        truncated = False
        # observation
        obs = self._get_obs()
        
        # update prev_pos / prev_yaw AFTER observation and stepping physics
        self.prev_pos = current_pos
        self.prev_yaw = yaw

        self.step_count += 1
        info = getattr(self, "debug_info", {})


        
        goal_dist = np.linalg.norm(current_pos - self.goal)
        if done and goal_dist < 0.5:
            self.success_counter += 1
        else:
            self.success_counter = max(0, self.success_counter - 1)

        if self.success_counter > 20:
            self.curriculum_level = min(self.curriculum_level + 1, 2)
            self.success_counter = 0

        # SB3-compatible return: obs, reward, done, info
        return obs, reward, terminated, truncated, info


   
    
    def _compute_reward_done(self, x, y, yaw):

        # -----------------------
        # parameters
        # -----------------------
        w_progress = 4.0
        w_yaw = 0.5
        w_vel = 1.2
        w_clear = 6.0
        w_obs_soft = 4.0
        w_obs_hard = 120.0

        obs_soft = 1.0
        obs_hard = 0.35
        success_dist = 0.5

        # -----------------------
        # state
        # -----------------------
        pos = np.array([x, y])
        goal = np.array(self.goal)
        dist = np.linalg.norm(goal - pos)

        reward = 0.0

        # -----------------------
        # velocity toward goal
        # -----------------------
        linear_vel, ang_vel = p.getBaseVelocity(self.robot)
        vel_vec = np.array([linear_vel[0], linear_vel[1]])

        vel_toward_goal = 0.0
        if dist > 1e-6:
            vel_toward_goal = float(np.dot(vel_vec, (goal - pos) / dist))

        # -----------------------
        # progress
        # -----------------------
        progress = self.prev_dist - dist
        reward += w_progress * progress
        
        # -----------------------
        # heading alignment
        # -----------------------
        goal_dir = math.atan2(goal[1] - y, goal[0] - x)
        yaw_err = math.atan2(math.sin(goal_dir - yaw),
                            math.cos(goal_dir - yaw))
        reward += w_yaw * math.cos(yaw_err)

        reward += w_vel * vel_toward_goal

        # small time penalty to discourage stalling
        reward -= 0.01
        if self.use_dce:
            reward -= 2.0 * self.last_collision_prob

        # -----------------------
        # occlusion logic
        # -----------------------
        ray_start = [x, y, 0.3]
        ray_end = [goal[0], goal[1], 0.3]
        hit_id, _, hit_frac, _, _ = p.rayTest(ray_start, ray_end)[0]

        goal_occluded = (hit_id != -1 and hit_frac < 0.98)

        if goal_occluded:
            clearance = hit_frac - self.prev_hit_frac
            reward += w_clear * clearance

            if vel_toward_goal > 0:
                reward -= 2.0 * vel_toward_goal

        # -----------------------
        # obstacle proximity
        # -----------------------
        min_obs_dist = min(
            np.linalg.norm(pos - np.array(p.getBasePositionAndOrientation(o)[0][:2]))
            for o in self.obstacles
        )

        if obs_hard < min_obs_dist < obs_soft:
            proximity = (obs_soft - min_obs_dist) / (obs_soft - obs_hard)
            reward -= w_obs_soft * proximity

        if min_obs_dist <= obs_hard:
            reward -= w_obs_hard
            self.prev_dist = dist
            return reward, True

        # -----------------------
        # anti-spin
        # -----------------------
        if abs(ang_vel[2]) > 0.9 and abs(vel_toward_goal) < 0.3:
            reward -= 0.1
        # penalize high speed near obstacles
        min_beam = self._last_min_beam
        reward -= 0.3 * (1.0 - min_beam / self.max_beam_range)

        # -----------------------
        # termination
        # -----------------------
        self.prev_dist = dist
        self.prev_hit_frac = hit_frac

        if dist < success_dist:
            return reward + 100.0, True

        if self.step_count > 1200:
            return reward - 10.0, True
        
        arena = self._curriculum_params()["arena_half"]
        if abs(x) > arena or abs(y) > arena:
            return reward - 50.0, True

        self.debug_info = {
            "dist_to_goal": float(dist),
            "progress": float(self.prev_dist - dist),
            "vel_toward_goal": float(vel_toward_goal),
            "ang_vel": float(ang_vel[2]),
            "min_obs_dist": float(min_obs_dist),
            "goal_occluded": int(goal_occluded),
        }

        return reward, False

    
    
    
    def _curriculum_params(self):
        if self.curriculum_level == 0:
            return {
                "goal_min": 4,
                "goal_max": 6,
                "num_obs": 8,
                "arena_half": 10
            }
        elif self.curriculum_level == 1:
            return {
                "goal_min": 8,
                "goal_max": 10,
                "num_obs": 9,
                "arena_half": 18
            }
        else:
            return {
                "goal_min": 10,
                "goal_max": 12,
                "num_obs": 12,
                "arena_half": 20
            }

    def _add_curriculum1_occlusion(self):
    # --- parameters ---
        wall_length = 2.0    # half-length → total 8 m
        wall_thickness = 0.2
        wall_height = 0.2

        # goal direction
        goal_vec = self.goal / np.linalg.norm(self.goal)
        goal_yaw = np.arctan2(goal_vec[1], goal_vec[0])

        # wall orientation: perpendicular to goal
        yaw = goal_yaw + np.pi / 2
        quat = p.getQuaternionFromEuler([0, 0, yaw])

        # wall center (midway)
        center = 0.5 * self.goal

        # --- main blocking wall ---
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[wall_length*0.25, wall_thickness, wall_height]
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[wall_length*0.25, wall_thickness, wall_height],
            rgbaColor=[0.3, 0.3, 0.3, 1]
        )
        wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[center[0], center[1], wall_height],
            baseOrientation=quat
        )
        self.obstacles.append(wall_id)

        # --- gap offset ---
        gap_offset = 1.5
        lateral = np.array([-goal_vec[1], goal_vec[0]])

        gap_center = center + gap_offset * lateral

        # short blocking segment on one side only
        col2 = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[wall_length * 0.4, wall_thickness, wall_height]
        )
        vis2 = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[wall_length * 0.4, wall_thickness, wall_height],
            rgbaColor=[0.3, 0.3, 0.3, 1]
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col2,
            baseVisualShapeIndex=vis2,
            basePosition=[gap_center[0], gap_center[1], wall_height],
            baseOrientation=quat
    )

    
    
    # --------------------------------------------------------
    def close(self):
        p.disconnect()