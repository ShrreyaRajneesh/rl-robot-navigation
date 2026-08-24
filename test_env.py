from envs.cyl_depth_env import CylDepthEnv

env = CylDepthEnv(render=False)

print("\n=== RESET TEST ===")
obs, _ = env.reset()
print("Initial obs shape:", obs.shape)

print("\n=== STEP TEST ===")
a = env.action_space.sample()
obs2, r, done, trunc, info = env.step(a)
print("Step obs shape:", obs2.shape)
print("Reward =", r, " Done =", done)


