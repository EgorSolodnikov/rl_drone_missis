import gymnasium
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import PPO
from stable_baselines3 import A2C

environment = "QuadX-Waypoints-v1"
# algorithm = "A2C"
algorithm = "PPO"

models_dir = "models"
model_path = f"{models_dir}/PPO_QuadX-Waypoints-v1_cxmnDyXhOMvrRomgBwGR/410000.zip"

env = make_vec_env(lambda: FlattenWaypointEnv(gymnasium.make(f"PyFlyt/{environment}"), context_length=1), n_envs=1)
model = PPO.load(model_path, env=env)

eval_episodes = 10
for ep in range(eval_episodes):
    render_env = FlattenWaypointEnv(gymnasium.make(f"PyFlyt/{environment}", render_mode="human"), context_length=1)
    obs = render_env.reset()
    obs = obs[0]
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = render_env.step(action)
        done = terminated or truncated
        render_env.render()
    render_env.close()