import gymnasium
from PyFlyt.gym_envs import FlattenWaypointEnv

model = ... # load the model somehow

eval_episodes = 10
for ep in range(eval_episodes):
    render_env = FlattenWaypointEnv(gymnasium.make("PyFlyt/QuadX-Waypoints-v1", render_mode="human"), context_length=1)
    obs = render_env.reset()
    obs = obs[0]
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = render_env.step(action)
        done = terminated or truncated
        render_env.render()
    render_env.close()