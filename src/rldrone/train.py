import gymnasium
import os
import random
import string

from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import PPO
from stable_baselines3 import A2C

LOG = True

environment = "QuadX-Waypoints-v1"
algorithm = "A2C"
# algorithm = "PPO"
id = ''.join(random.choices(string.ascii_letters, k=20))

models_dir = f"models/{algorithm+'_'+environment+'_'+id}"
logdir = "data"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

train_env = make_vec_env(lambda: FlattenWaypointEnv(gymnasium.make(f"PyFlyt/{environment}"), context_length=1), n_envs=1)

model = A2C("MlpPolicy", train_env, verbose=1, tensorboard_log=logdir)
# model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=logdir)


TIMESTEPS = 10000
for i in range(1, 50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=algorithm+'_'+environment+'_'+id)
    if LOG:
        model.save(f"{models_dir}/{TIMESTEPS*i}")
train_env.close()