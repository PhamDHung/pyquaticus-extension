import pyquaticus
from pyquaticus import pyquaticus_v0
import pyquaticus.utils.rewards as rew
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
# Parallel environments

class Models


reward_config = {0:rew.sparse, 1:rew.sparse}
env = pyquaticus_v0.PyQuaticusEnv(render_mode=None, team_size=1, reward_config=reward_config)

env.reset()
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")


model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=1e-3,
        batch_size=256,
    )

model.learn(total_timesteps=5000000)

env.close()