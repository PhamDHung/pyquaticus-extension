import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
import pyquaticus.utils.rewards as rew
from pyquaticus.base_policies.base_policies import DefendGen, AttackGen

env = pyquaticus_v0.PyQuaticusEnv(render_mode=None, reward_config=reward_config, team_size=2)

#env = ParallelPettingZooWrapper(pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=2))
#register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
obs_space = env.observation_spaces[0]
act_space = env.action_spaces[0]


