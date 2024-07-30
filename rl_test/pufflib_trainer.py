#Pufferlib Imports
import pufferlib
import pufferlib.emulation
import pufferlib.wrappers
import pufferlib.vector
import pufferlib.postprocess
import pufferlib.models
import pufferlib.pytorch

#Pyquaticus Environment Imports
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
import pyquaticus.utils.rewards as rew

#Additional Required Imports
from pdb import set_trace as T
import torch
from torch import nn
import numpy as np
import sys
import time

class PolicyHanlder(nn.Module):
    def __init__(self, env, policy_mapping):
        super().__init__()
        self.emulated = env.emulated
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.obs_size = env.single_observation_space.n
        



# from pyquaticus.base_policies.base_policies import DefendGen, AttackGen
def pyquaticus_creator():
    reward_config = {0:rew.sparse, 1:rew.sparse, 2:rew.sparse, 3:rew.sparse} 
    env = pyquaticus_v0.PyQuaticusEnv(render_mode=None, reward_config=reward_config, team_size=2)
    
    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env=env)
    env = pufferlib.postprocess.MultiagentEpisodeStats(env)
    env = pufferlib.postprocess.MeanOverAgents(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Train a 2v2 policy in a 2v2 PyQuaticus environment')
   # parser.add_argument('--render', help='Enable rendering', action='store_true')
    # Example Reward Config
    #Competitors: reward_config should be updated to reflect how you want to reward your learning agent
    policy_mapping = {0:"zero", 1: "one", 2: "two", 3:, 4: ,5: }
   # args = parser.parse_args()
    #Vectorization:
    backend = pufferlib.vector.Multiprocessing
    envs = pufferlib.vector.make(pyquaticus_creator, backend=backend, num_envs=2, num_workers=1)
    
    policy = Policy(envs.driver_env)
    #obs_space = env.observation_space
    #act_space = env.action_space

#https://github.com/CarperAI/nmmo-baselines/blob/release/reinforcement_learning/clean_pufferl.py