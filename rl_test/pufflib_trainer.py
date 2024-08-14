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

# class PolicyHanlder(nn.Module):
#     def __init__(self, env, policy_mapping):
#         super().__init__()
#         self.emulated = env.emulated
#         self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
#         self.num_actions = env.single_action_space.n
#         self.obs_size = env.single_observation_space.n
#         self.policies = {}
#         for k in policy_mapping:
#             if policy_mapping[k] == "ppo":
#                 self.policies[k] = 
        



# from pyquaticus.base_policies.base_policies import DefendGen, AttackGen
def pyquaticus_creator():
    reward_config = {0:rew.sparse, 1:rew.sparse} 
    env = pyquaticus_v0.PyQuaticusEnv(render_mode=None, reward_config=reward_config, team_size=1)
    return env
    #print("RESET OBS: ", env.reset())
    #
    #env = pufferlib.wrappers.PettingZooTruncatedWrapper(env=env)
    #env = pufferlib.postprocess.MultiagentEpisodeStats(env)
    # env = pufferlib.postprocess.MeanOverAgents(env)\
   #return env
   # return pufferlib.emulation.PettingZooPufferEnv(env=env)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Train a 2v2 policy in a 2v2 PyQuaticus environment')
   # parser.add_argument('--render', help='Enable rendering', action='store_true')
    # Example Reward Config
    
    #Competitors: reward_config should be updated to reflect how you want to reward your learning agent
    policy_mapping = {0:"ppo", 1: "ppo"}
   # args = parser.parse_args()
    #Vectorization:
    num_agents = 2
    num_workers  = 100
    num_envs = 1000
    #backend = pufferlib.vector.Multiprocessing
    #envs = pufferlib.vector.make(pyquaticus_creator, backend=backend, num_envs=num_envs, num_workers=num_workers)
    envs = pyquaticus_creator()
    obs, infos = envs.reset()
    print("Observation: ", obs)
    print("Information: ", infos)
    step2 = 0
    start = time.time()
    actions = {0:0,1:0}#[0 for i in range(num_agents * num_envs)]
    while True:
        step2 += 1
        obs, rewards, terminals, truncations, infos = envs.step(actions)#[{0:1,1:0},{0:1,1:0}])
        #print("Terminals: ", terminals)
        if terminals[0] == True:
            break
    end = time.time()
    print("Time Elapsed: ", end-start)
    print("Steps: ", step2)
        #print("Observation Strcture: ", obs)
    # policy = Policy(envs.driver_env)
    #obs_space = env.observation_space
    #act_space = env.action_space

#https://github.com/CarperAI/nmmo-baselines/blob/release/reinforcement_learning/clean_pufferl.py