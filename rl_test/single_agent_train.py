# (C) 2021 Massachusetts Institute of Technology.

# Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

# The software/firmware is provided to you on an As-Is basis

# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause
import argparse
import gymnasium as gym
import numpy as np
import pygame
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec, Policy
import os
import pyquaticus.utils.rewards as rew
from pyquaticus.base_policies.base_policies import DefendGen, AttackGen
from pyquaticus.config import config_dict_std

class RandPolicy(Policy):
    """
    Example wrapper for training against a random policy.

    To use a base policy, insantiate it inside a wrapper like this,
    and call it from self.compute_actions

    See policies and policy_mapping_fn for how policies are associated
    with agents
    """
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a 1v1 policy in a 1v1 PyQuaticus environment')
    parser.add_argument('--render', help='Enable rendering')
    reward_config = {0:rew.custom_v1, 1:None}#, 2:None, 3:None} # Example Reward Config
    #Competitors: reward_config should be updated to reflect how you want to reward your learning agent
    
    args = parser.parse_args()


    RENDER_MODE = 'human' if args.render else None #set to 'human' if you want rendered output
    cfd = config_dict_std
    cfd['teleport_on_tag'] = True
    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, config_dict=cfd, reward_config=reward_config, team_size=1)
    env = ParallelPettingZooWrapper(pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, config_dict=cfd, reward_config=reward_config, team_size=1))
    register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
    obs_space = env.observation_space
    act_space = env.action_space
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 0 or agent_id == 'agent-0':
            return "agent-0-policy"
        if agent_id == 1 or agent_id == 'agent-1':
            return "easy-attack-policy"
        #elif agent_id == 2 or agent_id == 'agent-2':
            # change this to agent-1-policy to train both agents at once
         #   return "easy-defend-policy"
        #else:
         #   return "easy-attack-policy"
    
    policies = {'agent-0-policy':(None, obs_space, act_space, {}),
                #'easy-defend-policy': (DefendGen(1, Team.RED_TEAM, 'competition_easy', 1, env.par_env.agent_obs_normalizer), obs_space, act_space, {}),
                'easy-attack-policy': (AttackGen(1, Team.RED_TEAM, 'competition_easy', 1, env.par_env.agent_obs_normalizer), obs_space, act_space, {})}
                #'no-action-policy': (AttackGen(1,Team.RED_TEAM,'competition_nothing',1,env.par_env.agent_obs_normalizer),obs_space, act_space, {})}
    env.close()
    #Current Config 30 rollout workers 
    #TODO: Try to make it 60 rollout workers and see if we can try and maximize gpu performance a bit more utilization is a bit low still
    #ppo_config = PPOConfig().environment(env='pyquaticus').rollouts(num_rollout_workers=30).resources(num_cpus_per_worker=0.5, num_gpus=1, num_gpus_per_worker=0.033)#int(os.environ.get("RLLIB_NUM_GPUS", "0")))#.evaluation(evaluation_interval=1, evaluation_num_workers=1)
    ppo_config = PPOConfig().environment(env='pyquaticus').rollouts(num_rollout_workers=10).resources(num_cpus_per_worker=0.3, num_gpus=1, num_gpus_per_worker=0.033)
    #If your system allows changing the number of rollouts can significantly reduce training times (num_rollout_workers=15)
    ppo_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["agent-0-policy"],)
    algo = ppo_config.build()


    for i in range(30001):
        print("Looping: ", i)
        algo.train()
        if np.mod(i, 500) == 0:
            print("Saving Checkpoint: ", i)
            chkpt_file = algo.save('./ray_test/')
            print(f'Saved to {chkpt_file}', flush=True)