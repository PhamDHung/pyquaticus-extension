# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
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
from pygame import KEYDOWN, QUIT, K_ESCAPE
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPOTF1Policy, PPOTorchPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec
import os
from pyquaticus.base_policies.base_policies import DefendGen, AttackGen
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import Policy
import pyquaticus.utils.rewards as rew
from pyquaticus.config import config_dict_std, ACTION_MAP
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper

RENDER_MODE = None#'human'
#RENDER_MODE = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy a trained policy in a 2v2 PyQuaticus environment')
    parser.add_argument('policy_one', help='Please enter the path to the model you would like to load in')
    parser.add_argument('fname', help='Please enter the file name for the saved actions')
    parser.add_argument('scores', help='Please enter the file name where you would want to save the game scores')
    #parser.add_argument('policy_two', help='Please enter the path to the model you would like to load in') 
    test = True
    reward_config = {0:rew.custom_v1, 1:None}
    args = parser.parse_args()
    scores = open(args.scores, 'a')
    f = open(args.fname, 'a')
   # def policy_mapping_fn(agent_id, episode, worker, **kwargs):
   #     if agent_id == 0 or agent_id == 'agent-0':
   #         return "agent-0-policy"
   #     if agent_id == 1 or agent_id == 'agent-1':
   #         return "easy-attack-policy"
        #elif agent_id == 2 or agent_id == 'agent-2':
            # change this to agent-1-policy to train both agents at once
         #   return "easy-defend-policy"
        #else:
         #   return "easy-attack-policy"
    #config_dict = config_dict_std
    #config_dict["max_time"] = 600.0
    #config_dict["max_score"] = 10000
    #config_dict["teleport_on_tag"] = True

    #env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, config_dict=config_dict, reward_config=reward_config, team_size=1)
    #env = ParallelPettingZooWrapper(pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, config_dict=config_dict, reward_config=reward_config, team_size=1))
    #obs_space = env.observation_space
    #act_space = env.action_space
    #policies = {'agent-0-policy':(None, obs_space, act_space, {}),
     #           'easy-defend-policy': (DefendGen(1, Team.RED_TEAM, 'competition_easy', 1, env.par_env.agent_obs_normalizer), obs_space, act_space, {}),
     #           'easy-attack-policy': (AttackGen(1, Team.RED_TEAM, 'competition_easy', 1, env.par_env.agent_obs_normalizer), obs_space, act_space, {})}

    #register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
    #ppo_config = ppo_config = PPOConfig().environment(env='pyquaticus').rollouts(num_rollout_workers=10).resources(num_cpus_per_worker=1, num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    #ppo_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["agent-0-policy"],)
    #algo = ppo_config.build()
    #algo.restore(args.policy_one)
    easy_score = 0
    for i in range(10):
        config_dict = config_dict_std
        config_dict["max_time"] = 600.0
        config_dict["max_score"] = 1
        config_dict["teleport_on_tag"] = True
        env = pyquaticus_v0.PyQuaticusEnv(render_mode='human', team_size=1,reward_config=reward_config, config_dict=config_dict)
        term_g = {0:False,1:False}
        truncated_g = {0:False,1:False}
        term = term_g
        trunc = truncated_g
        obs = env.reset()
        temp_score = env.game_score
        #print("Initial Game Score: ", temp_score)
    #H_one = BaseDefender(2, Team.RED_TEAM, mode='competition_easy')
    
        H_one = BaseAttacker(1, Team.RED_TEAM, mode='competition_easy')
        policy_one = Policy.from_checkpoint(args.policy_one)
 #  Policy.from_checkpoint(args.policy_two)

   #policy_two = Policy.from_checkpoint(args.policy_two)
        step = 0
    #max_step = 5000
    #print("Obs 0: ", obs[0])
    #print("obs 1: ", obs[1])
        while True:
            new_obs = {}
        #Get Unnormalized Observation for heuristic agents (H_one, and H_two)
            for k in obs:
                new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k])

        #Get learning agent action from policy
        #print("Obs: ", obs)
            zero = policy_one.compute_single_action(obs[0], explore=False)[0]
        #one = policy_two.compute_single_action(obs[1])[0]
        #Compute Heuristic agent actions
            one = H_one.compute_action(new_obs)
        #three = H_two.compute_action(new_obs)
        #Step the environment
        #print("Agent Action: ", zero)
            obs, reward, term, trunc, info = env.step({0:zero,1:one})
            x_pos = new_obs[0]["wall_3_distance"]
            y_pos = new_obs[0]["wall_2_distance"]
            f.write(str(step)+','+str(x_pos)+','+str(y_pos)+'\n')
        #for k in reward:
        #    if reward[k] > 0:
        #        print("Agent ID: ", k, " Reward: ", reward[k])
        #if step >= max_step:
        #    break
            step += 1
            end = False
        #print("Truncation: ", trunc)
        #print("Terminateds: ", term)
      #  for k in trunc:
      #      #print("trunc: ", k)
      #      if trunc[k] == True:
      #          print("Trunc Detected Agent_ID: ", k)
      #          print(trunc)
      #          end = True
      #          break
        #print("Terminated: ",term)
            end = False
            for k in term:
                if term[k] == True or trunc[k] == True:
                    end = True
                    break
            if end:
                break

        #if term[k[0]] == True or trunc[k[0]]==True:
        #    for k in env.game_score:
        #        temp_score[k] += env.game_score[k]
         #   env.reset()
        for k in env.game_score:

            temp_score[k] = env.game_score[k]
        env.close()
        #print("Final Game Score: ", temp_score)

        scores.write(str("Game:"+str(i)+",Score:"+str(temp_score['blue_captures'] - temp_score['red_captures'])+'\n'))
        easy_score += temp_score['blue_captures'] - temp_score['red_captures']


