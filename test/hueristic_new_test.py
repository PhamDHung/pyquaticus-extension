import sys
import os
import os.path
import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender, fieldpoint2action
from pyquaticus.base_policies.base_shield import BaseShield
from pyquaticus.base_policies.ctf_config import PYQUATICUS_ACTION_MAP
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.envs.pyquaticus import Team
from collections import OrderedDict
from pyquaticus.config import config_dict_std, ACTION_MAP

config_dict = config_dict_std
config_dict["max_time"] = 600.0
config_dict["max_score"] = 100
config_dict["sim_speedup_factor"] = 1

env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config_dict,render_mode='human', render_agent_ids=True)
term_g = {0:False,1:False}
truncated_g = {0:False,1:False}
term = term_g
trunc = truncated_g
obs = env.reset()
temp_score = env.game_score

H_one = BaseAttacker(3, Team.RED_TEAM, mode='competition_medium_new')
H_two = BaseDefender(4, Team.RED_TEAM, mode='easy_patrol')
H_three = BaseAttacker(5, Team.RED_TEAM, mode='competition_medium_new')

R_one = BaseShield(config_dict_std["agent_radius"], config_dict_std["catch_radius"], config_dict_std["world_size"])
R_two = BaseAttacker(1, Team.BLUE_TEAM, mode="competition_medium")
R_three = BaseShield(config_dict_std["agent_radius"], config_dict_std["catch_radius"], config_dict_std["world_size"])
step = 0
while True:
    new_obs = {}
    for k in obs:
        # if k == 0:
        #     new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k]),"blue_one", ["blue_two", "blue_three"], ["red_one", "red_two", "red_three"])
        # elif k == 2:
        #     new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k]), "blue_three", ["blue_one", "blue_two"], ["red_one", "red_two", "red_three"])
        # else:
        new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k])

    three = H_one.compute_action(new_obs)
    four = H_two.compute_action(new_obs)
    four = fieldpoint2action(four, env.players[4].pos, env.players[4].heading, Team.RED_TEAM)
    five = H_three.compute_action(new_obs)
    zero = PYQUATICUS_ACTION_MAP[R_one.compute_action(new_obs,0, [0, 1, 2], [3,4,5])]
    one = R_two.compute_action(new_obs)
    two = PYQUATICUS_ACTION_MAP[R_three.compute_action(new_obs, 2, [0, 1, 2], [3,4,5])]

    
    obs, reward, term, trunc, info = env.step({0:zero,1:one, 2:two, 3:three, 4:four, 5:five})
    k =  list(term.keys())

    step += 1
    if term[k[0]] == True or trunc[k[0]]==True:
        break
for k in env.game_score:
    temp_score[k] += env.game_score[k]
env.close()