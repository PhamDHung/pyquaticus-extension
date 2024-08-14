import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
# import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


from multi_agent_trainer import MultiAgent
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
import pyquaticus.utils.rewards as rew
import dqn
import random_policy as rdm
def pyquaticus_creator():
    reward_config = {0:rew.sparse, 1:rew.sparse} 
    env = pyquaticus_v0.PyQuaticusEnv(render_mode=None, reward_config=reward_config, team_size=1)
    return env


if __name__ == '__main__':
	multi_trainer = MultiAgent(pyquaticus_creator, agents={0:dqn.DQNPolicy(pyquaticus_creator(), dqn.QNetwork, dqn.DQNArgs(), 0), 1:rdm.RandomPolicy(pyquaticus_creator(), 1)})
	multi_trainer.train_agents(100)