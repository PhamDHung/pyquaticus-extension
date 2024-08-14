
#Pyquaticus:
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
#Python
import multiprocessing
import time

def gather_batch(env_creator=None, num_cores=8, num_games=128):
	manager = multiprocessing.Manager()
	return_dict = manager.dict()
	jobs = []
	start = time.time()
	games_per_core = num_games // num_cores
	for i in range(num_cores):
		#print('launching: ', i)
		p = multiprocessing.Process(target=worker, args=(pyquaticus_creator, games_per_core, return_dict, i*games_per_core))
		jobs.append(p)
		p.start()
	for proc in jobs:
		proc.join()
	end = time.time()
	return return_dict


def worker(env_creator, num_runs, return_dict, game_index):    
	for g in range(num_runs):
		env = pyquaticus_creator()
		game = {"obs":{}, "truncateds":{}, "terminateds":{}, "infos":{}, "rewards":{}, "actions":{}}
		obs, infos = env.reset()
		for a in obs:
			if a not in game["obs"]:
				game["obs"][a] = []
				game["truncateds"][a] = []
				game["terminateds"][a] = []
				game["infos"][a] = []
				game["rewards"][a] = []
				game["actions"][a] = []
			game["obs"][a].append(obs[a])
			game["infos"][a].append({})
			# if game["infos"] == {}:
			#     game["infos"][a].append({})
			# else:
			#     game["infos"][a].append(infos[a])
		while True:
			actions = {0:1,1:0}
			obs, rewards, terminals, truncations, infos = env.step({0:1,1:0})#[{0:1,1:0},{0:1,1:0}])
			for a in obs:
				if truncations[a] == False:
					game["obs"][a].append(obs[a])
					game["truncateds"][a].append(truncations[a])
					game["terminateds"][a].append(terminals[a])
					game["rewards"][a].append(rewards[a])
					game["infos"][a].append({})
					game["actions"][a].append(actions[a])
				elif game["truncations"][a][-1] == True:
					continue
			if terminals[0] == True:
				break
		return_dict[game_index + g] = game

class MultiAgent:
	def __init__(self, env_creator, agents = {}):
		self.env_creator = env_creator
		self.agents = agents
		self.global_step = 0
	def train_agents(self, iterations):

		game_batch = gather_batch(env_creator=self.env_creator, num_cores=8, num_games=128)
		for g in game_batch:
			for a in self.agents:
				if self.agents[a].trainable == True:
					b = {"obs":game_batch[g]["obs"][a], "truncateds":game_batch[g]["truncateds"][a], "terminateds":game_batch[g]["terminateds"][a], "rewards":game_batch[g]["rewards"][a], "actions":game_batch[g]["actions"], "infos":game_batch[g]["infos"][a]}
					self.agents[a].train(b)
				else:
					continue

		print("Length: ", len(game_batch))
		# 	for step in range()
		# 	for a in self.agents:

