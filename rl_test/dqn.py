import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro


class DQNArgs:
	"""the id of the environment"""
	total_timesteps: int = 500000
	"""total timesteps of the experiments"""
	learning_rate: float = 2.5e-4
	"""the learning rate of the optimizer"""
	num_envs: int = 1
	"""the number of parallel game environments"""
	buffer_size: int = 10000
	"""the replay memory buffer size"""
	gamma: float = 0.99
	"""the discount factor gamma"""
	tau: float = 1.0
	"""the target network update rate"""
	target_network_frequency: int = 500
	"""the timesteps it takes to update the target network"""
	batch_size: int = 128
	"""the batch size of sample from the reply memory"""
	start_e: float = 1
	"""the starting epsilon for exploration"""
	end_e: float = 0.05
	"""the ending epsilon for exploration"""
	exploration_fraction: float = 0.5
	"""the fraction of `total-timesteps` it takes from start-e to go end-e"""
	learning_starts: int = 10000
	"""timestep to start learning"""
	train_frequency: int = 10
	"""the frequency of training"""
class QNetwork(nn.Module):
	def __init__(self, obs_space, action_space):
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(np.array(obs_space.shape).prod(), 120),
			nn.ReLU(),
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.Linear(84, action_space.n),
		)
		print("Here2")

	def forward(self, x):
		return self.network(x)

class DQNPolicy:
	def __init__(self, env, network, args, agent_id):
		self.trainable = True
		self.env = env
		self.global_step = 0
		self.args = args
		print("w0: ", env.observation_spaces[agent_id].shape)
		print("Actions: ", env.action_spaces[agent_id].n)
		self.q_network = QNetwork(obs_space=env.observation_spaces[agent_id], action_space=env.action_spaces[agent_id])
		print("w1")
		self.target_network = QNetwork(obs_space=env.observation_spaces[agent_id], action_space=env.action_spaces[agent_id])
		print("w2")
		self.target_network.load_state_dict(self.q_network.state_dict())
		print("w3")
		self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.args.learning_rate)
	def forward(self, x):
		return self.network.forward(x)
	def train(self, batch):
		#Given a batch of games train a policy
		#for step in batch:
		print("Observations: ", batch["obs"])
		next_obs = np.array(batch["obs"][1:])
		obs = np.array(batch["obs"])
		rewards = np.array(batch["rewards"])
		actions = np.array(batch["actions"])

		with torch.no_grad():
			target_max, _ = self.target_network(next_obs).max(dim=1)
			td_target = rewards.flatten() + args.gamma * target_max * (1 - dones.flatten())
		old_val = self.q_network(obs).gather(1, actions).squeeze()
		loss = F.mse_loss(td_target, old_val)
		# optimize the model
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		self.global_step += len(batch["obs"])

		# update target network
		if self.global_step % self.args.target_network_frequency == 0:
			for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
				target_network_param.data.copy_(
					args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
				)
	def save_model(self, path):
		torch.save(q_network.state_dict(), path)