class RandomPolicy:
	def __init__(self, env, agent_id):
		self.trainable = False
		self.env = env
		self.agent_id = agent_id
		self.action_space = self.env.action_spaces[self.agent_id]
	def get_action(self):
		return self.action_space.sample()
	