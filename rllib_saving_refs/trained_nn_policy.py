import numpy as np
import os
from tensorflow import keras
import torch


class Trained_DQN_Policy():
    """
    class for serving a trained Dueling-DQN policy
    """
    def __init__(self, model_path, action_space, framework='torch'):
        model_path += os.sep
        if framework == 'torch':
            self.base = torch.jit.load(model_path + 'base.pt', map_location='cpu')
            self.advantage = torch.jit.load(model_path + 'advantage.pt', map_location='cpu')
            self.value = torch.jit.load(model_path + 'value.pt', map_location='cpu')

        else:
            #load keras models
            self.base = keras.models.load_model(model_path + 'base')
            self.advantage = keras.models.load_model(model_path + 'advantage')
            self.value = keras.models.load_model(model_path + 'value')

        self.action_space = action_space
        self.framework = framework

    def inference(self, x):
        """
        Can perform batch or single inference
        """
        if len(x.shape) == 1:
            x = x[np.newaxis, :]

        if self.framework == 'torch':
            base_out = self.base(torch.from_numpy(x.astype('float32'))).detach()
            a = self.advantage(base_out).detach().numpy()
            v = self.value(base_out).detach().numpy()
        else:
            #keras
            base_out = self.base.predict(x, verbose=0)
            a = self.advantage.predict(base_out[0], verbose=0)[0]
            v = self.value.predict(base_out[0], verbose=0)

        return a, v

    def compute_action(self, obs):
        """
        Can compute a single action from a single obvervation...
        or an action batch from an observation batch
        """
        A, V = self.inference(obs)
        A_mean = A.mean(axis=1)[:, np.newaxis]

        Q = (A - A_mean) + V
        assert Q.shape[1] == self.action_space.n, 'number of DQN output logits != number of discrete actions'

        action = Q.argmax(axis=1)[:, np.newaxis]

        if action.shape == (1,1):
            #extract action from array for convenience (when batch size = 1)
            return action[0][0]

        return action


class Trained_PPO_Policy():
    """
    class for serving a trained PPO policy NN
    """
    def __init__(self, model_path, action_space):
        self.model = keras.models.load_model(model_path)
        self.action_space = action_space

    def inference(self, x):
        """
        Can perform batch or single inference
        """
        if len(x.shape) == 1:
            x = x[np.newaxis, :]

        return self.model.predict(x, verbose=0)

    def compute_action(self, obs):
        """
        Can compute a single action from a single obvervation...
        or an action batch from an observation batch
        """
        results = self.inference(obs)

        #will assume a discrete action space
        logits = results[0]
        assert logits.shape[1] == self.action_space.n, 'number of PPO output logits != number of discrete actions'

        action_prob = np.exp(logits) / np.expand_dims(np.exp(logits).sum(axis=1), axis=1)
        prob_intervals = np.insert(action_prob.cumsum(axis=1), 0, 0., axis=1)

        n_samples = prob_intervals.shape[0]
        sample = np.random.rand(n_samples, 1)

        interval_bool = np.where(sample < prob_intervals) #find index of first interval endpoint > sample
        sample_idices = interval_bool[0]

        unique, first_idx = np.unique(sample_idices, return_index=True)

        action = interval_bool[1][first_idx] - 1 #index of upper bound on interval will be +1 of interval number
        action = action[:, np.newaxis]
        

        if action.shape == (1,1):
            #extract action from array for convenience (when batch size = 1)
            return action[0][0]

        return action