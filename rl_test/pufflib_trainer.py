import pufferlib





def make_pettingzoo():
    reward_config = {0:rew.sparse, 1:rew.sparse, 2:rew.sparse, 3:rew.sparse} 
    env = pyquaticus_v0.PyQuaticusEnv(render_mode=None, reward_config=reward_config, team_size=2)
    env = pufferlib.postprocess.MultiagentEpisodeStats(env)
    vec = pufferlib.vectorization.Multiprocessing
    return vec(pufferlib.emulation.PettingZooPufferEnv(env=env))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a 2v2 policy in a 2v2 PyQuaticus environment')
    parser.add_argument('--render', help='Enable rendering', action='store_true')
    # Example Reward Config
    #Competitors: reward_config should be updated to reflect how you want to reward your learning agent
    
    args = parser.parse_args()
    make_pettingzoo()
    
    obs_space = env.observation_space
    act_space = env.action_space

https://github.com/CarperAI/nmmo-baselines/blob/release/reinforcement_learning/clean_pufferl.py