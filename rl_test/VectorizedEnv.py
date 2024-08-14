#Python
import multiprocessing
import time

#Pyquaticus Environment Imports
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
import pyquaticus.utils.rewards as rew

def pyquaticus_creator():
    reward_config = {0:rew.sparse, 1:rew.sparse} 
    env = pyquaticus_v0.PyQuaticusEnv(render_mode=None, reward_config=reward_config, team_size=1)
    return env

def worker(env_creator, num_runs, return_dict, game_index):
    """TODO: Handle Truncations"""
    
    for g in range(num_runs):
        env = env_creator()
        
        game = {"obs":{}, "truncateds":{}, "terminateds":{}, "infos":{}, "rewards":{}}
        obs, infos = env.reset()
        for a in obs:
            if a not in game["obs"]:
                game["obs"][a] = []
                game["truncateds"][a] = []
                game["terminateds"][a] = []
                game["infos"][a] = []
                game["rewards"][a] = []
            game["obs"][a].append(obs[a])
            game["infos"][a].append({})
            # if game["infos"] == {}:
            #     game["infos"][a].append({})
            # else:
            #     game["infos"][a].append(infos[a])
        while True:
            obs, rewards, terminals, truncations, infos = env.step({0:1,1:0})#[{0:1,1:0},{0:1,1:0}])
            for a in obs:
                if truncations[a] == False:
                    game["obs"][a].append(obs[a])
                    game["truncateds"][a].append(truncations[a])
                    game["terminateds"][a].append(terminals[a])
                    game["rewards"][a].append(rewards[a])
                    game["infos"][a].append({})
                elif game["truncations"][a][-1] == True:
                    continue
                #print("Game Infos: ", game["infos"])
                # if game["infos"] == {}:
                #     game["infos"][a].append({})
                # else:
                #     game["infos"][a].append(infos[a])
            #print("Terminals: ", terminals)
            if terminals[0] == True:
                break
        #print("saving Dict: ", game_index+g)
        return_dict[game_index + g] = game



if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    start = time.time()
    num_cores = 8
    num_games = 128
    games_per_core = num_games // num_cores
    print("Games Per Core: ", games_per_core)
    for i in range(num_cores):
        #print('launching: ', i)
        p = multiprocessing.Process(target=worker, args=(pyquaticus_creator, games_per_core, return_dict, i*games_per_core))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    end = time.time()
    print("Number of Games: ", len(return_dict))
    print("Time Required: ", end-start)