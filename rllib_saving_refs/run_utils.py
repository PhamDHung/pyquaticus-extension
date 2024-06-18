from base_policies.ctf import get_ctf_policy
from base_policies.defend import get_defense_policy
from base_policies.no_op import NoOpPolicy
from base_policies.random import get_random_policy
from base_policies.reclaim_flag import get_reclaim_flag_policy
import copy
from ctf_gym.envs.main import ctf_v0
import numpy as np
import os
from pathlib import Path
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from rl_test.utils import get_alg_name_from_trainer
import torch


TO_TRAINER = {
    'dqn': DQNConfig,
    'ppo': PPOConfig
}

def get_trainer(alg):
    return TO_TRAINER[alg]

def gen_policy(single_env):
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    return (None,obs_space,act_space,{})

def get_policy_mapping_fn(training_mode):
    '''
    Pick a red/blue policy to train against based on training mode
    Args:
        training_mode: environment mode (target1, target2, track, attack, retreat, ctf)
    Returns:
        function: a policy mapping function for blue and red agents
    '''
    blue_policy = None
    red_policy = None

    if training_mode == 'ctf':
        blue_policy = 'blue'
        red_policy = 'ctf'
    elif training_mode.startswith('target') or training_mode.startswith('track'):
        blue_policy = 'random'
        red_policy = 'red'
    elif training_mode.startswith('attack'):
        if training_mode == 'attack1':
            blue_policy = 'random'
        elif training_mode == 'attack2':
            blue_policy = 'defend'
        red_policy = 'red'
    elif training_mode.startswith('retreat'):
        if training_mode == 'retreat1':
            blue_policy = 'random'
        elif training_mode == 'retreat2':
            blue_policy = 'reclaim_flag'
        red_policy = 'red'
    else:
        raise Exception("Sorry, specified environment mode is either not recognized or not supported for training")

    def pmapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id.startswith('blue'):
            return blue_policy
        else:
            assert agent_id.startswith('red')
            return red_policy

    return pmapping_fn

def get_config(algorithm, num_workers, num_gpus, env_config_dict, framework='tf', log_level='WARN'):
    single_agent_env = ctf_v0(config_dict=env_config_dict)

    alg_config = get_trainer(algorithm)  

    #policies
    policy_graphs={'blue':         gen_policy(single_agent_env),
                   'red':          gen_policy(single_agent_env),
                   'no-op':        (NoOpPolicy, single_agent_env.observation_space, single_agent_env.action_space, {}),
                   'random':       (get_random_policy(single_agent_env), single_agent_env.observation_space, single_agent_env.action_space, {}),
                   'defend':       (get_defense_policy(single_agent_env, 'blue', 'red', human_opp=False), single_agent_env.observation_space, single_agent_env.action_space, {}),
                   'reclaim_flag': (get_reclaim_flag_policy(single_agent_env, 'blue', 'red', human_opp=False), single_agent_env.observation_space, single_agent_env.action_space, {}),
                   'ctf':          (get_ctf_policy(single_agent_env, 'red', 'blue', human_opp=False), single_agent_env.observation_space, single_agent_env.action_space, {})}

    #policy mapping function
    policy_mapping_fn = get_policy_mapping_fn(env_config_dict['env_mode'])

    #policy to train (blue or red)
    if env_config_dict['env_mode'] == 'ctf':
        policy_to_train = 'blue'
    else:
        policy_to_train = 'red'

    #create algorithm config
    config = (
        alg_config()
        .training(
            gamma=0.99,
            lr=5e-5,
        )
        .environment(
            env="ctf-v0",
            observation_space=single_agent_env.observation_space,
            action_space=single_agent_env.action_space
        )
        .framework(framework)
        .resources(num_gpus=num_gpus)
        .rollouts(
            num_rollout_workers=num_workers,
            rollout_fragment_length='auto'
        )
        .multi_agent(
            policies=policy_graphs, 
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[policy_to_train]    
        )
    )

    #set other configs
    if algorithm == 'ppo':
        config = config.training(kl_coeff=0.)
    elif algorithm == 'dqn':
        #D3QN (Rainbow DQN w/out noisy)
        config = config.training(
            double_q=True,
            dueling=True,
            n_step=1,
            noisy=False,
            num_atoms=1
        )

    config["log_level"] = log_level
    config["model"]["fcnet_hiddens"] = [256, 256]
    config["ignore_worker_failures"] = True
    config["render_env"] = False
    config["rollout_fragment_length"] = 'auto'

    return config

def restore_trainer(chkpt_file, training_mode):
    chkpt_path = Path(chkpt_file)

    loader = Algorithm.from_checkpoint(chkpt_path)
    config = loader.config.copy(copy_frozen=False)

    if training_mode == 'ctf':
        policy_to_train = 'blue'
    else:
        policy_to_train = 'red'

    policy_mapping_fn = get_policy_mapping_fn(training_mode)
    config.multi_agent(
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=[policy_to_train]
    )

    weights = loader.get_policy(policy_to_train).get_weights()
    loader.workers.stop() #close local and remote worker(s) for loader
    
    trainer = config.build()
    trainer.get_policy(policy_to_train).set_weights(weights)

    return trainer, config

def save_torch_model(algo, config, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    alg = get_alg_name_from_trainer(algo)

    torch_config = copy.deepcopy(config)
    torch_config.framework(framework='torch')
    torch_config.resources(num_gpus=0)
    torch_config.rollouts(num_rollout_workers=0)
    assert len(torch_config['policies_to_train']) == 1, 'Number of models to save != 1'
    
    policy_id_save = torch_config['policies_to_train'][0]
    model_weights = algo.get_policy(policy_id=policy_id_save).get_weights()

    ##### convert weights from framework to torch #####
    model_weights_torch = []
    if config['framework_str'] == 'tf':
        for v in model_weights.values():
            model_weights_torch.append(v.copy().T)
    elif config['framework_str'] == 'tf2':
        for i in model_weights:
            model_weights_torch.append(i.copy().T)
    else:
        model_weights_torch = copy.deepcopy(model_weights)

    ##### build saver and set weights #####
    saver = torch_config.build()
    saver_weights = saver.get_policy(policy_id=policy_id_save).get_weights()

    if config['framework_str'].startswith('tf'):
        saver_weights_keys = list(saver_weights.keys())
        for i in range(len(saver_weights_keys)):
            saver_weights[saver_weights_keys[i]] = model_weights_torch[i]
    else:
        saver_weights = model_weights_torch

    saver.get_policy(policy_id=policy_id_save).set_weights(saver_weights)

    ##### save modules #####
    if alg == 'dqn':
        #cannot use rllib built-in export_model() method
        assert(torch_config['dueling']), 'expecting dueling DQN architecture'

        #extract dueling-dqn torch models
        base = saver.get_policy(policy_id=policy_id_save).model._modules['_hidden_layers']
        advantage = saver.get_policy(policy_id=policy_id_save).model._modules['advantage_module']
        value = saver.get_policy(policy_id=policy_id_save).model._modules['value_module']

        #save dueling-dqn torch models
        n_obs = np.prod(torch_config.observation_space.shape)
        base_output_size = torch_config["model"]["fcnet_hiddens"][-1]

        torch.jit.save(torch.jit.trace(base, torch.zeros(n_obs)), save_path + '/base.pt')
        torch.jit.save(torch.jit.trace(advantage, torch.zeros(base_output_size)), save_path + '/advantage.pt')
        torch.jit.save(torch.jit.trace(value, torch.zeros(base_output_size)), save_path + '/value.pt')
    else:
        raise NotImplementedError("Sorry, {} has not been implemented".format(alg))

    return saver

def save_tf2_model(algo, config, save_path):
    alg = get_alg_name_from_trainer(algo)

    tf2_config = copy.deepcopy(config)
    tf2_config.framework(framework='tf2')
    tf2_config.resources(num_gpus=0)
    tf2_config.rollouts(num_rollout_workers=0)
    assert len(tf2_config['policies_to_train']) == 1, 'Number of models to save != 1'
    
    policy_id_save = tf2_config['policies_to_train'][0]
    model_weights = algo.get_policy(policy_id=policy_id_save).get_weights()

    if config['framework_str'] == 'tf':
        model_weights_tf2 = list(model_weights.values()) #convert OrderedDict to list for tf2 model
    elif config['framework_str'] == 'tf2':
        model_weights_tf2 = copy.deepcopy(model_weights)
    else:
        raise NotImplementedError('Conversion from {} trainer to tf2 saver'.format(config['framework_str']))

    saver = tf2_config.build()
    saver.get_policy(policy_id=policy_id_save).set_weights(model_weights_tf2)

    if alg == 'dqn':
        #cannot use rllib built-in export_model() method
        assert(tf2_config['dueling']), 'expecting dueling DQN architecture'

        #extract dueling-dqn keras models
        base = saver.get_policy(policy_id=policy_id_save).model.base_model
        advantage = saver.get_policy(policy_id=policy_id_save).model.q_value_head
        value = saver.get_policy(policy_id=policy_id_save).model.state_value_head

        #save dueling-dqn keras models
        base.save(save_path + '/base', save_format='tf')
        advantage.save(save_path + '/advantage', save_format='tf')
        value.save(save_path + '/value', save_format='tf')

    elif alg == 'ppo':
        saver.get_policy(policy_id=policy_id_save).export_model(save_path)
    else:
        raise NotImplementedError("Sorry, {} has not been implemented".format(alg))

    return saver