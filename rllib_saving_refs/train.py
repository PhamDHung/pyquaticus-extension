import argparse
from ctf_gym.envs.main import ctf_v0, config_dict_std
import numpy as np
import ray
from ray.tune.registry import register_env
from rl_test.run_utils import get_config, restore_trainer, save_torch_model
from rl_test.utils import get_alg_name_from_trainer, get_checkpoint_root_path, get_model_save_path


ENV_CONFIG_DICT = config_dict_std.copy()

def main():
    parser = argparse.ArgumentParser(description='Main entry for training')
    parser.add_argument('--alg', default=None, choices=[None, 'dqn', 'ppo'])
    parser.add_argument('--env_mode', choices=['ctf', 'target1', 'target2', 'track1', 'track2', 'attack1', 'attack2', 'retreat1', 'retreat2'])
    parser.add_argument('--chkpt', default=None, help='The checkpoint file to load')
    parser.add_argument('--iter', default=1000, type=int, help='Number of rllib training iterations to run')
    args = parser.parse_args()

    ##Setup ray parameters and custom gym env##
    ray.init(ignore_reinit_error=True, local_mode=False)
    name_env = "ctf-v0"
    ENV_CONFIG_DICT['env_mode'] = args.env_mode

    def env_creator(_):
        return ctf_v0(config_dict=ENV_CONFIG_DICT)

    register_env(name_env, env_creator)

    #create trainer
    if args.chkpt: 
        #restore trainer from checkpoint
        print('Restoring checkpoint and algorithm from:', args.chkpt)

        if args.alg:
            print("WARNING: You specified a checkpoint to restore from. Algorithm choice specified with --alg will be overwritten by checkpoint's algo.")

        trainer, config = restore_trainer(args.chkpt, args.env_mode)

        alg = get_alg_name_from_trainer(trainer)

        chkpt_root, home_dir = get_checkpoint_root_path(f'{alg}_{args.env_mode}')
        model_save_path = get_model_save_path(f'{alg}_{args.env_mode}')
    else:
        if args.alg == None:
            raise Exception("If not restoring from a checkpoint, you must specify an algorithm to use for training!")

        #create new trainer from algorithm config object
        chkpt_root, home_dir = get_checkpoint_root_path(f'{args.alg}_{args.env_mode}')
        model_save_path = get_model_save_path(f'{args.alg}_{args.env_mode}')

        config = get_config(
            algorithm=args.alg,
            num_workers=4,
            num_gpus=1.,
            env_config_dict=ENV_CONFIG_DICT,
            log_level='ERROR'
        )

        assert len(config['policies_to_train']) == 1, 'config should not specify to train more than 1 policy at a time'
        trainer = config.build()

    print('rllib checkpoint files will be saved to: ', chkpt_root)
    print('Torch model(s) will be saved to: ', model_save_path)
    ray_results = "{}/ray_results/".format(home_dir)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    for n in range(args.iter+1):
        result = trainer.train()

        if np.mod(n,100) == 0:
            chkpt_file = trainer.save(chkpt_root)

        print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            chkpt_file
        ))

    #save model with torch framework
    saver = save_torch_model(trainer, config, model_save_path)
    print('Torch model(s) saved to: ', model_save_path)

    ray.shutdown()

if __name__ == "__main__":
    main()
