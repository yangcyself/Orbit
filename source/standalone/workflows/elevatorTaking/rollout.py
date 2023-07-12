# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to rollout a directory of policies from robomimic, store them with tensorboard"""


"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--checkpoints", type=str, default=None, help="the pytorch checkpoints to load. Use __e__ as a placeholder for epoch number.")
parser.add_argument("--resume", type=str, default=None, help="resume a previous rollout. Provide the path to the `rollout_log.txt`.")
parser.add_argument("--task", type=str, default="pushbtn", help="Name of the task: can be pushbtn or movetobtn")

args_cli = parser.parse_args()
# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import gym
import torch
from datetime import datetime

import robomimic  # noqa: F401
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.utils.log_utils import DataLogger

from robomimic.algo import RolloutPolicy
from robomimic.envs.env_gym import EnvGym

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml, load_pickle, load_yaml
from omni.isaac.orbit.utils import update_class_from_dict
from glob import glob
import re
import os
import sys

sys.path.append(os.path.dirname(__file__))
from utils.mimic_utils import RobomimicWrapper, myEnvGym
from utils.env_presets import (modify_cfg_to_robomimic, 
        modify_cfg_to_task_push_btn, 
        modify_cfg_to_task_move_to_btn)

# Define a function to parse a single line of the log file
def parse_log_line(line):
    # Split the line into epoch and timestamp using the colon and space as the separator
    epoch, timestamp = line.split(': ')
    # Remove the newline character from the timestamp
    timestamp = timestamp.strip()
    return int(epoch), timestamp


def compare_dicts(dict1, dict2, path=""):
    # Check if both are dictionaries
    if not (isinstance(dict1, dict) and isinstance(dict2, dict)):
        if dict1 != dict2:
            print(f"Different values at path '{path}': {dict1} != {dict2}")
        return
    
    # Check if both dictionaries have the same keys
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    
    # Check for keys in dict1 but not in dict2
    for key in keys1 - keys2:
        print(f"Key '{key}' found in first dict but not in second at path '{path}'")
    
    # Check for keys in dict2 but not in dict1
    for key in keys2 - keys1:
        print(f"Key '{key}' found in second dict but not in first at path '{path}'")
    
    # Recursively compare values
    for key in keys1 & keys2:
        new_path = f"{path}.{key}" if path else key
        value1 = dict1[key]
        value2 = dict2[key]
        
        # If values are both dictionaries, recursively compare
        if isinstance(value1, dict) and isinstance(value2, dict):
            compare_dicts(value1, value2, new_path)
        # Otherwise, check for equality
        elif value1 != value2:
            print(f"Different values at path '{new_path}': {value1} != {value2}")

def main():
    """Run a trained policy from robomimic with Isaac Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg("Isaac-Elevator-Franka-v0", use_gpu=not args_cli.cpu, num_envs=1)
    # modify configuration
    modify_cfg_to_robomimic(env_cfg)
    if args_cli.task.lower() == "pushbtn":
        modify_cfg_to_task_push_btn(env_cfg)
    elif args_cli.task.lower() == "movetobtn":
        modify_cfg_to_task_move_to_btn(env_cfg)
    else:
        raise NotImplementedError(f"Task {args_cli.task} not implemented.")
    env_cfg.env.episode_length_s = 15.

    policy_config_update = dict(
        algo=dict(
         rollout=dict(
            temporal_ensemble=True
         )   
        )
    )

    if args_cli.resume is not None:
        print("Resuming from a previous rollout.", args_cli.resume)
        rollout_log = args_cli.resume
        log_dir = os.path.dirname(rollout_log)
        _env_cfg = load_pickle(os.path.join(log_dir, "env.pkl"))
        # These two paths sometimes changes between http://omniverse-content-productionxxxxx and omniverse://localhost/N
        _env_cfg.robot.meta_info.usd_path = env_cfg.robot.meta_info.usd_path
        _env_cfg.marker.usd_path = env_cfg.marker.usd_path
        env_cfg = _env_cfg
        # Necessary configs for mimic
        env_cfg.observations.return_dict_obs_in_group = True
        env_cfg.control.substract_action_from_obs_frame = True
        with open(os.path.join(log_dir, 'rollout_log.txt'), 'r') as logf:
            args_cli.checkpoints = logf.readline().strip()
            done_files = [
                parse_log_line(line)[0] for line in logf.readlines()
            ]
        policy_config_update = load_yaml(os.path.join(log_dir, "policy_cfg_update.yaml"))
        
    else:
        log_dir = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir = os.path.join(os.path.dirname(os.path.dirname(args_cli.checkpoints)),"logs", "rollout", log_dir)
        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)
        dump_pickle(os.path.join(log_dir, "env.pkl"), env_cfg)
        dump_yaml(os.path.join(log_dir, "policy_cfg_update.yaml"), policy_config_update)
        done_files = []
        with open(os.path.join(log_dir, 'rollout_log.txt'), 'a') as logf:
            logf.write(f"{args_cli.checkpoints}\n")

    # create environment
    env = myEnvGym(f"Isaac-Elevator-Franka-v0", cfg= env_cfg, headless= False)

    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    data_logger = DataLogger(log_dir, log_tb=True)

    allfiles = [
        (int(re.search(args_cli.checkpoints.replace("__e__", "(\d+)"), f).group(1)), f)
        for f in glob(args_cli.checkpoints.replace("__e__", "*"))    
    ]
    allfiles.sort()
    allfiles = [f for f in allfiles if f[0] not in done_files]

    for epoch, f in allfiles:
        print("Loading checkpoint", f)
        # restore policy
        policy = RobomimicWrapper(f, policy_config_update, device)

        # reset environment
        obs_dict = env.reset()

        # create environment
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "video", f"e_{epoch}"),
            "step_trigger": lambda step: step == 0,
            "video_length": 600,
        }
        env.wrap_with_gym_recorder(video_kwargs)

        # robomimic only cares about policy observations
        # print("Observation",{k: (v.shape, v.shape) for k, v in obs.items()})
        # simulate environment
        num_episodes = 50
        all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
            policy=policy,
            envs={"orbit": env},
            horizon=int(env_cfg.env.episode_length_s*10), # cannot be more than max_len of the iad algorithm
            use_goals=False,
            num_episodes=num_episodes,
            render=False,
            video_dir=None,
            epoch=epoch, 
            video_skip=5,
            terminate_on_success=True,
        )

        for env_name in all_rollout_logs:
            rollout_logs = all_rollout_logs[env_name]
            for k, v in rollout_logs.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Rollout_{env_name}_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Rollout/{k}/{env_name}", v, epoch, log_stats=True)
        
        data_logger.flush()
        with open(os.path.join(log_dir, 'rollout_log.txt'), 'a') as logf:
            stamp = datetime.now().strftime("%b%d_%H-%M-%S")
            logf.write(f"{epoch}: {stamp}\n")

    # close the simulator
    data_logger.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
