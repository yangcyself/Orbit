# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained policy from robomimic."""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--task", type=str, default="pushbtn", help="Name of the task: can be pushbtn or movetobtn")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
args_cli = parser.parse_args()
# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import gym
import torch

import robomimic  # noqa: F401
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg
from utils.mimic_utils import RobomimicWrapper
from utils.env_presets import modify_cfg_to_robomimic, modify_cfg_according_to_task

def main():
    """Run a trained policy from robomimic with Isaac Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg("Isaac-Elevator-Franka-v0", use_gpu=not args_cli.cpu, num_envs=1)
    # modify configuration
    # env_cfg.control.control_type = "inverse_kinematics"
    # env_cfg.control.inverse_kinematics.command_type = "pose_rel"
    modify_cfg_according_to_task(env_cfg, args_cli.task)
    modify_cfg_to_robomimic(env_cfg)
    env_cfg.env.episode_length_s = 10.0
    policy_config_update = dict(
        algo=dict(
         rollout=dict(
            temporal_ensemble=True
         )   
        )
    )

    # create environment
    env = gym.make("Isaac-Elevator-Franka-v0", cfg=env_cfg, headless=False)

    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    policy = RobomimicWrapper(
        checkpoint = args_cli.checkpoint, 
        config_update = policy_config_update, 
        device = device, 
        verbose = False
    )
    # reset environment
    obs_dict = env.reset()
    policy.start_episode()
    # simulate environment
    while simulation_app.is_running():
        # compute actions
        actions = policy(obs_dict)
        obs_dict, _, done, info = env.step(actions)
        # check if simulator is stopped
        if env.unwrapped.sim.is_stopped():
            break
        if done.any():
            policy.start_episode()
    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
