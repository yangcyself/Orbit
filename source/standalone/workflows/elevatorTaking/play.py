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
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
args_cli = parser.parse_args()
args_cli.task = "Isaac-Elevator-Franka-v0"
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
from utils.mimic_utils import RobomimicWrapper, myEnvGym


def main():
    """Run a trained policy from robomimic with Isaac Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=1)
    # modify configuration
    # env_cfg.control.control_type = "inverse_kinematics"
    # env_cfg.control.inverse_kinematics.command_type = "pose_rel"
    env_cfg.env.episode_length_s = 2.0
    env_cfg.terminations.episode_timeout = True
    env_cfg.terminations.is_success = "pushed_btn"
    env_cfg.terminations.collision = False
    env_cfg.observations.return_dict_obs_in_group = True
    env_cfg.control.substract_action_from_obs_frame = True
    env_cfg.control.control_type = "default"
    env_cfg.observation_grouping = {"policy":"privilege", "rgb":None, "low_dim":None, "goal":["goal","goal_lowdim"]}
    env_cfg.initialization.robot.position_cat = "uniform"
    env_cfg.initialization.elevator.moving_elevator_prob = -1
    env_cfg.initialization.elevator.nonzero_floor_prob = 1



    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=False)

    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    policy = RobomimicWrapper(checkpoint = args_cli.checkpoint, device = device, verbose = False)
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
