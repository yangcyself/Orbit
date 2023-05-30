# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Orbit manipulation environments."""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="keyboard+gamepad", help="Device for interacting with environment")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
args_cli = parser.parse_args()
args_cli.task = "Isaac-Elevator-Franka-v0"
# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import gym
import torch

import carb

from omni.isaac.orbit.devices import Se2Keyboard, Se3Gamepad

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg


CHECK_SAME_OBS_REWARD = False
PRINT_REWARD_BREAKDOWN = True

def pre_process_actions(base_vel: torch.Tensor, delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # resolve gripper command
    gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
    gripper_vel[:] = -1.0 if gripper_command else 1.0
    # compute actions
    return torch.concat([base_vel, delta_pose, gripper_vel], dim=1)


def main():
    """Running keyboard teleoperation with Orbit manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    # modify configuration
    env_cfg.control.control_type = "inverse_kinematics"
    env_cfg.control.inverse_kinematics.command_type = "pose_rel"
    env_cfg.terminations.episode_timeout = False
    env_cfg.terminations.is_success = False
    if CHECK_SAME_OBS_REWARD:
        env_cfg.observation_grouping = {"privilege":None}
        env_cfg.observations.return_dict_obs_in_group = True
        env_cfg.num_envs = 2
        # disable any randomness in initialization
        env_cfg.initialization.robot.position_cat = "default"
        env_cfg.initialization.elevator.moving_elevator_prob = -1 #set to -1 for close door, set to 2 for wait elevator
        env_cfg.initialization.elevator.nonzero_floor_prob = -1
        env_cfg.initialization.elevator.max_init_wait_time = 0
    else:
        env_cfg.observation_grouping = {"rgb":None,"privilege":None}
        env_cfg.initialization.elevator.moving_elevator_prob = -1
        env_cfg.initialization.elevator.nonzero_floor_prob = -1
        env_cfg.initialization.elevator.max_init_wait_time = 0
        env_cfg.initialization.elevator.max_init_floor = 2
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        carb.log_warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.device.lower() == "keyboard+gamepad":
        teleop_interface_arm = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
        teleop_interface_base = Se2Keyboard(
            v_x_sensitivity=0.2 * args_cli.sensitivity,
            v_y_sensitivity=0.2 * args_cli.sensitivity,
            omega_z_sensitivity=0.2 * args_cli.sensitivity,
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    teleop_interface_base.add_callback("L", env.reset)
    # print helper for keyboard
    print("Base teleop:", teleop_interface_base)
    print("Arm teleop:", teleop_interface_arm)

    # reset environment
    env.reset()
    teleop_interface_base.reset()
    teleop_interface_arm.reset()

    episodic_rewards = {k:v.clone() for k,v in env.reward_manager.episode_sums.items()}
    step_rewards = {k:0 for k,v in env.reward_manager.episode_sums.items()}

    # simulate environment
    while simulation_app.is_running():
        # get keyboard command
        base_cmd = teleop_interface_base.advance()
        delta_pose, gripper_command = teleop_interface_arm.advance()
        # convert to torch
        base_cmd = torch.tensor(base_cmd, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
        delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
        # pre-process actions
        actions = pre_process_actions(base_cmd, delta_pose, gripper_command)
        # apply actions
        obs, rew, downs, info = env.step(actions)

        for k,v in env.reward_manager.episode_sums.items():
            step_rewards[k] = v - episodic_rewards[k]
            episodic_rewards[k] = v.clone()

        if(CHECK_SAME_OBS_REWARD):
            for kk,vv in obs.items():
                for k,v in vv.items():
                    if((v[0,...] - v[1,...]).abs().max()>1e-3):
                        print(f"{k} observation different\n{v[0,...]} \t {v[1,...]}")
            if(abs(rew[0] - rew[1])>1e-5):
                print(f"reward different {rew[0]} \t {rew[1]}")
            for k,v in step_rewards.items():
                if(abs(v[0] - v[1])>1e-5):
                    print(f"\t{k}: {v[0]}\t{v[1]}")
        # check if simulator is stopped
        if env.unwrapped.sim.is_stopped():
            break
        
        if (PRINT_REWARD_BREAKDOWN):
            print("#### elevator_state", obs["privilege"]["elevator_state"])
            print("#### elevator_is_zerofloor", obs["privilege"]["elevator_is_zerofloor"])
            for k,v in step_rewards.items():
                print(f"{k}: ", v)

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
