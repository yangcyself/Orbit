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
parser.add_argument("--device", type=str, default="gamepad", help="Device for interacting with environment")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
args_cli = parser.parse_args()
args_cli.task = "Isaac-Elevator-Franka-v0"
# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import gym
import torch
import sys
import carb

from omni.isaac.orbit.devices import Se3Keyboard, Se3Gamepad

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg
from utils.env_presets import (modify_cfg_according_to_button_panel, 
            modify_cfg_according_to_button_task, 
            modify_cfg_according_to_task)


CHECK_SAME_OBS_REWARD = False
PRINT_REWARD_BREAKDOWN = False

def pre_process_actions(cmd: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # resolve gripper command
    base_dpos = torch.zeros(cmd.shape[0], 6, device=cmd.device)
    delta_pose = torch.zeros(cmd.shape[0], 6, device=cmd.device)
    delta_pose[:, 0] = 2*cmd[:, 2] # tool x
    base_dpos[:, 5] = - cmd[:, 1] # base yaw
    delta_pose[:, 2] = 2*cmd[:, 0] # tool z
    delta_pose[:, 1] = -2*cmd[:, 5] # tool y
    base_dpos[:, 2] = cmd[:, 6] # base z
    base_dpos[:, 0] = - 5* cmd[:, 4] # base x
    base_dpos[:, 1] = 5* cmd[:, 3] # base y
    gripper_vel = torch.zeros(cmd.shape[0], 1, device=delta_pose.device)
    gripper_vel[:] = -1.0 if gripper_command else 1.0
    # compute actions
    return torch.concat([base_dpos, delta_pose, gripper_vel], dim=1)


def update_terminal_table(data_dict, num_prev_lines=0):
    """
    Updates a table on the terminal with the provided data.
    
    :param data_dict: Dictionary where keys are labels and values are the data.
    :param num_prev_lines: The number of lines in the table from the previous call.
    :return: The number of lines printed in this call.
    """
    
    # Move cursor up for the number of previously printed lines
    for _ in range(num_prev_lines):
        sys.stdout.write('\033[F')
        sys.stdout.flush()
    
    # Print the current lines
    lines_printed = 0
    for label, value in data_dict.items():
        print(f"\r{label}: {value}", end='', flush=True)
        sys.stdout.write('\n')
        sys.stdout.flush()
        lines_printed += 1
    
    return lines_printed


def main():
    """Running keyboard teleoperation with Orbit manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    modify_cfg_according_to_button_panel(env_cfg, "panel01")
    modify_cfg_according_to_button_task(env_cfg, "in_right", "movetobtn")
    env_cfg.initialization.robot.position_cat = "see-point"
    # modify_cfg_according_to_task(env_cfg, "pushbtn")
    # modify configuration
    env_cfg.control.control_type = "inverse_kinematics"
    env_cfg.terminations.episode_timeout = False
    env_cfg.terminations.is_success = False
    env_cfg.terminations.collision = False
    env_cfg.observations.low_dim.enable_corruption=False
    if CHECK_SAME_OBS_REWARD:
        env_cfg.observation_grouping = {"privilege":None, "low_dim":None, "debug": None}
        env_cfg.observations.return_dict_obs_in_group = True
        env_cfg.num_envs = 2
        # disable any randomness in initialization
        env_cfg.initialization.robot.position_cat = "default"
        env_cfg.initialization.elevator.moving_elevator_prob = -1 #set to -1 for close door, set to 2 for wait elevator
        env_cfg.initialization.elevator.nonzero_floor_prob = -1
        env_cfg.initialization.elevator.max_init_wait_time = 0
        env_cfg.initialization.scene.obs_frame_bias_range = [0,0,0]
    else:
        env_cfg.observation_grouping = {"rgb":None,"privilege":None, "low_dim":None, "debug": None, "semantic":None}
        env_cfg.initialization.elevator.moving_elevator_prob = -1
        env_cfg.initialization.elevator.nonzero_floor_prob = -1
        env_cfg.initialization.elevator.max_init_wait_time = 0
        env_cfg.initialization.elevator.max_init_floor = 20
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        carb.log_warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
    elif args_cli.device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.2 * args_cli.sensitivity,
            rot_sensitivity=0.4 * args_cli.sensitivity,
        )
        teleop_interface.add_callback("L", env.reset)
    else:
        raise ValueError(f"Invalid device interface '{args_cli.device}'. Supported: 'keyboard', 'gamepad'.")
    # add teleoperation key for env reset
    # print helper for keyboard
    print("teleop:", teleop_interface)


    # reset environment
    env.reset()
    teleop_interface.reset()
    # teleop_interface_arm.reset()

    episodic_rewards = {k:v.clone() for k,v in env.reward_manager.episode_sums.items()}
    step_rewards = {k:0 for k,v in env.reward_manager.episode_sums.items()}

    num_prev_lines = 0
    # simulate environment
    while simulation_app.is_running():
        # get keyboard command
        cmd, gripper_command = teleop_interface.advance()

        # convert to torch
        cmd = torch.tensor(cmd, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
        # pre-process actions
        actions = pre_process_actions(cmd, gripper_command)
        # apply actions
        obs, rew, downs, info = env.step(actions)

        for k,v in env.reward_manager.episode_sums.items():
            step_rewards[k] = v - episodic_rewards[k]
            episodic_rewards[k] = v.clone()

        print_info_dict = {
            "pos": obs["privilege"]["dof_pos"][0,:4].cpu().numpy()
        }
        
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
            print_info_dict.update({"elevator_state": obs["privilege"]["elevator_state"]})
            print_info_dict.update({"elevator_is_zerofloor": obs["privilege"]["elevator_is_zerofloor"]})
            print_info_dict.update(step_rewards)

        num_prev_lines = update_terminal_table(print_info_dict, num_prev_lines)
    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
