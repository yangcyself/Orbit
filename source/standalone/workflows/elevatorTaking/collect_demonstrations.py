# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Orbit environments."""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="keyboard+gamepad", help="Device for interacting with environment")
parser.add_argument("--num_demos", type=int, default=1, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
args_cli = parser.parse_args()
args_cli.task = "Isaac-Elevator-Franka-v0"
# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import contextlib
import gym
import os
import torch

from omni.isaac.orbit.devices import Se2Keyboard, Se3Gamepad
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils.data_collector import RobomimicDataCollector
from omni.isaac.orbit_envs.utils.parse_cfg import parse_env_cfg


def pre_process_actions(base_vel: torch.Tensor, delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # resolve gripper command
    gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
    gripper_vel[:] = -1.0 if gripper_command else 1.0
    # compute actions
    return torch.concat([base_vel, delta_pose, gripper_vel], dim=1)


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    # modify configuration
    env_cfg.control.control_type = "inverse_kinematics"
    env_cfg.control.inverse_kinematics.command_type = "pose_rel"
    env_cfg.terminations.episode_timeout = False
    env_cfg.terminations.is_success = True
    env_cfg.observations.return_dict_obs_in_group = True

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)

    # create controller
    if args_cli.device.lower() == "keyboard+gamepad":
        teleop_interface_arm = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.5 * args_cli.sensitivity
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
    # print helper
    print("Base teleop:", teleop_interface_base)
    print("Arm teleop:", teleop_interface_arm)


    # specify directory for logging experiments
    log_dir = os.path.join("./logs/robomimic", args_cli.task)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
        flush_freq=env.num_envs,
        env_config={"device": args_cli.device},
    )

    # reset environment
    obs_dict = env.reset()
    # print({k: v.keys() for k, v in obs_dict.items()})
    # robomimic only cares about policy observations
    # obs = obs_dict["policy"]
    obs = {k:v for kk,vv in obs_dict.items() for k,v in vv.items()}
    # reset interfaces
    teleop_interface_base.reset()
    teleop_interface_arm.reset()
    collector_interface.reset()

    # simulate environment
    with contextlib.suppress(KeyboardInterrupt):
        while not collector_interface.is_stopped():
            # get keyboard command
            base_cmd = teleop_interface_base.advance()
            delta_pose, gripper_command = teleop_interface_arm.advance()
            # convert to torch
            base_cmd = torch.tensor(base_cmd, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            # compute actions based on environment
            actions = pre_process_actions(base_cmd, delta_pose, gripper_command)

            # TODO: Deal with the case when reset is triggered by teleoperation device.
            #   The observations need to be recollected.
            # store signals before stepping
            # -- obs
            for key, value in obs.items():
                collector_interface.add(f"obs/{key}", value)
            # -- actions
            collector_interface.add("actions", actions)
            # perform action on environment
            obs_dict, rewards, dones, info = env.step(actions)
            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break
            # robomimic only cares about policy observations
            # obs = obs_dict["policy"]
            obs = {k:v for kk,vv in obs_dict.items() for k,v in vv.items()}
            # store signals from the environment
            # -- next_obs
            for key, value in obs.items():
                collector_interface.add(f"next_obs/{key}", value.cpu().numpy())
            # -- rewards
            collector_interface.add("rewards", rewards)
            # -- dones
            collector_interface.add("dones", dones)
            # -- is-success label
            try:
                collector_interface.add("success", info["is_success"])
            except KeyError:
                raise RuntimeError(
                    f"Only goal-conditioned environment supported. No attribute named 'is_success' found in {list(info.keys())}."
                )
            # flush data from collector for successful environments
            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            collector_interface.flush(reset_env_ids)

    # close the simulator
    collector_interface.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
