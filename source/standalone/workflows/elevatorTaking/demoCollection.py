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
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--prefix", type=str, default=".", help="Save dataset to <prefix>/logs/rolloutCollection.")
parser.add_argument("--notCollectDemonstration", action="store_true", default=False, help="Do not collect demonstration.")
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
from datetime import datetime

from omni.isaac.orbit.devices import Se2Keyboard, Se3Gamepad
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils.data_collector import RobomimicDataCollector
from omni.isaac.orbit_envs.utils.parse_cfg import parse_env_cfg

from rsl_rl.runners import OnPolicyRunner

# For RSL-RL
from rslrl_config import parse_rslrl_cfg
from omni.isaac.orbit_envs.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.orbit_envs.utils.wrappers.rsl_rl import RslRlVecEnvWrapper, export_policy_as_onnx

# Default arguments for actor wrappers
ACTOR_CONFIGS = {
    "rslrl": {
        "checkpoint": None
    },
    "human":{
        "device": "keyboard+gamepad",
        "sensitivity": 1.0
    }
}

# A simple configuration for rollout collection
EXP_CONFIGS = {
    "actor_type": "rslrl",
    "collect_demonstration": True,
    "wrapper_cfg": None,
    "collect_extra_info": True,
    "num_demos": args_cli.num_demos,
    "apply_action_noise": 0.1 # the noise to be applied on action, in order to generate diverse trajectory
}


## 
# Remote Control
##

def pre_process_actions(base_vel: torch.Tensor, delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # resolve gripper command
    gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
    gripper_vel[:] = -1.0 if gripper_command else 1.0
    # compute actions
    return torch.concat([base_vel, delta_pose, gripper_vel], dim=1)

##
# RSL-RL Rollout
## 

def dict_to_tensor(tensorDict, names, concatdim=1):
    tensors = [tensorDict[n] for n in names]
    return torch.cat(tensors, concatdim)

class ActorWrapperBase:
    """
    Wrapper for human or machine agent that provide actions
    """
    def get_action(self, obs):
        raise NotImplementedError


class RslRlActor(ActorWrapperBase):
    """
    Wrapper for RSL-RL agent that provide actions
    """
    def __init__(self, env, task, checkpoint):
        agent_cfg = parse_rslrl_cfg(task)
        resume_path = os.path.abspath(checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        rslenv = RslRlVecEnvWrapper(env)
        self.ppo_runner = OnPolicyRunner(rslenv, agent_cfg, log_dir=None, device=agent_cfg["device"])
        self.ppo_runner.load(resume_path)
        # obtain the trained policy for inference
        self.policy = self.ppo_runner.get_inference_policy(device=rslenv.unwrapped.device)

    def get_action(self, obs):
        # convert obs to tensor
        obs_rsl = dict_to_tensor(obs['policy'], ['dof_pos_normalized', 'dof_vel', 'ee_position', 'elevator_btn_pressed', 
           'elevator_is_zerofloor', 'elevator_state', 'elevator_waittime'])
        with torch.no_grad():
            action = self.policy(obs_rsl)
            return action

class HumanActor(ActorWrapperBase):
    """
    Wrapper for human agent that provide actions
    """
    def __init__(self, env, device="keyboard+gamepad", sensitivity=1.0):
        # create controller
        if device.lower() == "keyboard+gamepad":
            self.teleop_interface_arm = Se3Gamepad(
                pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.5 * sensitivity
            )
            self.teleop_interface_base = Se2Keyboard(
                v_x_sensitivity=0.2 * sensitivity,
                v_y_sensitivity=0.2 * sensitivity,
                omega_z_sensitivity=0.2 * sensitivity,
            )
        else:
            raise ValueError(f"Invalid device interface '{args_cli.device}'. Supported: 'keyboard', 'spacemouse'.")
        self.env = env
        # add teleoperation key for env reset
        self.teleop_interface_base.add_callback("L", env.reset)
        # print helper
        print("Base teleop:", self.teleop_interface_base)
        print("Arm teleop:", self.teleop_interface_arm)
        self.teleop_interface_base.reset()
        self.teleop_interface_arm.reset()

    def get_action(self, obs):
        # get actions from teleop interfaces
        base_cmd = self.teleop_interface_base.advance()
        delta_pose, gripper_command = self.teleop_interface_arm.advance()
        # convert to torch
        base_cmd = torch.tensor(base_cmd, dtype=torch.float, device=self.env.device).repeat(self.env.num_envs, 1)
        delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=self.env.device).repeat(self.env.num_envs, 1)
        # compute actions based on environment
        return pre_process_actions(base_cmd, delta_pose, gripper_command)
        


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    # modify configuration
    env_cfg.env.episode_length_s = 5.
    env_cfg.initialization.elevator.max_init_floor = 5 # wait for at most 5 seconds
    env_cfg.initialization.elevator.moving_elevator_prob = 0 # wait for at most 5 seconds
    env_cfg.initialization.elevator.nonzero_floor_prob = 1 # wait for at most 5 seconds
    env_cfg.initialization.robot.position_uniform_min = [1.4, 0.9, -1.6]  # position (x,y,z)
    env_cfg.initialization.robot.position_uniform_max = [1.6, 1.1, -1.4]  # position (x,y,z)
    
    env_cfg.terminations.episode_timeout = True
    env_cfg.terminations.is_success = "pushed_btn"
    env_cfg.terminations.collision = True
    env_cfg.observations.return_dict_obs_in_group = True
    env_cfg.observation_grouping = {"policy":"privilege", "rgb":None}
    EXP_CONFIGS["wrapper_cfg"] = ACTOR_CONFIGS[EXP_CONFIGS["actor_type"]]
    if(EXP_CONFIGS["actor_type"] == "human"):    
        # Set wrapper config
        EXP_CONFIGS["wrapper_cfg"]["device"] = args_cli.device
        EXP_CONFIGS["wrapper_cfg"]["sensitivity"] = args_cli.sensitivity
        # Set env config
        env_cfg.control.control_type = "inverse_kinematics"
        env_cfg.control.inverse_kinematics.command_type = "pose_rel"
    elif(EXP_CONFIGS["actor_type"] == "rslrl"):
        # Set wrapper config
        EXP_CONFIGS["wrapper_cfg"]["checkpoint"] = args_cli.checkpoint
    else:
        raise ValueError(f"Invalid actor type '{EXP_CONFIGS['actor_type']}'. Supported: 'human', 'rslrl'.")

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)

    if(EXP_CONFIGS["actor_type"] == "human"):
        # create actor
        actor = HumanActor(env, device=args_cli.device, sensitivity=args_cli.sensitivity)
    elif(EXP_CONFIGS["actor_type"] == "rslrl"):
        # create actor
        actor = RslRlActor(env, args_cli.task, args_cli.checkpoint)
    
    # specify directory for logging experiments
    log_dir = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(args_cli.prefix, "logs/rolloutCollection", args_cli.task, log_dir)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "exp.yaml"), EXP_CONFIGS)


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
    # obs_dict = env.reset()
    obs = env.reset()
    obs_mimic = {f"{kk}:{k}":v for kk,vv in obs.items() for k,v in vv.items()}

    # # reset interfaces
    collector_interface.reset()

    # simulate environment
    with contextlib.suppress(KeyboardInterrupt):
        while not collector_interface.is_stopped():
            
            actions = actor.get_action(obs)

            # -- obs
            for key, value in obs_mimic.items():
                collector_interface.add(f"obs/{key}", value)
            
            # -- states
            collector_interface.add("states", env.get_state().cpu().numpy())

            if(EXP_CONFIGS["apply_action_noise"] is not None and EXP_CONFIGS["apply_action_noise"]):
                actions += EXP_CONFIGS["apply_action_noise"] * torch.randn_like(actions)

            # -- actions
            collector_interface.add("actions", actions)
            # perform action on environment
            obs, rewards, dones, info = env.step(actions)
            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break
            obs_mimic = {f"{kk}:{k}":v for kk,vv in obs.items() for k,v in vv.items()}

            # store signals from the environment
            # -- next_obs
            for key, value in obs_mimic.items():
                collector_interface.add(f"next_obs/{key}", value.cpu().numpy())
            # -- rewards
            collector_interface.add("rewards", rewards)
            # -- dones
            collector_interface.add("dones", dones)

            # -- is-success label
            try:
                success = info["is_success"]
                collector_interface.add("success", success)
            except KeyError:
                raise RuntimeError(
                    f"Only goal-conditioned environment supported. No attribute named 'is_success' found in {list(info.keys())}."
                )
            if EXP_CONFIGS["collect_extra_info"]:
                for k,v in info["episode"].items():
                    collector_interface.add(f"episode_info/{k}", v.reshape(success.shape))
            # flush data from collector for successful environments
            done_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            success_env_ids = success.nonzero(as_tuple=False).squeeze(-1)
            collector_interface.flush(success_env_ids)
            collector_interface.reset_buf_idx(done_env_ids)
            # Need to manully reset the environment, otherwise the "states" does not get updated before the next episode
            if len(done_env_ids) > 0:
                env.reset_idx(done_env_ids)
                env.reset_buf[done_env_ids] = 0.



    # close the simulator
    collector_interface.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
