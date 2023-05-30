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
parser.add_argument("--checkpoints", type=str, default=None, help="the pytorch checkpoints to load. Use __e__ as a placeholder for epoch number.")
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
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_gym import EnvGym

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg
from robomimic.utils.log_utils import DataLogger
from glob import glob
import re
import os



class RobomimicWrapper(RolloutPolicy):
    """The Wrapper of the RolloffPolicy
    """
    def __init__(self, checkpoint, device):
        self.device = device
        self.policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint, device=device, verbose=True)

    def start_episode(self):
        self.policy.start_episode()
    
    def _prepare_observation(self, ob):
        self.policy._prepare_observation(ob)

    def __call__(self, ob, goal=None):
        obs = {f"{kk}:{k}":v[0] for kk,vv in ob.items() for k,v in vv.items()}
        obs["rgb:hand_camera_rgb"] = obs["rgb:hand_camera_rgb"].permute(2, 0, 1)
        return torch.tensor(self.policy(obs)).to(self.device)[None,...]

class myEnvGym(EnvGym):
    def get_observation(self, obs=None):
        """
        Get current environment observation dictionary.

        Args:
            ob (np.array): current flat observation vector to wrap and provide as a dictionary.
                If not provided, uses self._current_obs.
        """
        return obs
    
    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, reward, done, info = self.env.step(action)
        self._current_obs = obs
        self._current_reward = reward
        self._current_done = done
        return self.get_observation(obs), reward.detach().cpu().numpy(), self.is_done(), info


    
def main():
    """Run a trained policy from robomimic with Isaac Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=1)
    # modify configuration
    # env_cfg.control.control_type = "inverse_kinematics"
    # env_cfg.control.inverse_kinematics.command_type = "pose_rel"
    env_cfg.env.episode_length_s = 6.0
    env_cfg.terminations.episode_timeout = True
    env_cfg.terminations.is_success = True
    env_cfg.terminations.collision = False
    env_cfg.observations.return_dict_obs_in_group = True
    env_cfg.control.control_type = "default"
    env_cfg.observation_grouping = {"policy":"privilege", "rgb":None}

    # create environment
    env = myEnvGym(args_cli.task, cfg= env_cfg, headless= args_cli.headless)
    

    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    log_dir = os.path.join(os.path.dirname(os.path.dirname(args_cli.checkpoints)),"logs")
    data_logger = DataLogger(log_dir, log_tb=True)

    for f in glob(args_cli.checkpoints.replace("__e__", "*")):
        m = re.search(args_cli.checkpoints.replace("__e__", "(\d+)"), f)
        if not m:
            continue
        print("Loading checkpoint", f)
        epoch = int(m.group(1))

        # restore policy
        policy = RobomimicWrapper(f, device)

        # reset environment
        obs_dict = env.reset()
        # robomimic only cares about policy observations
        # print("Observation",{k: (v.shape, v.shape) for k, v in obs.items()})
        # simulate environment
        num_episodes = 500
        all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
            policy=policy,
            envs={"orbit": env},
            horizon=1000,
            use_goals=False,
            num_episodes=num_episodes,
            render=False,
            video_dir=None,
            epoch=epoch, # TODO
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

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
