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
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml
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

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        return {
            "task": self.env.is_success()["task"].cpu().numpy(),
            "pushed_btn": self.env.is_success()["pushed_btn"].cpu().numpy()
        }
    
def main():
    """Run a trained policy from robomimic with Isaac Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=1)
    # modify configuration
    # env_cfg.control.control_type = "inverse_kinematics"
    # env_cfg.control.inverse_kinematics.command_type = "pose_rel"
    env_cfg.env.episode_length_s = 30.0
    env_cfg.terminations.episode_timeout = True
    env_cfg.terminations.is_success = True
    env_cfg.terminations.collision = False
    env_cfg.observations.return_dict_obs_in_group = True
    env_cfg.control.control_type = "ohneHand"
    env_cfg.observation_grouping = {"policy":"privilege", "rgb":None}
    env_cfg.initialization.robot.position_cat = "uniform"
    env_cfg.initialization.elevator.moving_elevator_prob = 0.4
    env_cfg.initialization.elevator.nonzero_floor_prob = 1

    # create environment
    env = myEnvGym(args_cli.task, cfg= env_cfg, headless= args_cli.headless)
    
    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    log_dir = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(os.path.dirname(os.path.dirname(args_cli.checkpoints)),"logs", "rollout", log_dir)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "env.pkl"), env_cfg)

    data_logger = DataLogger(log_dir, log_tb=True)

    allfiles = [
        (int(re.search(args_cli.checkpoints.replace("__e__", "(\d+)"), f).group(1)), f)
        for f in glob(args_cli.checkpoints.replace("__e__", "*"))    
    ]
    allfiles.sort()

    for epoch, f in allfiles:
        print("Loading checkpoint", f)
        # restore policy
        policy = RobomimicWrapper(f, device)

        # reset environment
        obs_dict = env.reset()
        # robomimic only cares about policy observations
        # print("Observation",{k: (v.shape, v.shape) for k, v in obs.items()})
        # simulate environment
        num_episodes = 100
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
        
        with open(os.path.join(log_dir, 'rollout_log.txt'), 'a') as logf:
            stamp = datetime.now().strftime("%b%d_%H-%M-%S")
            logf.write(f"{epoch}: {stamp}\n")

    # close the simulator
    simulation_app.close()


if __name__ == "__main__":
    main()
