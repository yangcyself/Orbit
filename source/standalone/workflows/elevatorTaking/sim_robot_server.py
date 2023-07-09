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
from utils.mimic_utils import RobomimicWrapper
from utils.env_presets import modify_cfg_to_robomimic, modify_cfg_to_task_push_btn

import socket
import time
import threading
import copy
from utils.tcp_utils import send_data, recv_data, Racedata, print_warning, print_error
from utils.myTypes import myNumpyArray, myStr, myInt, myFloat

class tcpRecipient(threading.Thread):
    def __init__(self, conn, race_data):
        threading.Thread.__init__(self)
        self.conn = conn
        self.race_data = race_data

        self.signatures = {
            "get_value": (myStr,),
            "set_value": (myStr, myNumpyArray),
            "set_value_ref": (myStr, myStr),
            "del_value": (myStr,),
            "exec_command": (myStr, myInt), # command, count
        }

    def run(self):
        while True:
            data = recv_data(self.conn)
            if data is None:
                print("No more data. Closing connection...")
                break
            cmd, args = self.parse_data(data)
            try:
                getattr(self, cmd)(*args)
            except Exception as e:
                print_error(f"Error when executing command {cmd}: {e}")
        self.conn.close()

    def parse_data(self, data):
        """Parse the data received from the client."""
        cmd, l = myStr.from_buffer(data)
        data = data[l:]
        args = []
        for sig_t in self.signatures[cmd]:
            arg, l = sig_t.from_buffer(data)
            data = data[l:]
            args.append(arg)
        return cmd, args

    def get_value(self, name):
        # get value from race data
        value = self.race_data.getdata(name)
        # send value to client
        send_data(self.conn, myNumpyArray(value).to_bytes())

    def set_value(self, name, value):
        self.race_data.setdata(name, value)

    def set_value_ref(self, name, ref_name):
        self.race_data.setdataref(name, ref_name)

    def del_value(self, name):
        self.race_data.deldata(name)

    def exec_command(self, cmd, count):
        self.race_data.setcommand(cmd, count)


class RobotActionBase:
    """The base class for robot actions."""
    def __init__(self):
        pass
    
    def __call__(self, obs_dict):
        """Get the action from the observation."""
        raise NotImplementedError

class RobotActionMoveto(RobotActionBase):
    """The action to move the robot to a target position."""
    def __init__(self, target_pos):
        self.target_pos = torch.tensor(target_pos)
        print(self.target_pos)
    
    def __call__(self, obs_dict):
        """Move the robot to the target"""
        res = torch.zeros(1,10)
        res[0,:2] = self.target_pos[:2]
        return res

class RobotActionPushbtn(RobotActionBase):
    """The action to push the button."""
    def __init__(self, race_data, checkpoint, cfg, device):
        self.race_data = race_data
        self.policy = RobomimicWrapper(
            checkpoint = checkpoint, 
            config_update = cfg, 
            device = device, 
            verbose = False
        )
        self.policy.start_episode()
    
    def __call__(self, obs_dict):
        """Push the button."""
        goal_dict = {
          k : torch.tensor(self.race_data.getdata(f"pushbtn_{k}"))
          for k in [
            "goal_dof_pos", 
            "goal_base_rgb",
            "goal_base_semantic",
            "goal_hand_rgb",
            "goal_hand_semantic"
          ]  
        }
        obs_dict.update({"goal": goal_dict})
        actions = self.policy(obs_dict)
        return actions

class RobotActorServer:
    """The server that provide robot high-level control interface.
    It waits tcp commands and get actions from the policy.
    """
    def __init__(self, env, policycfgs, port, device=None):
        self.env = env 
        self.policycfgs = policycfgs
        self.port = port
        self.race_data = Racedata()
        self.device = device
        
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(('localhost', port))
        server.listen(1)

        print("Server started. Waiting for connections...")
        conn, addr = server.accept()
        self.conn = conn
        self.addr = addr
        print(f"Connection from {addr}")

        self.tcp_recipient = tcpRecipient(conn, self.race_data)
        self.tcp_recipient.start()
        self.current_cmd = None
        self.current_action: RobotActionBase = None

    def actionFactory(self, action_type):
        if action_type == "moveto":
            target_pos = self.race_data.getdata("moveto_target_pos")
            return RobotActionMoveto(target_pos)
        elif action_type == "pushbtn":
            checkpoint = self.policycfgs["pushbtn"]["checkpoint"]
            cfg = self.policycfgs["pushbtn"]["cfg"]
            device = self.device
            return RobotActionPushbtn(self.race_data, checkpoint, cfg, device)
        else:
            raise NotImplementedError

    def step(self, obs_dict):
        """Get actions from the policy."""
        cmd, count = self.race_data.getcommand()
        for kk, vv in obs_dict.items():
            for k,v in vv.items():
                self.race_data.setdata("obs/"+kk+"/"+k, v.cpu().numpy())
        
        if cmd is not None and cmd != self.current_cmd:
            print("executing new command:", cmd)
            self.current_cmd = cmd
            self.current_action = self.actionFactory(cmd)
        if self.current_action is not None:
            action = self.current_action(obs_dict)
            count = self.race_data.command_count_dec()
            if count <= 0:
                self.current_action = None
                self.current_cmd = None
            return action

        return self.env.zero_action()


def main():
    """Run a trained policy from robomimic with Isaac Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=1)

    modify_cfg_to_task_push_btn(env_cfg)
    modify_cfg_to_robomimic(env_cfg)
    env_cfg.terminations.episode_timeout = False
    env_cfg.observation_grouping.update({"debug":None})
    del env_cfg.observation_grouping["goal"]
    action_cfgs = {
        "moveto": {},
        "pushbtn": {
            "checkpoint": args_cli.checkpoint,
            "cfg": {}
        }
    }

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=False)
    # reset environment
    obs_dict = env.reset()

    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    server = RobotActorServer(env, action_cfgs, 12345)
    # simulate environment
    while simulation_app.is_running():
        # compute actions
        actions = server.step(obs_dict)
        # actions = policy(obs_dict)
        obs_dict, _, done, info = env.step(actions)
        # check if simulator is stopped
        if env.unwrapped.sim.is_stopped():
            break

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
