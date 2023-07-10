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
parser.add_argument("--moveto_ckpt", type=str, default=None, help="Pytorch model ckpt to load for the task 'moveto'")
parser.add_argument("--pushbtn_ckpt", type=str, default=None, help="Pytorch model ckpt to load for the task 'pushbtn'")
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

import socket
import time
import threading
import copy
from utils.mimic_utils import RobomimicWrapper
from utils.env_presets import modify_cfg_to_simServer
from utils.tcp_utils import send_data, recv_data, Racedata, print_warning, print_error
from utils.myTypes import myNumpyArray, myStr, myInt, myFloat
from utils.task_configs import SimRobotServerCfg


class tcpRecipient(threading.Thread):
    def __init__(self, host, port, race_data):
        threading.Thread.__init__(self)
        self.host = host
        self.port = port
        self.race_data = race_data

        self.signatures = {
            "get_value": (myStr,),
            "set_value": (myStr, myNumpyArray),
            "set_value_ref": (myStr, myStr),
            "del_value": (myStr,),
            "exec_command": (myStr, myInt), # command, count
            "get_command": tuple(), # command, count
        }

    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(1)

        print("Server started.")
        while True:
            print("Waiting for connection...")
            conn, addr = server.accept()
            self.conn = conn
            self.addr = addr
            print(f"Connection from {addr}")

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
            self.race_data.reset() # reset the communication data once the connection is closed
            print("Connection closed.")

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

    def get_command(self):
        cmd, count = self.race_data.getcommand()
        if cmd is None:
            cmd = "none"
        send_data(self.conn, myStr(cmd).to_bytes()+myInt(count).to_bytes())


class RobotActionBase:
    """The base class for robot actions."""
    def __init__(self, race_data, device):
        self.race_data = race_data
        task_frame_shift = race_data.getdata("base_task_frame_shift")
        if task_frame_shift.ndim == 1: # add batch dim
            task_frame_shift = task_frame_shift.reshape(1,-1)
        if task_frame_shift.shape[1] == 10: # length of dof
            task_frame_shift = task_frame_shift[:,[0,1,3]] # x,y,yaw
        assert task_frame_shift.shape[1] == 3, "task_frame_shift should of shape 3"
        self.task_frame_shift = torch.tensor(task_frame_shift).to(device)
    
    def __call__(self, obs_dict):
        """Get the action from the observation."""
        raise NotImplementedError

    ##! Different from elevatorEnv, here we add the frame on to action and substruct from observation
    def task_frame_add(self, input_vec, px_idx=None, py_idx=None, pr_idx=None, vx_idx=None, vy_idx=None):
        """ Add the pose of the observation frame to the observation """
        x,y,r = self.task_frame_shift[:, 0], self.task_frame_shift[:, 1], self.task_frame_shift[:, 2]
        if(px_idx is not None and py_idx is not None):
            ##! the clones are necessary to avoid in-place operations
            px,py = input_vec[:, px_idx].clone(), input_vec[:, py_idx].clone()
            input_vec[:, px_idx] = x + px * torch.cos(r) - py * torch.sin(r)
            input_vec[:, py_idx] = y + px * torch.sin(r) + py * torch.cos(r)
        if(pr_idx is not None):
            input_vec[:, pr_idx] = r + input_vec[:, pr_idx]
            input_vec[:, pr_idx] = torch.atan2(torch.sin(input_vec[:, pr_idx]), torch.cos(input_vec[:, pr_idx]))
        if(vx_idx is not None and vy_idx is not None):
            ##! the clones are necessary to avoid in-place operations
            vx,vy = input_vec[:, vx_idx].clone(), input_vec[:, vy_idx].clone()
            input_vec[:, vx_idx] = vx * torch.cos(r) - vy * torch.sin(r)
            input_vec[:, vy_idx] = vx * torch.sin(r) + vy * torch.cos(r)

    def task_frame_subtract(self, input_vec, px_idx=None, py_idx=None, pr_idx=None, vx_idx=None, vy_idx=None):
        """Minus the pose of the observation frame to the observation """
        x,y,r = self.task_frame_shift[:, 0], self.task_frame_shift[:, 1], self.task_frame_shift[:, 2]
        if(px_idx is not None and py_idx is not None):
            px,py = input_vec[:, px_idx].clone(), input_vec[:, py_idx].clone()
            input_vec[:, px_idx] = (px - x) * torch.cos(r) + (py - y) * torch.sin(r)
            input_vec[:, py_idx] = -(px - x) * torch.sin(r) + (py - y) * torch.cos(r)
        if(pr_idx is not None):
            input_vec[:, pr_idx] = input_vec[:, pr_idx] - r
            input_vec[:, pr_idx] = torch.atan2(torch.sin(input_vec[:, pr_idx]), torch.cos(input_vec[:, pr_idx]))
        if(vx_idx is not None and vy_idx is not None):
            vx,vy = input_vec[:, vx_idx].clone(), input_vec[:, vy_idx].clone()
            input_vec[:, vx_idx] = vx * torch.cos(r) + vy * torch.sin(r)
            input_vec[:, vy_idx] = -vx * torch.sin(r) + vy * torch.cos(r)

class RobotActionMoveto(RobotActionBase):
    """The action to move the robot to a target position."""
    def __init__(self, race_data, device):
        super().__init__(race_data, device)
        target_pos = self.race_data.getdata("moveto_target_pos")
        self.target_pos = torch.tensor(target_pos)
    
    def __call__(self, obs_dict):
        """Move the robot to the target"""
        res = torch.zeros(1,10)
        res[:,:4] = self.target_pos[:, :4]
        self.task_frame_add(res, px_idx=0, py_idx=1, pr_idx=3) 
        return res

class RobotActionRoboMimicBase(RobotActionBase):
    """The action to push the button."""
    def __init__(self, race_data, cfg, device):
        super().__init__(race_data, device)
        self.policy = RobomimicWrapper(
            checkpoint = cfg.policy.checkpoint, 
            config_update = cfg.policy.mimic_update, 
            device = device, 
            verbose = False
        )
        self.cfg = cfg
        self.policy.start_episode()
        self.obscfg_dict = cfg.observations.to_dict()

    def process_obs(obs_dict):
        # Add goal observation
        goal_dict = {
          k : torch.tensor(self.race_data.getdata(f"mimic_{k}"))
          for k in [
            "goal_dof_pos", 
            "goal_base_rgb",
            "goal_base_semantic",
            "goal_hand_rgb",
            "goal_hand_semantic"
          ]  
        }
        obs_dict.update({"goal": goal_dict})

        # Task frame shift
        self.task_frame_substract(obs_dict["goal"]["goal_dof_pos"], px_idx=0, py_idx=1, pr_idx=3)
        self.task_frame_substract(obs_dict["low_dim"]["dof_pos_obsframe"], px_idx=0, py_idx=1, pr_idx=3)
        self.task_frame_substract(obs_dict["low_dim"]["dof_vel_obsframe"], vx_idx=0, vy_idx=1)
        self.task_frame_substract(obs_dict["low_dim"]["ee_position_obsframe"], px_idx=0, py_idx=1)

        # Observation normalization
        for kk,vv in obs_dict.items():
            for k,v in vv.items():
                arg = self.obscfg_dict.get(k, {})
                normalizer = arg.get("normalizer", None)
                if normalizer is None:
                    continue
                mean = normalizer["mean"].to(v)
                std = normalizer["std"].to(v)
                obs_dict[kk][k] = (obs_dict[kk][k]-mean)/std
                
    def __call__(self, obs_dict):
        """Push the button."""

        self.process_obs(obs_dict)
        with torch.no_grad():
            actions = self.policy(obs_dict)
        self.task_frame_add(actions, px_idx=0, py_idx=1, pr_idx=3)
        return actions

class RobotActionPushBtn(RobotActionRoboMimicBase):
    def __init__(self, race_data, cfg, device):
        super().__init__(race_data, cfg, device)
    # every thing are the same as the base

class RobotActionMovetoBtn(RobotActionRoboMimicBase):
    def __init__(self, race_data, cfg, device):
            super().__init__(race_data, cfg, device)

    def __call__(self, obs_dict):
        """Push the button."""
        self.process_obs(obs_dict)
        with torch.no_grad():
            actions = self.policy(obs_dict)
        self.task_frame_add(actions, px_idx=0, py_idx=1, pr_idx=3)
        actions = torch.cat([actions, torch.zeros((actions.shape[0], 6))], dim=1)
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
        
        self.tcp_recipient = tcpRecipient("localhost", port, self.race_data)
        self.tcp_recipient.start()
        self.current_cmd = None
        self.current_action: RobotActionBase = None

    def actionFactory(self, action_type):
        if action_type == "moveto":
            return RobotActionMoveto(target_pos)
        elif action_type == "pushBtn":
            cfg = self.policycfgs.pushBtn
            device = self.device
            return RobotActionPushBtn(self.race_data, cfg, device)
        elif action_type == "movetoBtn":
            cfg = self.policycfgs.movetoBtn
            device = self.device
            return RobotActionMovetoBtn(self.race_data, cfg, device)
        else:
            raise NotImplementedError

    def step(self, obs_dict):
        """Get actions from the policy."""
        cmd, count = self.race_data.popcommand()
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

    modify_cfg_to_simServer(env_cfg)

    action_cfgs = SimRobotServerCfg()
    action_cfgs.movetoBtn.policy.checkpoint = args_cli.moveto_ckpt
    action_cfgs.pushBtn.policy.checkpoint = args_cli.pushbtn_ckpt

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=False)
    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    server = RobotActorServer(env, action_cfgs, 12345, device = device)

    # reset environment
    obs_dict = env.reset()
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
