# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the physics engine to add an elevator to the scene.
From the default configuration file for these robots, zero actions imply a default pose.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import torch

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.robots.robot_base import RobotBase, RobotBaseCfg
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.orbit.actuators.group import ActuatorGroupCfg
from omni.isaac.orbit.actuators.group.actuator_group_cfg import ActuatorControlCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg

import numpy as np

from omni.isaac.orbit.robots.config.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG
from omni.isaac.orbit.robots.legged_robot import LeggedRobot

def main():
    """Spawns a single arm manipulator and applies random joint commands."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )
    
    robot_cfg = RobotBaseCfg(
    meta_info=RobotBaseCfg.MetaInfoCfg(
        usd_path="/home/chenyu/opt/orbit/source/standalone/elevator1.usd",
    ),
    init_state=RobotBaseCfg.InitialStateCfg(
        dof_pos={".*": 0.0},
        
        dof_vel={".*": 0.0},
    ),
    actuator_groups={
        "arm": ActuatorGroupCfg(
            dof_names=[".*"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=1000000.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": None},
                damping={".*": None},
            ),
        ),
    },)

    # -- Spawn robot
    robot = RobotBase(cfg=robot_cfg)
    robot.spawn("/World/elevator_1", translation=(0.0, 0, 0.0), orientation=(np.sqrt(1/2), np.sqrt(1/2), 0, 0.0))

    robot_c = LeggedRobot(cfg=ANYMAL_C_CFG)
    robot_c.spawn("/World/Anymal_c/Robot_1", translation=(1.5, -1.5, 0.65))

    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    robot.initialize("/World/elevator_1")
    robot_c.initialize("/World/Anymal_c/Robot_1")
    # Reset states
    robot.reset_buffers()
    robot_c.reset_buffers()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy actions
    actions = torch.rand(robot.count, robot.num_actions, device=robot.device)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        # reset
        if ep_step_count % 1000 == 0:
            sim_time = 0.0
            ep_step_count = 0
            # reset dof state
            dof_pos, dof_vel = robot.get_default_dof_state()
            robot.set_dof_state(dof_pos, dof_vel)
            print("dof_posshape", dof_pos.shape)
            robot.reset_buffers()
            # reset command
            actions = torch.rand(robot.count, robot.num_actions, device=robot.device)

            print("[INFO]: Resetting robots state...")

            dof_pos_c, dof_vel_c = robot_c.get_default_dof_state()
            robot_c.set_dof_state(dof_pos_c, dof_vel_c)
            robot_c.reset_buffers()
            actions_c = torch.zeros(robot_c.count, robot_c.num_actions, device=robot_c.device)

        
        # apply action to the robot
        # robot.apply_action(actions)
        robot.set_dof_state(dof_pos+torch.tensor([1,-1,1,-1]) * ep_step_count/1000 * 0.8, dof_vel)
        robot_c.apply_action(actions_c)
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        ep_step_count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)
            robot_c.update_buffers(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
