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

## Modules for Elevator
import re
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union
from pxr import Gf, UsdGeom
from omni.isaac.core.articulations import ArticulationView

class Elevator:
    """
    simple class for elevator.
    """
    articulations: ArticulationView = None
    def __init__(self):
        self._is_spawned = False
        self._door_state = 0 # 0: closed, 1: open

    """
    Properties
    """

    @property
    def count(self) -> int:
        """Number of prims encapsulated."""
        return self.articulations.count

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self.articulations._device  # noqa: W0212

    @property
    def num_dof(self) -> int:
        """Total number of DOFs in articulation."""
        return self.articulations.num_dof

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        internal_translation = (0., 0., 0.69)
        internal_orientation = (np.sqrt(1/2), np.sqrt(1/2), 0, 0.0)
        internal_transform = Gf.Matrix4d()
        internal_transform.SetTransform(Gf.Rotation(Gf.Quatd(*internal_orientation)), Gf.Vec3d(*internal_translation))
        if translation is not None:
            translation = (0., 0., 0.)
        if orientation is None:
            orientation = (1., 0., 0., 0.)
        transform = Gf.Matrix4d()
        transform.SetTransform(Gf.Rotation(Gf.Quatd(*orientation)), Gf.Vec3d(*translation))
        transform = transform * internal_transform
                # -- save prim path for later
        self._spawn_prim_path = prim_path
        # -- spawn asset if it doesn't exist.
        if not prim_utils.is_prim_path_valid(prim_path):
            # add prim as reference to stage
            quat = transform.ExtractRotation().GetQuat()
            prim_utils.create_prim(
                self._spawn_prim_path,
                usd_path="/home/chenyu/opt/orbit/source/standalone/elevator1.usd",
                translation=transform.ExtractTranslation(),
                orientation=(quat.real, *quat.imaginary),
            )
        else:
            carb.log_warn(f"A prim already exists at prim path: '{prim_path}'. Skipping...")
        self._is_spawned = True

    def initialize(self, prim_paths_expr: Optional[str] = None):
        # default prim path if not cloned
        if prim_paths_expr is None:
            if self._is_spawned is not None:
                self._prim_paths_expr = self._spawn_prim_path
            else:
                raise RuntimeError(
                    "Initialize the robot failed! Please provide a valid argument for `prim_paths_expr`."
                )
        else:
            self._prim_paths_expr = prim_paths_expr
        # create handles
        # -- robot articulation
        self.articulations = ArticulationView(self._prim_paths_expr, reset_xform_properties=False)
        self.articulations.initialize()
        # set the default state
        self.articulations.post_reset()
        self._ALL_INDICES = torch.arange(self.count, dtype=torch.long, device=self.device)
    
    def setDoorState(self, toopen=True):
        dof_target = 0.8 if toopen else 0.0 
        self._door_state = 1 if toopen else 0
        print("set door state to ", self._door_state)
        dof_pos_targets = (torch.Tensor([1.0, -1.0, 1.0, -1.0]) * dof_target).to(self.device).repeat(self.count, 1)
        self.articulations._physics_view.set_dof_position_targets(dof_pos_targets, self._ALL_INDICES)

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


    # -- Spawn robot
    elevator = Elevator()
    elevator.spawn("/World/elevator_1", translation=(0.0, 0, 0), orientation=(1, 0, 0, 0.0))

    robot_c = LeggedRobot(cfg=ANYMAL_C_CFG)
    robot_c.spawn("/World/Anymal_c/Robot_1", translation=(2.25, 0.9, 0.65))

    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    elevator.initialize("/World/elevator_1")
    robot_c.initialize("/World/Anymal_c/Robot_1")
    # Reset states
    robot_c.reset_buffers()

    # Now we are ready!
    print("[INFO]: Setup complete...")

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
            elevator.setDoorState( 1 - elevator._door_state)
            print("[INFO]: Resetting robots state...")

            dof_pos_c, dof_vel_c = robot_c.get_default_dof_state()
            robot_c.set_dof_state(dof_pos_c, dof_vel_c)
            robot_c.reset_buffers()
            actions_c = torch.zeros(robot_c.count, robot_c.num_actions, device=robot_c.device)

        robot_c.apply_action(actions_c)
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        ep_step_count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot_c.update_buffers(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
