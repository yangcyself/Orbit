# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gym.spaces
import math
import numpy as np
import os
import torch
from enum import Enum
from math import sqrt
# Modules for Elevator
from typing import Optional, Sequence, Union  # Dict, List, Tuple

import carb
import omni.isaac.core.utils.prims as prim_utils
# replicator
import omni.replicator.core as rep
import warp as wp
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidContactView
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from omni.isaac.sensor import ContactSensor
from pxr import Gf

from omni.isaac.assets import ASSETS_DATA_DIR

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematics
from omni.isaac.orbit.markers import PointMarker, StaticMarker
from omni.isaac.orbit.objects.button import ButtonObjectCfg, ButtonPanel, ButtonPanelCfg
from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulator
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import (combine_frame_transforms, matrix_from_quat, quat_apply, quat_from_euler_xyz,
                                         quat_mul, sample_uniform, scale_transform)
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs

from .elevator_cfg import ElevatorEnvCfg

# import omni.isaac.orbit_envs  # noqa: F401
# from omni.isaac.orbit_envs.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


class DoorState(Enum):
    """States for the elevator door."""

    OPEN = wp.constant(1)
    CLOSE = wp.constant(0)


class ButtonSmState(Enum):
    """States for the elevator door."""

    ON = wp.constant(1)
    OFF = wp.constant(0)


class ElevatorSmState(Enum):
    """States for the elevator state machine."""

    REST = wp.constant(0)
    DOOR_OPENING = wp.constant(1)
    DOOR_CLOSING = wp.constant(2)
    MOVE = wp.constant(3)


class ElevatorSmWaitTime(Enum):
    """Additional wait times (in s) for states for before switching."""

    DOOR_OPENING = wp.constant(20.0)
    DOOR_CLOSING = wp.constant(3.0)
    MOVE = wp.constant(25.0)


@wp.func
def BtnOpensFloor(tid: wp.int32, floor: wp.int32, btn_state: wp.array(dtype=wp.int32)):  # Check floor and button
    toOpenDoor = False
    if floor == 0 and (btn_state[tid]) == ButtonSmState.ON.value:  # At the floor 0
        btn_state[tid] = ButtonSmState.OFF.value
        toOpenDoor = True
    return toOpenDoor


@wp.kernel
def infer_state_machine(
    dt: wp.float32,
    sm_state: wp.array(dtype=wp.int32, ndim=2), # state machine state
    sm_wait_time: wp.array(dtype=wp.float32), # state machine wait time
    btn_state: wp.array(dtype=wp.int32), # button state
    door_state: wp.array(dtype=wp.int32),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid, 0]
    # update the floor states
    floor = sm_state[tid, 1]

    # decide next state
    if state == ElevatorSmState.REST.value:
        door_state[tid] = DoorState.CLOSE.value

        toOpenDoor = BtnOpensFloor(tid, floor, btn_state)
        if toOpenDoor:
            sm_state[tid, 0] = ElevatorSmState.DOOR_OPENING.value
            sm_wait_time[tid] = 0.0

        if floor != 0:
            sm_state[tid, 0] = ElevatorSmState.MOVE.value
            sm_wait_time[tid] = ElevatorSmWaitTime.MOVE.value - 1.0 - 1.0 * float(floor)

    elif state == ElevatorSmState.DOOR_OPENING.value:
        door_state[tid] = DoorState.OPEN.value

        toOpenDoor = BtnOpensFloor(tid, floor, btn_state)
        if toOpenDoor:
            sm_state[tid, 0] = ElevatorSmState.DOOR_OPENING.value
            sm_wait_time[tid] = 0.0

        if sm_wait_time[tid] >= ElevatorSmWaitTime.DOOR_OPENING.value:
            # move to next state and reset wait time
            sm_state[tid, 0] = ElevatorSmState.DOOR_CLOSING.value
            sm_wait_time[tid] = 0.0
    elif state == ElevatorSmState.DOOR_CLOSING.value:
        door_state[tid] = DoorState.CLOSE.value

        toOpenDoor = BtnOpensFloor(tid, floor, btn_state)
        if toOpenDoor:
            sm_state[tid, 0] = ElevatorSmState.DOOR_OPENING.value
            sm_wait_time[tid] = 0.0

        if sm_wait_time[tid] >= ElevatorSmWaitTime.DOOR_CLOSING.value:
            # move to next state and reset wait time
            sm_state[tid, 0] = ElevatorSmState.MOVE.value
            sm_wait_time[tid] = 0.0
    elif state == ElevatorSmState.MOVE.value:
        door_state[tid] = DoorState.CLOSE.value
        # wait for a while
        if sm_wait_time[tid] >= ElevatorSmWaitTime.MOVE.value:
            # move to next state and reset wait time
            sm_state[tid, 0] = ElevatorSmState.REST.value
            sm_state[tid, 1] = 0  # assume always move to floor 0
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt


class ElevatorSm:
    """A simple state machine for an elevator.

    The state machine is implemented as a warp kernel. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The elevator is at rest.
    2. DOOR_OPENING: The elevator opens the door when btn pushed.
    3. DOOR_CLOSING: The elevator close the door after waited certain time.
    4. MOVE: The elevator keep close the door.
    """

    def __init__(self, num_envs: int, device: Union[torch.device, str] = "cpu"):
        """Initialize the state machine.

        Args:
            num_envs (int): The number of environments to simulate.
            device (Union[torch.device, str], optional): The device to run the state machine on.
        """
        # save parameters
        self.num_envs = num_envs
        self.device = device
        print("\n\n\nDEVICE:", self.device)
        # states for the floor and buttons, [elevstate, Floor]
        self.sm_state = torch.full((self.num_envs, 2), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        # desired state for the door
        self.door_state = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        # convert to warp
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.door_state_wp = wp.from_torch(self.door_state, wp.int32)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = ...
        self.sm_state[env_ids,:] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, dt: float, btn_state: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert to warp
        btn_state = btn_state.to(self.device, dtype=torch.int32)
        btn_state_wp = wp.from_torch(btn_state, wp.int32)
        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[dt, self.sm_state_wp, self.sm_wait_time_wp, btn_state_wp, self.door_state_wp],
        )
        wp.synchronize()
        # convert to torch
        return self.door_state.bool(), btn_state.bool()


class Elevator:
    """
    simple class for elevator.
    """

    articulations: ArticulationView = None

    def __init__(self):
        self._is_spawned = False
        self._door_state = None  # 0: closed, 1: open
        self._sm_state = None
        self._door_pos_targets = None
        self._dof_default_pos = None
        self._sm = None
        self._dof_index = None
        self._dof_index_door = None
        self._dof_index_btn = None
        self._dof_index_light = None

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

    @property
    def all_mask(self):
        return torch.arange(self.count, dtype=torch.long, device=self.device)

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        # Hard code a basic transformation for the elevator
        internal_translation = (0.0, 0.0, 0.69)
        internal_orientation = (sqrt(1 / 2), sqrt(1 / 2), 0, 0.0)
        internal_transform = Gf.Matrix4d()
        internal_transform.SetTransform(Gf.Rotation(Gf.Quatd(*internal_orientation)), Gf.Vec3d(*internal_translation))
        if translation is None:
            translation = (0.0, 0.0, 0.0)
        if orientation is None:
            orientation = (1.0, 0.0, 0.0, 0.0)
        transform = Gf.Matrix4d()
        transform.SetTransform(Gf.Rotation(Gf.Quatd(*orientation)), Gf.Vec3d(*translation))
        transform = internal_transform * transform
        # -- save prim path for later
        self._spawn_prim_path = prim_path
        # -- spawn asset if it doesn't exist.
        if not prim_utils.is_prim_path_valid(prim_path):
            # add prim as reference to stage
            quat = transform.ExtractRotation().GetQuat()
            prim_utils.create_prim(
                self._spawn_prim_path,
                # usd_path="/home/chenyu/opt/orbit/source/standalone/elevator1.usd",
                usd_path=os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "elevator.usd"),
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

        self._door_state = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self._dof_default_targets = self.articulations._physics_view.get_dof_position_targets()
        self._dof_pos = self.articulations.get_joint_positions(indices=self.all_mask, clone=False)
        print("DoF Name", self.articulations.dof_names)
        # DoF Name ['PJoint_LO_Door', 'PJoint_RO_Door', 'PJoint_LI_Door', 'PJoint_RI_Door', 'PJoint_OU_Btn', 'PJoint_OD_Btn', 'RJoint_OU_Light', 'RJoint_OD_Light']
        print("ELEVATOR DEVICE", self.device)
        # summarize the DoF index
        self._dof_index = {name: i for i, name in enumerate(self.articulations.dof_names)}
        self._dof_index_door = [
            self._dof_index[n] for n in ["PJoint_LO_Door", "PJoint_RO_Door", "PJoint_LI_Door", "PJoint_RI_Door"]
        ]
        self._sm = ElevatorSm(self.count, "cuda")
        self._sm_state = self._sm.sm_state

    def reset_idx(self, env_ids: Optional[Sequence[int]] = None):
        """
        This function can be replaced by the state machine. In update_buffers
        """
        if env_ids is None:
            env_ids = self.all_mask
        elif len(env_ids) == 0:
            return
        self._sm.reset_idx(env_ids)
        self.articulations.set_joint_positions(torch.zeros((len(env_ids), 4),device = self.device), env_ids, self._dof_index_door)
        self.articulations.set_joint_velocities(torch.full((len(env_ids), len(self._dof_index)),0.,device = self.device), env_ids)

    def update_buffers(self, btn_state:torch.Tensor,  dt: float):
        """Step the elevator state machine.
            The interaction between elevator and button systems are extracted out
        Args:
            btn_state (torch.Tensor): The state of the buttons.
            dt (float): not used. Time between steps.

        Returns:
            torch.Tensor: new btn_state
        """
        self._dof_pos[:] = self.articulations.get_joint_positions(indices=self.all_mask, clone=False)
        door_state, btn_state_new = self._sm.compute(dt, btn_state)
        self._door_state = door_state.to(self.device)
        sm_state = self._sm_state.to(self.device)
        self._door_pos_targets = (
            torch.Tensor([[1.0, -1.0, 1.0, -1.0]]).to(self.device) * torch.where(self._door_state[..., None], 0.8, 0.0)
        ).to(self.device)
        self.articulations.set_joint_position_targets(self._door_pos_targets, self.all_mask, self._dof_index_door)
        return btn_state_new

    @property
    def state_should_dims(self):
        state_should_dims = [0]
        state_should_dims.append(state_should_dims[-1] + self._dof_pos.shape[1]) # dof_pos
        state_should_dims.append(state_should_dims[-1] + self._dof_pos.shape[1]) # dof_vel
        state_should_dims.append(state_should_dims[-1] + self._sm_state.shape[1]) # sm state
        state_should_dims.append(state_should_dims[-1] + self._sm.sm_wait_time.shape[0]) # sm wait time
        return state_should_dims

    def get_state(self):
        # Return the underlying state of a simulated environment. Should be compatible with reset_to.
        elevator_dofpos = self.articulations.get_joint_positions(indices=self.all_mask, clone=True).to(self.device)
        elevator_dofvel = self.articulations.get_joint_velocities(indices=self.all_mask, clone=True).to(self.device)
        elevator_state = self._sm_state.to(self.device)
        elevator_wait_time = self._sm.sm_wait_time.to(self.device).unsqueeze(1)
        return torch.cat([elevator_dofpos, elevator_dofvel, elevator_state, elevator_wait_time], dim=1)
    
    def reset_to(self, state):
        # Reset the simulated environment to a given state. Useful for reproducing results
        # state: N x D tensor, where N is the number of environments and D is the dimension of the state
        state_should_dims = self.state_should_dims
        assert state.shape[1] == state_should_dims[-1], "state should have dimension {} but got shape {}".format(state_should_dims[-1], state.shape)
        self._dof_pos[:,:] = state[:, state_should_dims[0]:state_should_dims[1]].to(self._dof_pos)
        _dof_vel = state[:, state_should_dims[1]:state_should_dims[2]].to(self._dof_pos)
        self._sm_state[:,:] = state[:, state_should_dims[2]:state_should_dims[3]].to(self._sm_state)
        self._sm.sm_wait_time[:] = state[:, state_should_dims[3]:state_should_dims[4]].to(self._sm.sm_wait_time).squeeze(1)
        self.articulations.set_joint_positions(self._dof_pos, indices=self.all_mask)
        self.articulations.set_joint_velocities(_dof_vel, indices=self.all_mask)


class ElevatorEnv(IsaacEnv):
    """Initializes the of a mobileManipulator and an elevator

    Args:
        cfg (ElevatorEnvCfg): The configuration dictionary.
        kwargs (dict): Additional keyword arguments. See IsaacEnv for more details.
    """

    def __init__(self, cfg: ElevatorEnvCfg = None, **kwargs):
        # copy configuration
        self.cfg = cfg
        # parse the configuration for controller configuration
        # note: controller decides the robot control mode
        self._pre_process_cfg()
        # create classes (these are called by the function :meth:`_design_scene`
        self.robot = MobileManipulator(cfg=self.cfg.robot)
        self.elevator = Elevator()
        self.buttonPanel = ButtonPanel(
            cfg=ButtonPanelCfg(
                panel_size = (0.3, 0.4),
                panel_grids = (3, 4),
                btn_cfgs=[
                    ButtonObjectCfg(
                        usd_path = os.path.join(os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "button_obj.usd")),
                        symbol_usd_path = "/home/chenyu/projects/2023Elevator/button/text_icons/text_4.usd"
                    ),
                    ButtonObjectCfg(
                        usd_path = os.path.join(os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "button_obj.usd")),
                        symbol_usd_path = "/home/chenyu/projects/2023Elevator/button/text_icons/text_5.usd"
                    ),
                    ButtonObjectCfg(
                        usd_path = os.path.join(os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "button_obj.usd")),
                        symbol_usd_path = "/home/chenyu/projects/2023Elevator/button/text_icons/text_6.usd"
                    ),
                    ButtonObjectCfg(
                        usd_path = os.path.join(os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "button_obj.usd")),
                        symbol_usd_path = "/home/chenyu/projects/2023Elevator/button/text_icons/text_7.usd"
                    ),
                    ButtonObjectCfg(
                        usd_path = os.path.join(os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "button_obj.usd")),
                        symbol_usd_path = "/home/chenyu/projects/2023Elevator/button/text_icons/text_8.usd"
                    ),
                    ButtonObjectCfg(
                        usd_path = os.path.join(os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "button_obj.usd")),
                        symbol_usd_path = "/home/chenyu/projects/2023Elevator/button/text_icons/text_9.usd"
                    ),
                    ButtonObjectCfg(
                        usd_path = os.path.join(os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "button_obj.usd")),
                        symbol_usd_path = "/home/chenyu/projects/2023Elevator/button/text_icons/text_0.usd"
                    ),
                    ButtonObjectCfg(
                        usd_path = os.path.join(os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "button_obj.usd")),
                        symbol_usd_path = "/home/chenyu/projects/2023Elevator/button/text_icons/text_1.usd"
                    ),
                    ButtonObjectCfg(
                        usd_path = os.path.join(os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "button_obj.usd")),
                        symbol_usd_path = "/home/chenyu/projects/2023Elevator/button/text_icons/text_up.usd"
                    ),
                    ButtonObjectCfg(
                        usd_path = os.path.join(os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "button_obj.usd")),
                        symbol_usd_path = "/home/chenyu/projects/2023Elevator/button/text_icons/text_down.usd"
                    ),
                    ButtonObjectCfg(
                        usd_path = os.path.join(os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "button_obj.usd")),
                        symbol_usd_path = "/home/chenyu/projects/2023Elevator/button/text_icons/text_E.usd"
                    ),
                    ButtonObjectCfg(
                        usd_path = os.path.join(os.path.join(ASSETS_DATA_DIR, "objects", "elevator", "button_obj.usd")),
                        symbol_usd_path = "/home/chenyu/projects/2023Elevator/button/text_icons/text_left.usd"
                    ),
                ]
            )
        )
        if("rgb" in self.modalities):
            camera_cfg = PinholeCameraCfg(
                sensor_tick=0,
                height=128,
                width=128,
                data_types=["rgb", "semantic_segmentation"],
                usd_params=PinholeCameraCfg.UsdCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                ),
            )
            self.camera = Camera(cfg=camera_cfg, device="cuda")
            base_camera_cfg = PinholeCameraCfg(
                sensor_tick=0,
                height=128,
                width=160,
                data_types=["rgb", "semantic_segmentation"],
                # FOV = 2 * arctan(horizontal_aperture / (2 * focal_length))
                # base_camera is wide about 120 degree
                usd_params=PinholeCameraCfg.UsdCameraCfg(
                    focal_length=6.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                ),
            )
            self.base_camera = Camera(cfg=base_camera_cfg, device="cuda")
            if(self.cfg.spawn_goal_camera):
                goal_camera_cfg = PinholeCameraCfg(
                    sensor_tick=0,
                    height=480,
                    width=640,
                    data_types=["rgb", "semantic_segmentation"],
                    usd_params=PinholeCameraCfg.UsdCameraCfg(
                        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                    ),
                )
                self.goal_camera = Camera(cfg=goal_camera_cfg, device="cuda")
            else:
                self.goal_camera = None
        else:
            self.camera = None
            self.base_camera = None

        # initialize the base class to setup the scene.
        super().__init__(self.cfg, **kwargs)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()

        # An array to record if the robot has pushed the button in the episode
        self._hasdone_pushbtn = torch.zeros((self.num_envs, ), dtype = bool, device=self.device)
        # An array to keep track which frame is this it. # traj_id, frame_id
        self.debug_tracker = torch.zeros((self.num_envs, 2), dtype = torch.int32, device=self.device)

        assert (self.base_camera is None or self.camera is not None), "camera must exist if base_camera exists"
        assert (self.num_envs == 1 or self.camera is None), "ElevatorEnv only supports num_envs=1 Otherwise camera shape is wrong"
        assert (self.enable_render or self.camera is None), "ElevatorEnv need `headless=False` if camera is wanted"
        # prepare the observation manager
        obs_cfg_dict = class_to_dict(self.cfg.observations)
        obs_cfg_dict = {k: v for k, v in obs_cfg_dict.items() if k in self.modalities}
        obs_cfg_dict["return_dict_obs_in_group"] = self.cfg.observations.return_dict_obs_in_group
        self._observation_manager = ElevatorObservationManager(obs_cfg_dict, self, self.device)

        # prepare the reward manager
        self.reward_manager = ElevatorRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        # print("[INFO] Observation Manager:", self._observation_manager)
        # print("[INFO] Reward Manager: ", self.reward_manager)

        # compute the observation space
        modality_space_dict = {}
        for ob in self.modalities:
            if(ob == "rgb"):
                rgb_num_obs = self._observation_manager._group_obs_dim["rgb"]
                modality_space_dict["rgb"] = gym.spaces.Box(low=0, high=255, shape=rgb_num_obs, dtype=np.uint8)
            else:
                num_obs = self._observation_manager._group_obs_dim[ob][0]
                modality_space_dict[ob] = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))

        obs_space_dict = {}
        for k,v in self.cfg.observation_grouping.items():
            if(type(v) == list):
                obs_space_dict[k] = gym.spaces.Dict({k2: modality_space_dict[k2] for k2 in v})
            elif(type(v) == str):
                obs_space_dict[k] = modality_space_dict[v]
            else:
                obs_space_dict[k] = modality_space_dict[k]
        self.observation_space = gym.spaces.Dict(obs_space_dict)
        # print("[INFO] Observation Space: ", self.observation_space)

        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        print("[INFO]: Completed setting up the environment...")
        # Take an initial step to initialize the scene.
        self.sim.step()
        # -- fill up buffers
        self.robot.update_buffers(self.dt)
        self.buttonPanel.update_buffers(self.dt)
        btn_state = self.elevator.update_buffers(self.buttonPanel.get_state_env_any(), self.dt)
        self.buttonPanel.set_state_env_all(0, None, torch.nonzero(~btn_state).flatten())
        self.buttonPanel.update_buffers(self.dt)
        if(self.camera is not None):
            self.camera.update(dt=self.dt)
        if(self.base_camera is not None):
            self.base_camera.update(dt=self.dt)

    """
    Implementation specifics.
    """

    def _design_scene(self):
        # ground plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)

        # robot
        self.robot.spawn(self.template_env_ns + "/Robot", translation=(0, 0, -0.5))
        self.elevator.spawn(
            self.template_env_ns + "/Elevator",
            translation=(1.5, -2.0, 0.0),
            orientation=(sqrt(1 / 2), 0.0, 0.0, sqrt(1 / 2)),
        )
        self.buttonPanel.spawn(
            self.template_env_ns + "/ButtonPanel",
            translation=(0, -0.75, 0.6),
            # orientation=(1 ,0.0, 0.0 , 0.0),
            orientation=(0.0, 0.0, sqrt(1 / 2), sqrt(1 / 2)),
        )

        # Spawn camera
        if(self.camera is not None):
            up_axis = Gf.Vec3d(0, -1, 0)
            eye_position = Gf.Vec3d(0.05, -0.1, 0.1)
            target_position = Gf.Vec3d(10, 0, 0)
            matrix_gf = Gf.Matrix4d(1).SetLookAt(eye_position, target_position, up_axis)
            # camera position and rotation in world frame
            matrix_gf = matrix_gf.GetInverse()
            cam_pos = np.array(matrix_gf.ExtractTranslation())
            cam_quat = gf_quat_to_np_array(matrix_gf.ExtractRotationQuat())
            self.camera.spawn(
                self.template_env_ns + "/Robot/dynaarm_ELBOW" + "/CameraSensor",
                translation=cam_pos,
                orientation=cam_quat,
            )
        if(self.base_camera is not None):
            self.base_camera.spawn(
                self.template_env_ns + "/Robot/base" + "/CameraSensor",
                translation=(0.3, 0, 0.2),
                orientation=(-0.5, -0.5, 0.5, 0.5),
            )
        if(self.goal_camera is not None):
            self.goal_camera.spawn(
                self.template_env_ns + "/CameraSensor",
                translation=(0, 0, 0),
                orientation=(1, 0, 0, 0),
            )

        # setup debug visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            # create point instancer to visualize the goal points
            self._goal_markers = PointMarker("/Visuals/ee_goal", self.num_envs, radius=0.025)
            # create marker for viewing end-effector pose
            self._ee_markers = StaticMarker(
                "/Visuals/ee_current", self.num_envs, usd_path=self.cfg.marker.usd_path, scale=self.cfg.marker.scale
            )
            # create marker for viewing command (if task-space controller is used)
            if self.cfg.control.control_type == "inverse_kinematics":
                self._cmd_markers = StaticMarker(
                    "/Visuals/ik_command", self.num_envs, usd_path=self.cfg.marker.usd_path, scale=self.cfg.marker.scale
                )
        # return list of global prims
        return ["/World/defaultGroundPlane"]

    def reset_idx(self, env_ids: VecEnvIndices):
        # ugly trick to bypass the private attribute check
        self._reset_idx(env_ids)

    def _reset_idx(self, env_ids: VecEnvIndices):
        # randomize the MDP
        # -- robot DOF state
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        self.elevator.reset_idx(env_ids=env_ids)
        self.buttonPanel.reset_buffers(env_ids=env_ids)

        # -- init pose
        self._randomize_robot_initial_pose(env_ids=env_ids)
        self._randomize_elevator_initial_state(env_ids=env_ids)

        # --desire position
        self.robot_des_pose_w[env_ids, 0:3] =  torch.tensor([[1.53, -2.08, -1.61]], device = self.device)

        # -- Reward logging
        # fill extras with episode information
        self.extras["episode"] = dict()
        # reset
        # -- rewards manager: fills the sums for terminated episodes
        self.reward_manager.reset_idx(env_ids, self.extras["episode"])
        # -- obs manager
        self._observation_manager.reset_idx(env_ids)
        # -- reset history
        self.previous_actions[env_ids] = 0
        # -- MDP reset
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        # -- Success reset
        self._hasdone_pushbtn[env_ids] = False
        self.debug_tracker[env_ids, 0] = torch.randint(0, int(1e9), (len(env_ids),)).to(device=self.device,dtype=torch.int32)
        self.debug_tracker[env_ids, 1] = 0

        # controller reset
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller.reset_idx(env_ids)

        # -- reset the scene
        for i in range(3):
            r = self.cfg.initialization.scene.obs_frame_bias_range[i]
            self._obs_shift_w[env_ids,i] = ((torch.rand(len(env_ids), )*2*r)-r).to(device=self.device,dtype=torch.float32)

        # -- reset textures and materials with replicator

        ## Replicator related code
        if self.cfg.initialization.scene.enable_replicator:
            if self.cfg.initialization.scene.randomize_ground_materials is not None:
                assert self.num_envs == 1, "randomize ground material only supports single environment"
                with rep.get.prims(path_pattern = "/World/defaultGroundPlane"):
                    rep.randomizer.materials(self.cfg.initialization.scene.randomize_ground_materials)
            if self.cfg.initialization.scene.randomize_wall_materials is not None:
                with rep.get.prims(path_pattern = self.env_ns + "/.*/Elevator/wall"):
                    rep.randomizer.materials(self.cfg.initialization.scene.randomize_wall_materials)
            if self.cfg.initialization.scene.randomize_door_materials is not None:
                with rep.get.prims(path_pattern = self.env_ns + "/.*/Elevator/.*sideDoor"):
                    rep.randomizer.materials(self.cfg.initialization.scene.randomize_door_materials)
            for i in range(5):
                self.sim.step()

        ## update the goal dict if it is in observation
        if "goal" in self.modalities:
            self.goal_dict.update(self.random_goal_image())

    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        self.actions = actions.clone().to(device=self.device)
        # transform actions based on controller
        if self.cfg.control.control_type == "inverse_kinematics":
            # set the controller commands
            # update ee_des_pos_base with actions
            self.ee_des_pos_base[:,:3] += self.actions[:,6:9] * self.dt

            base_r = self.robot.data.base_dof_pos[:, 3]
            ik_cmd = torch.zeros((self.num_envs, 7), device=self.device)
            ik_cmd[:, 0:3] = self.robot.data.base_dof_pos[:, :3]
            ik_cmd[:, 0] += self.ee_des_pos_base[:,0] * torch.cos(base_r) - self.ee_des_pos_base[:,1] * torch.sin(base_r)
            ik_cmd[:, 1] += self.ee_des_pos_base[:,0] * torch.sin(base_r) + self.ee_des_pos_base[:,1] * torch.cos(base_r)
            ik_cmd[:, 2] += self.ee_des_pos_base[:,2]
            # z foward pointing in local frame
            quat0 = torch.tensor([[math.sqrt(1/2),0.,math.sqrt(1/2),0.]], device = self.device).repeat(self.num_envs,1)
            # rotate of base
            quat1 = torch.stack([torch.cos(base_r/2), torch.zeros_like(base_r), torch.zeros_like(base_r), torch.sin(base_r/2)], dim=1)

            ik_cmd[:, 3:7] = quat_mul(quat1, quat0)

            self._ik_controller.set_command(ik_cmd)
            # compute the joint commands
            self.robot_actions[
                :, self.robot.base_num_dof : self.robot.base_num_dof + self.robot.arm_num_dof
            ] = self._ik_controller.compute(
                self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                self.robot.data.ee_state_w[:, 3:7],
                self.robot.data.ee_jacobian,
                self.robot.data.arm_dof_pos,
            )
            # offset actuator command with position offsets
            self.robot_actions[
                :, self.robot.base_num_dof : self.robot.base_num_dof + self.robot.arm_num_dof
            ] -= self.robot.data.actuator_pos_offset[
                :, self.robot.base_num_dof : self.robot.base_num_dof + self.robot.arm_num_dof
            ]
            # we assume the first is base command and we rotate it into robot's frame
            base_x = self.robot.data.base_dof_pos[:, 0] # x, y, z, yaw
            base_y = self.robot.data.base_dof_pos[:, 1] # x, y, z, yaw
            base_z = self.robot.data.base_dof_pos[:, 2] # x, y, z, yaw
            base_r = self.robot.data.base_dof_pos[:, 3] # x, y, z, yaw
            cmd_x = self.actions[:, 0] * torch.cos(base_r) - self.actions[:, 1] * torch.sin(base_r) + base_x
            cmd_y = self.actions[:, 0] * torch.sin(base_r) + self.actions[:, 1] * torch.cos(base_r) + base_y
            cmd_z = self.actions[:, 2] + base_z
            cmd_r = self.actions[:, 5] + base_r
            self.robot_actions[:, : self.robot.base_num_dof] = torch.cat(
                [cmd_x.unsqueeze(1), cmd_y.unsqueeze(1), cmd_z.unsqueeze(1), cmd_r.unsqueeze(1)], 1
            )
        elif self.cfg.control.control_type == "default":
            actions = self.actions.clone()
            if self.cfg.control.substract_action_from_obs_frame:
                self.obs_pose_substract(actions, px_idx=0, py_idx=1)
            self.robot_actions[:, :] = actions
        # perform physics stepping
        for _ in range(self.cfg.control.decimation):
            # set actions into buffers
            self.robot.apply_action(self.robot_actions)
            # simulate
            self.sim.step(render=self.enable_render)
            # check that simulation is playing
            if self.sim.is_stopped():
                return
        # post-step:
        # -- compute common buffers
        self.robot.update_buffers(self.dt)
        self.buttonPanel.update_buffers(self.dt)
        btn_state = self.elevator.update_buffers(self.buttonPanel.get_state_env_any(), self.dt)
        self.buttonPanel.set_state_env_all(0, None, torch.nonzero(~btn_state).flatten())
        self.buttonPanel.update_buffers(self.dt)
        if(self.camera is not None):
            self.camera.update(dt=self.dt)
        if(self.base_camera is not None):
            self.base_camera.update(dt=self.dt)

        # -- compute mid-level states # this should happen before reward, termination, observation and success
        elevator_state = self.elevator._sm_state.to(self.device)
        self._hasdone_pushbtn = torch.where(self.buttonPanel.get_state_env_any(), True, self._hasdone_pushbtn)
        self.debug_tracker[:,1] += 1

        # -- compute MDP signals
        # reward
        self.reward_penalizing_factor = torch.ones(self.num_envs, device=self.device)
        self.reward_penalizing_factor[elevator_state[:,0]==1] *= 2.
        self.reward_penalizing_factor[elevator_state[:,0]==2] *= 2.
        self.reward_penalizing_factor[(elevator_state[:,0]==3) & (elevator_state[:,1]==0)] *= 3.
        self.reward_penalizing_factor[(elevator_state[:,0]==3) & (elevator_state[:,1]!=0) & (self.buttonPanel.get_state_env_any())] *= 2.
        self.reward_buf = self.reward_manager.compute()
        # terminations
        self._check_termination()
        # -- store history
        self.previous_actions = self.actions.clone()

        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        self.extras["is_success"] = self.is_success()["task"]
        # -- update USD visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            self._debug_vis()

    def get_observations(self) -> VecEnvObs:
        # Just bypass the private method check
        return self._get_observations()

    def _get_observations(self) -> VecEnvObs:
        # compute observations
        obs = self._observation_manager.compute()
        # build return observation dict
        obs_dict = {}
        for k,v in self.cfg.observation_grouping.items():
            if(type(v) == list):
                obs_dict[k] = {k3: obs[k2][k3] for k2 in v for k3 in obs[k2].keys()}
            elif(type(v) == str):
                obs_dict[k] = obs[v]
            else:
                obs_dict[k] = obs[k]
        return obs_dict

    @property
    def state_should_dims(self):
        """
        The dims for vectorized state, used in get_state and reset_to_state
        """
        state_should_dims = [0] 
        state_should_dims.append(state_should_dims[-1] + self._obs_shift_w.shape[1])
        state_should_dims.append(state_should_dims[-1] + self.robot.state_should_dims[-1])
        state_should_dims.append(state_should_dims[-1] + self.buttonPanel.state_should_dims[-1])
        state_should_dims.append(state_should_dims[-1] + self.elevator.state_should_dims[-1])
        state_should_dims.append(state_should_dims[-1] + self.debug_tracker.shape[1])
        return state_should_dims

    def get_state(self):
        # Return the underlying state of a simulated environment. Should be compatible with reset_to.
        obs_shift_w = self._obs_shift_w
        robot_state = self.robot.get_state()
        buttonPanel_state = self.buttonPanel.get_state()
        elevator_state = self.elevator.get_state()
        debug_info = self.debug_tracker.to(self.device)
        return torch.cat([obs_shift_w, robot_state, buttonPanel_state, elevator_state, debug_info], dim=1)

    def reset_to(self, state):
        # Reset the simulated environment to a given state. Useful for reproducing results
        # state: N x D tensor, where N is the number of environments and D is the dimension of the state
        
        state_should_dims = self.state_should_dims
        assert state.shape[1] == state_should_dims[-1], "state should have dimension {} but got shape {}".format(state_should_dims[-1], state.shape)
        self._obs_shift_w[:,:] = state[:, state_should_dims[0]:state_should_dims[1]].to(self._obs_shift_w)
        self.robot.reset_to_state(state[:, state_should_dims[1]:state_should_dims[2]])
        self.buttonPanel.reset_to(state[:, state_should_dims[2]:state_should_dims[3]])
        self.elevator.reset_to(state[:, state_should_dims[3]:state_should_dims[4]])
        self.debug_tracker[:,:] = state[:, state_should_dims[4]:state_should_dims[5]].to(self.debug_tracker)

    """
    Helper functions - Scene handling.
    """

    def _pre_process_cfg(self) -> None:
        """Pre processing of configuration parameters."""
        # set configuration for task-space controller
        if self.cfg.control.control_type == "inverse_kinematics":
            print("Using inverse kinematics controller...")
            # enable jacobian computation
            self.cfg.robot.data_info.enable_jacobian = True
            # enable gravity compensation
            self.cfg.robot.rigid_props.disable_gravity = True
            # set the end-effector offsets
            self.cfg.control.inverse_kinematics.position_offset = self.cfg.robot.ee_info.pos_offset
            self.cfg.control.inverse_kinematics.rotation_offset = self.cfg.robot.ee_info.rot_offset
        else:
            print("Using default joint controller...")

        # Only Keep the observations that are used in observation grouping
        modalities = []
        for k,v in self.cfg.observation_grouping.items():
            if(type(v) == list):
                modalities += v
            elif(type(v) == str):
                modalities.append(v)
            else:
                modalities.append(k)
        self.modalities = set(modalities)
        print("[INFO] Observation modalities: ", self.modalities)

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()

        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")
        self.elevator.initialize(self.env_ns + "/.*/Elevator")
        self.buttonPanel.initialize(self.env_ns + "/.*/ButtonPanel")

        # Create the contact views
        self.rigidContacts = RigidContactView(self.env_ns + "/.*/Elevator/.*", [], prepare_contact_sensors=False, apply_rigid_body_api=False)
        self.rigidContacts.initialize()

        if(self.camera is not None):
            self.camera.initialize()
        if(self.base_camera is not None):
            self.base_camera.initialize()
        # self.camera.initialize(self.env_ns + "/.*/Robot/panda_hand/CameraSensor/Camera")
        if(self.goal_camera is not None):
            self.goal_camera.initialize()

        # create controller
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller = DifferentialInverseKinematics(
                self.cfg.control.inverse_kinematics, self.robot.count, self.device
            )
            self.num_actions = self.robot.base_num_dof + self._ik_controller.num_actions
            self.ee_des_pos_base = torch.tensor([[0.2, 0, 0.2]], device=self.device).tile((self.num_envs, 1))
        elif self.cfg.control.control_type == "default":
            self.num_actions = self.robot.base_num_dof + self.robot.arm_num_dof

        # history
        print("num_actions: ", self.num_actions)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # robot joint actions
        self.robot_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)

        # commands
        self.robot_des_pose_w = torch.zeros((self.num_envs, 3), device=self.device)

        # the transition from the world frame to the frame of the observation
        ## The robot observation should add this transform
        ## The robot action should substract this transform
        self._obs_shift_w = torch.zeros((self.num_envs, 3), device=self.device) # x, y, r

        # The Buffer for goal observation
        ## Will be updated by @random_goal_image and queriedy as obs in goal group
        if "goal" in self.modalities:
            class_names = self.cfg.observations.semantic.hand_camera_semantic["class_names"]
            self.goal_dict = {
                "rgb": torch.zeros((self.num_envs, *self.goal_camera.image_shape[:2], 3), device=self.device),
                "semantic": torch.zeros((self.num_envs, *self.goal_camera.image_shape[:2], len(class_names)), device=self.device),
                "campos": torch.zeros((self.num_envs, 3), device=self.device),
                "camrot": torch.zeros((self.num_envs, 4), device=self.device)
            }

    def _debug_vis(self):
        # compute error between end-effector and command

        # -- end-effector
        self._ee_markers.set_world_poses(self.robot.data.ee_state_w[:, 0:3], self.robot.data.ee_state_w[:, 3:7])
        # -- task-space commands
        if self.cfg.control.control_type == "inverse_kinematics":
            # convert to world frame
            ee_positions = self._ik_controller.desired_ee_pos + self.envs_positions
            ee_orientations = self._ik_controller.desired_ee_rot
            # set poses
            self._cmd_markers.set_world_poses(ee_positions, ee_orientations)

    """
    Helper functions - MDP.
    """

    def _check_termination(self) -> None:
        # extract values from buffer
        # compute resets
        self.reset_buf[:] = 0
        # -- episode length
        if self.cfg.terminations.is_success:
            self.reset_buf = torch.where(self.is_success()["task"].to(dtype = bool), 1, self.reset_buf)
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)
        if self.cfg.terminations.collision:
            collid = self.rigidContacts.get_net_contact_forces().abs().sum(axis = -1).reshape(self.num_envs,-1).sum(axis = -1)
            self.reset_buf = torch.where(collid > 10., 1, self.reset_buf)

    def _randomize_robot_initial_pose(self, env_ids: torch.Tensor):
        if(self.cfg.initialization.robot.position_cat=="uniform"):
            dof, dof_vel = self.robot.get_default_dof_state(env_ids)
            dof[:, 0:4] = sample_uniform(
                torch.tensor([self.cfg.initialization.robot.position_uniform_min], device=self.device), 
                torch.tensor([self.cfg.initialization.robot.position_uniform_max], device=self.device), 
                (len(env_ids), 4), device=self.device
            )
            self.robot.set_dof_state(dof, dof_vel, env_ids=env_ids)

    def _randomize_elevator_initial_state(self, env_ids: torch.Tensor):
        movingElevatorFlag = torch.rand((len(env_ids),),device=self.device) < self.cfg.initialization.elevator.moving_elevator_prob
        self.elevator._sm.sm_state[env_ids[movingElevatorFlag], 0] = 3
        self.elevator._sm.sm_state[env_ids[movingElevatorFlag], 1] = torch.randint(1, self.cfg.initialization.elevator.max_init_floor, 
            (movingElevatorFlag.sum(),), device = "cuda", dtype = torch.int32)
        self.elevator._sm.sm_wait_time[env_ids[movingElevatorFlag]] = torch.rand(movingElevatorFlag.sum(), device = "cuda")\
            * self.cfg.initialization.elevator.max_init_wait_time
        
        nonzeroFloorFlag = torch.rand((len(env_ids),),device=self.device) < self.cfg.initialization.elevator.nonzero_floor_prob
        self.elevator._sm.sm_state[env_ids[nonzeroFloorFlag], 1] = torch.randint(1, self.cfg.initialization.elevator.max_init_floor,
            (nonzeroFloorFlag.sum(),), device = "cuda", dtype = torch.int32)


    def is_success(self):
        """ Required by robomimic
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        robot_pos_error = torch.norm(self.robot.data.base_dof_pos[:,:2] - self.robot_des_pose_w[:,:2], dim=1)
        success_dict = {"enter_elevator": torch.where(robot_pos_error < self.cfg.terminations.is_success_threshold, 1, 0),    
                    "pushed_btn": torch.where(self._hasdone_pushbtn, 1, 0)
                }
        if type(self.cfg.terminations.is_success) == str:
            success_dict["task"] = success_dict[self.cfg.terminations.is_success]
        else:
            success_dict["task"] = success_dict["enter_elevator"]
        return success_dict


    def obs_pose_add(self, input_vec, px_idx=None, py_idx=None, pr_idx=None, vx_idx=None, vy_idx=None):
        """ Add the pose of the observation frame to the observation """
        x,y,r = self._obs_shift_w[:, 0], self._obs_shift_w[:, 1], self._obs_shift_w[:, 2]
        if(px_idx is not None and py_idx is not None):
            ##! the clones are necessary to avoid in-place operations
            px,py = input_vec[:, px_idx].clone(), input_vec[:, py_idx].clone()
            input_vec[:, px_idx] = px * torch.cos(r) - py * torch.sin(r)
            input_vec[:, py_idx] = px * torch.sin(r) + py * torch.cos(r)
        if(pr_idx is not None):
            input_vec[:, pr_idx] = r + input_vec[:, pr_idx]
        if(vx_idx is not None and vy_idx is not None):
            ##! the clones are necessary to avoid in-place operations
            vx,vy = input_vec[:, vx_idx].clone(), input_vec[:, vy_idx].clone()
            input_vec[:, vx_idx] = vx * torch.cos(r) - vy * torch.sin(r)
            input_vec[:, vy_idx] = vx * torch.sin(r) + vy * torch.cos(r)


    def obs_pose_substract(self, input_vec, px_idx=None, py_idx=None, pr_idx=None, vx_idx=None, vy_idx=None):
        """Minus the pose of the observation frame to the observation """
        x,y,r = self._obs_shift_w[:, 0], self._obs_shift_w[:, 1], self._obs_shift_w[:, 2]
        if(px_idx is not None and py_idx is not None):
            px,py = input_vec[:, px_idx].clone(), input_vec[:, py_idx].clone()
            input_vec[:, px_idx] = (px - x) * torch.cos(r) + (py - y) * torch.sin(r)
            input_vec[:, py_idx] = -(px - x) * torch.sin(r) + (py - y) * torch.cos(r)
        if(pr_idx is not None):
            input_vec[:, pr_idx] = pr - input_vec[:, pr_idx]
        if(vx_idx is not None and vy_idx is not None):
            vx,vy = input_vec[:, vx_idx].clone(), input_vec[:, vy_idx].clone()
            input_vec[:, vx_idx] = vx * torch.cos(r) + vy * torch.sin(r)
            input_vec[:, vy_idx] = -vx * torch.sin(r) + vy * torch.cos(r)

    @property
    def obs_shift_w(self):
        return self._obs_shift_w


    # overwrite the render function of the base env
    # concatenat the camera observation desides its viewpoint
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
            x-by-y pixel image, suitable for turning into a video.

        Args:
            mode (str, optional): The mode to render with. Defaults to "human".
        """
        rgb_data = super(ElevatorEnv, self).render(mode)
        if(mode == "rgb_array" and self.camera is not None):
            assert rgb_data is not None
            cam_data = (wp.torch.to_torch(self.camera.data.output["rgb"])[:, :, :3]).to(self.device).numpy()
            # Calculate the height difference
            height_diff = rgb_data.shape[0] - cam_data.shape[0]
            # Pad the smaller image with zeros at the bottom
            if height_diff > 0: # img1 is taller
                padding = ((0, height_diff), (0, 0), (0, 0))
                cam_data = np.pad(cam_data, padding, mode='constant')
            elif height_diff < 0: # img2 is taller
                padding = ((0, -height_diff), (0, 0), (0, 0))
                rgb_data = np.pad(rgb_data, padding, mode='constant')            
            # Concatenate the images horizontally
            rgb_data = np.concatenate((rgb_data, cam_data), axis=1)
            if("goal" in self.modalities and "rgb" in self.goal_dict.keys()):
                goal_rgb = self.goal_dict["rgb"].squeeze(0).numpy()
                goal_seg = self.goal_dict["semantic"].to(torch.uint8).squeeze(0).numpy()*255
                goal_imgs = [goal_rgb, *[np.tile(goal_seg[:,:,[i]],(1,1,3)) for i in range(goal_seg.shape[2])]]
                goal_data = np.concatenate(goal_imgs, axis = 1)
                width_diff = goal_data.shape[1] - rgb_data.shape[1]
                if width_diff > 0: # img1 is taller
                    padding = ((0, 0), (0, width_diff), (0, 0))
                    rgb_data = np.pad(rgb_data, padding, mode='constant')
                elif width_diff < 0: # img2 is taller
                    padding = ((0, 0), (0, -width_diff), (0, 0))
                    goal_data = np.pad(goal_data, padding, mode='constant')
                rgb_data = np.concatenate((goal_data, rgb_data), axis=0)

        return rgb_data

    def random_goal_image(self):
        ## Replicator related code
        assert self.num_envs == 1, "randomize goal image only support one environment"
        assert self.goal_camera is not None, "goal camera is not set"
        # the points: x,y, ry that are make sure the button are contained in the camera view
        eye = torch.rand(3) * torch.tensor([6., 5., 0.4]) + torch.tensor([-2, 3., 0.3])
        target = torch.rand(3) * torch.tensor([1., 1., 0.2]) + torch.tensor([0.5, -1, 0.4])
        prim = prim_utils.get_prim_at_path(self.robot._spawn_prim_path)
        prim_utils.set_prim_visibility(prim, visible=False)
        self.goal_camera.set_world_pose_from_view(eye = eye, target = target)
        self.sim.step()
        self.goal_camera.update(dt=self.dt)
        self.goal_camera.set_world_pose_from_view(eye = eye, target = target)
        self.sim.step()
        self.goal_camera.update(dt=self.dt)
        prim_utils.set_prim_visibility(prim, visible=True)
        
        # Process the semantic segmentation data
        class_names = self.cfg.observations.semantic.hand_camera_semantic["class_names"]
        idToLabels = self.goal_camera.data.output["semantic_segmentation"]['info']["idToLabels"]
        labelToIds = {label["class"]: int(idx) for idx, label in idToLabels.items()}
        data = wp.torch.to_torch(self.goal_camera.data.output["semantic_segmentation"]['data'])[None, :, :].squeeze(3) # n_env, H, W
        channels = []
        for labels in class_names:
            # Combine binary masks for each label in the group
            binary_mask = torch.zeros_like(data, dtype=torch.bool)
            for label in labels:
                idx = labelToIds.get(label)
                if idx is not None:
                    binary_mask = binary_mask | (data == idx)
            # Append to the list
            channels.append(binary_mask)
        segment = torch.stack(channels,dim=3).to(self.device) # n_env, H, W, C

        # Camera pose
        campos, camrot = self.goal_camera._data.position, self.goal_camera._data.orientation 
        _zero_vec = torch.zeros((self.num_envs,)).to(self._obs_shift_w)
        new_campos, new_camrot = combine_frame_transforms(
                torch.cat([self._obs_shift_w[:,:2],_zero_vec.unsqueeze(0)], dim=1) ,
                quat_from_euler_xyz(_zero_vec, _zero_vec, self._obs_shift_w[:, 2]),
                torch.tensor(campos).unsqueeze(0).to(self._obs_shift_w), 
                torch.tensor(camrot).unsqueeze(0).to(self._obs_shift_w))
        return_dict = {
            "rgb": wp.torch.to_torch(self.goal_camera.data.output["rgb"])[None, :, :, :3].to(self.device),
            "semantic": segment,
            "campos": new_campos,
            "camrot": new_camrot
        }
        return return_dict
        

class ElevatorObservationManager(ObservationManager):
    """Reward manager for single-arm reaching environment."""

    def dof_pos_normalized(self, env: ElevatorEnv):
        """DOF positions for the arm normalized to its max and min ranges."""
        return scale_transform(
            env.robot.data.dof_pos,
            env.robot.data.soft_dof_pos_limits[:, :, 0],
            env.robot.data.soft_dof_pos_limits[:, :, 1],
        )

    def dof_vel(self, env: ElevatorEnv):
        """DOF velocity of the arm."""
        return env.robot.data.dof_vel

    def ee_position(self, env: ElevatorEnv):
        """Current end-effector position of the arm."""
        return env.robot.data.ee_state_w[:, :3]

    def dof_pos_obsframe(self, env: ElevatorEnv):
        """DOF positions for the arm in observation frame."""
        dof_pos = env.robot.data.dof_pos.clone()
        env.obs_pose_add(dof_pos, px_idx=0, py_idx=1, pr_idx=2)
        return scale_transform(
            dof_pos,
            env.robot.data.soft_dof_pos_limits[:, :, 0],
            env.robot.data.soft_dof_pos_limits[:, :, 1],
        )
    
    def dof_vel_obsframe(self, env: ElevatorEnv):
        """DOF velocity for the arm in observation frame."""
        dof_vel = env.robot.data.dof_vel.clone()
        env.obs_pose_add(dof_vel, vx_idx=0, vy_idx=1)
        return dof_vel
    
    def ee_position_obsframe(self, env: ElevatorEnv):
        """Current end-effector position of the arm in observation frame."""
        ee_pos = env.robot.data.ee_state_w[:,:3].clone()
        env.obs_pose_add(ee_pos, px_idx=0, py_idx=1)
        return ee_pos

    def elevator_state(self, env: ElevatorEnv):
        """The state of the elevator"""
        return torch.nn.functional.one_hot(env.elevator._sm_state[:,0].to(dtype = torch.int64, device = env.device), num_classes = 4).to(torch.float32)
    
    def elevator_waittime(self, env: ElevatorEnv):
        """The waittime of the elevator"""
        return (env.elevator._sm.sm_wait_time/30).reshape((env.num_envs, 1)).to(env.device)

    def elevator_is_zerofloor(self, env: ElevatorEnv):
        """The waittime of the elevator"""
        return (env.elevator._sm_state[:,1,None].to(dtype = torch.int64, device = env.device) == 0).to(dtype = torch.float32, device = env.device)

    def elevator_btn_pressed(self, env: ElevatorEnv):
        """Whether the button is pressed"""
        elevator_state = env.elevator._sm_state.to(env.device)
        return env.buttonPanel.get_state_env_any().unsqueeze(1).to(dtype = torch.float32, device = env.device)

    def hand_camera_rgb(self, env: ElevatorEnv):
        """RGB camera observations.
        type uint8 and be stored in channel-last (H, W, C) format.
        """
        image_shape = env.camera.image_shape
        if env.camera.data.output["rgb"] is None:
            return torch.zeros((env.num_envs, image_shape[0], image_shape[1], 3), device=env.device)
        else:
            return (wp.torch.to_torch(env.camera.data.output["rgb"])[None, :, :, :3]).to(env.device)

    def hand_camera_semantic(self, env: ElevatorEnv, class_names=None):
        """Semantic camera observations.
        type uint8 and be stored in channel-last (H, W, C) format.
        class_names: a list of tuples, each tuple contains the names for the channel of the output
        """
        class_names = [] if class_names is None else class_names
        num_classes = len(class_names)
        image_shape = env.camera.image_shape
        if env.camera.data.output["semantic_segmentation"] is None:
            return torch.zeros((env.num_envs, image_shape[0], image_shape[1], num_classes), dtype=torch.bool, device=env.device)
        else:
            idToLabels = env.camera.data.output["semantic_segmentation"]['info']["idToLabels"]
            labelToIds = {label["class"]: int(idx) for idx, label in idToLabels.items()}
            data = wp.torch.to_torch(env.camera.data.output["semantic_segmentation"]['data'])[None, :, :].squeeze(3) # n_env, H, W
            channels = []
            for labels in class_names:
                # Combine binary masks for each label in the group
                binary_mask = torch.zeros_like(data, dtype=torch.bool)
                for label in labels:
                    idx = labelToIds.get(label)
                    if idx is not None:
                        binary_mask = binary_mask | (data == idx)
                # Append to the list
                channels.append(binary_mask)
            return torch.stack(channels,dim=3).to(env.device) # n_env, H, W, C


    def actions(self, env: ElevatorEnv):
        """Last actions provided to env."""
        return env.actions
    
    def debug_info(self, env: ElevatorEnv):
        return env.debug_tracker
    
    def obs_shift_w(self, env: ElevatorEnv):
        return env.obs_shift_w

    def goal_rgb(self, env: ElevatorEnv):
        return env.goal_dict["rgb"]
    def goal_semantic(self, env: ElevatorEnv):
        return env.goal_dict["semantic"]

    def goal_campos(self, env: ElevatorEnv):
        return env.goal_dict["campos"]
    def goal_camrot(self, env: ElevatorEnv):
        return env.goal_dict["camrot"]


class ElevatorRewardManager(RewardManager):
    """Reward manager for single-arm reaching environment."""

    def penalizing_robot_dof_velocity_l2(self, env: ElevatorEnv):
        """Penalize large movements of the robot arm."""
        reward = torch.sum(torch.square(env.robot.data.arm_dof_vel), dim=1)
        return reward * env.reward_penalizing_factor

    def penalizing_robot_dof_acceleration_l2(self, env: ElevatorEnv):
        """Penalize fast movements of the robot arm."""
        reward = torch.sum(torch.square(env.robot.data.dof_acc), dim=1)
        return reward * env.reward_penalizing_factor

    def penalizing_action_rate_l2(self, env: ElevatorEnv):
        """Penalize large variations in action commands."""
        reward = torch.sum(torch.square(env.actions[:, :] - env.previous_actions[:, :]), dim=1)
        return reward * env.reward_penalizing_factor

    def penalizing_action_l2(self, env: ElevatorEnv):
        """Penalize large actions."""
        reward = torch.sum(torch.square(env.actions[:, :]), dim=1)
        return reward * env.reward_penalizing_factor
    
    def penalizing_collision(self, env:ElevatorEnv):
        """Penalize collision"""
        return (env.rigidContacts.get_net_contact_forces().abs().sum(axis = -1).reshape(env.num_envs,-1).sum(axis = -1)>0.1).to(torch.float32)

    def penalizing_camera_lin_vel_l2(self, env:ElevatorEnv):
        """Penalize camera movement"""
        lin_vel = env.robot.data.ee_state_w[:,7:10]
        reward = torch.sum(torch.square(lin_vel), dim=1)
        return reward * env.reward_penalizing_factor

    def penalizing_camera_ang_vel_l2(self, env:ElevatorEnv):
        """Penalize camera movement"""
        ang_vel = env.robot.data.ee_state_w[:,10:13]
        reward = torch.sum(torch.square(ang_vel), dim=1)
        return reward * env.reward_penalizing_factor

    def penalizing_nonflat_camera_l2(self, env:ElevatorEnv):
        axis_z = matrix_from_quat(env.robot.data.ee_state_w[:, 3:7])[:,2,:]
        w = torch.tensor([[0.5,0.,1.]], device = env.device)
        ref = torch.tensor([[1.,0.,0.]], device = env.device)
        reward = torch.sum(torch.square(axis_z - ref) * w, dim=1)
        return reward * env.reward_penalizing_factor

    def look_at_moving_direction_exp(self, env:ElevatorEnv, sigma):
        """
        Make the camera look at the moving direction of the robot
        """
        # The x,y direction of the camera
        lookDir = matrix_from_quat(env.robot.data.ee_state_w[:, 3:7])[:,:2,2]
        baseVelDir = env.robot.data.base_dof_vel[:,:2]
        return torch.exp(- torch.sum(lookDir * baseVelDir, dim=-1) 
            / (torch.norm(lookDir, dim=-1) + 1e-6)
            / (torch.norm(baseVelDir, dim=-1) + 0.2) / sigma) - 1.

    def tracking_reference_button_pos(self, env: ElevatorEnv, sigma):
        ee_state_w = env.robot.data.ee_state_w[:, :7]
        ee_state_w[:,:3] -= env.envs_positions
        # make the first element positive
        target_ee_pose_push_btn = torch.tensor([0.3477, -0.7259,  0.1966,  0.2992,  0.6214, -0.5843,  0.4277],
            device = env.device)
        error_position = torch.sum(torch.square(ee_state_w[:,:3] - target_ee_pose_push_btn[:3]), dim=1)
        reward = torch.exp(-error_position / sigma)
        
        elevator_state = env.elevator._sm_state.to(env.device)
        reward[elevator_state[:,0]==1 ] = 1. # No reward gradient for button pushing when elevator is not at rest
        reward[(elevator_state[:,0]==2) ] = 0.
        reward[(elevator_state[:,0]==3) & (elevator_state[:,1]==0)] = 0.
        reward[(elevator_state[:,0]==3) & (elevator_state[:,1]!=0) & (env.buttonPanel.get_state_env_any())] = 1.
        return reward

    def tracking_reference_button_rot(self, env: ElevatorEnv, sigma):
        ee_state_w = env.robot.data.ee_state_w[:, :7]
        # make the first element positive
        ee_state_w[ee_state_w[:,3]<0, 3:7] *= -1

        target_ee_pose_push_btn = torch.tensor([0.3477, -0.7259,  0.1966,  0.2992,  0.6214, -0.5843,  0.4277],
                    device = env.device)
        error_rotation = torch.sum(torch.square(ee_state_w[:,3:7] - target_ee_pose_push_btn[3:]), dim=1)
        reward = torch.exp(-error_rotation / sigma)
        elevator_state = env.elevator._sm_state.to(env.device)
        reward[elevator_state[:,0]==1 ] = 1. # No reward gradient for button pushing when elevator is not at rest
        reward[(elevator_state[:,0]==2) ] = 0.
        reward[(elevator_state[:,0]==3) & (elevator_state[:,1]==0)] = 0.
        reward[(elevator_state[:,0]==3) & (env.buttonPanel.get_state_env_any())] = 1.
        return reward

    def tracking_reference_enter(self, env: ElevatorEnv, sigma):        
        robot_pos_error = torch.norm(env.robot.data.base_dof_pos[:,:3] - env.robot_des_pose_w[:,:3], dim=1)
        reward = torch.exp(-robot_pos_error / sigma)
        elevator_state = env.elevator._sm_state.to(env.device)
        reward[(elevator_state[:,0] != 1)] = 0.
        return reward

    def tracking_reference_waitin(self, env: ElevatorEnv, sigma):
        robot_pos_error = torch.norm(env.robot.data.base_dof_pos[:,:3] - env.robot_des_pose_w[:,:3], dim=1)
        reward = torch.exp(-robot_pos_error / sigma)
        # no reward if the door is closed and robot is outside of the elevator
        elevator_state = env.elevator._sm_state.to(env.device)
        reward[(elevator_state[:,0] <= 1) | (elevator_state[:,1] != 0)] = 0.
        return reward
    
    def tracking_reference_waitout(self, env: ElevatorEnv, sigma):
        target_base_pose_waitout = torch.tensor([[1.6093,  0.4073, -1.6123]], device = env.device)
        robot_pos_error = torch.norm(env.robot.data.base_dof_pos[:,:3] - target_base_pose_waitout, dim=1)
        reward = torch.exp(-robot_pos_error / sigma)
        # no reward if the door is closed and robot is outside of the elevator
        elevator_state = env.elevator._sm_state.to(env.device)
        reward[elevator_state[:,1] == 0] = 0.
        reward[~env.buttonPanel.get_state_env_any()] = 0. # Neigther btn is pressed
        return reward

