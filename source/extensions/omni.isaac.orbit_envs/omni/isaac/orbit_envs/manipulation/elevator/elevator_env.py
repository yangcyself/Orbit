# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import gym.spaces
import math
import numpy as np
import torch
from enum import Enum
from math import sqrt

# Modules for Elevator
from typing import Optional, Sequence, Union  # Dict, List, Tuple

import carb
import omni.isaac.core.utils.prims as prim_utils
import warp as wp
from omni.isaac.core.articulations import ArticulationView 
from omni.isaac.core.prims import RigidContactView
from omni.isaac.sensor import ContactSensor
from pxr import Gf

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematics
from omni.isaac.orbit.markers import PointMarker, StaticMarker
from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulator
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import quat_apply, quat_mul, scale_transform, sample_uniform, matrix_from_quat
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs
from omni.isaac.assets import ASSETS_DATA_DIR
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

    DOOR_OPENING = wp.constant(25.0)
    DOOR_CLOSING = wp.constant(3.0)
    MOVE = wp.constant(25)


@wp.func
def BtnOpensFloor(tid: wp.int32, floor: wp.int32, sm_state: wp.array(dtype=wp.int32, ndim=2)):  # Check floor and button
    toOpenDoor = False
    if floor == 0 and (sm_state[tid, 2]) == ButtonSmState.ON.value:  # At the floor 0
        sm_state[tid, 2] = ButtonSmState.OFF.value
        toOpenDoor = True
    if floor == 0 and (sm_state[tid, 3]) == ButtonSmState.ON.value:  # At the floor 0
        sm_state[tid, 3] = ButtonSmState.OFF.value
        toOpenDoor = True
    return toOpenDoor


@wp.kernel
def infer_state_machine(
    dt: wp.float32,
    Nbtn: wp.int32,
    sm_state: wp.array(dtype=wp.int32, ndim=2),
    sm_wait_time: wp.array(dtype=wp.float32),
    btn_pose: wp.array(dtype=wp.float32, ndim=2),
    door_state: wp.array(dtype=wp.int32),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid, 0]
    # update the floor states
    floor = sm_state[tid, 1]

    # update the btn states
    for i in range(Nbtn):
        if btn_pose[tid, i] < 0.0:
            sm_state[tid, i + 2] = ButtonSmState.ON.value

    # decide next state
    if state == ElevatorSmState.REST.value:
        door_state[tid] = DoorState.CLOSE.value

        toOpenDoor = BtnOpensFloor(tid, floor, sm_state)
        if toOpenDoor:
            sm_state[tid, 0] = ElevatorSmState.DOOR_OPENING.value
            sm_wait_time[tid] = 0.0

        if floor != 0:
            sm_state[tid, 0] = ElevatorSmState.MOVE.value
            sm_wait_time[tid] = ElevatorSmWaitTime.MOVE.value - 5.0 - 10.0 * float(floor)

    elif state == ElevatorSmState.DOOR_OPENING.value:
        door_state[tid] = DoorState.OPEN.value

        toOpenDoor = BtnOpensFloor(tid, floor, sm_state)
        if toOpenDoor:
            sm_state[tid, 0] = ElevatorSmState.DOOR_OPENING.value
            sm_wait_time[tid] = 0.0

        if sm_wait_time[tid] >= ElevatorSmWaitTime.DOOR_OPENING.value:
            # move to next state and reset wait time
            sm_state[tid, 0] = ElevatorSmState.DOOR_CLOSING.value
            sm_wait_time[tid] = 0.0
    elif state == ElevatorSmState.DOOR_CLOSING.value:
        door_state[tid] = DoorState.CLOSE.value

        toOpenDoor = BtnOpensFloor(tid, floor, sm_state)
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
        # initialize state machine
        self.sm_Nbtn = torch.full((self.num_envs,), 2, dtype=torch.int32, device=self.device)
        # states for the floor and buttons, [elevstate, Floor, down btn, up btn]
        self.sm_state = torch.full((self.num_envs, 4), 0, dtype=torch.int32, device=self.device)
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

    def compute(self, dt: float, btn_pos: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert to warp
        btn_pos_wp = wp.from_torch(btn_pos.to(self.device), wp.float32)
        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[dt, 2, self.sm_state_wp, self.sm_wait_time_wp, btn_pos_wp, self.door_state_wp],
        )
        wp.synchronize()
        # convert to torch
        return self.door_state.bool()


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
        self._dof_index_btn = [self._dof_index[n] for n in ["PJoint_OU_Btn", "PJoint_OD_Btn"]]
        self._dof_index_light = [self._dof_index[n] for n in ["RJoint_OU_Light", "RJoint_OD_Light"]]

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

    def update_buffers(self, dt: float):
        self._dof_pos[:] = self.articulations.get_joint_positions(indices=self.all_mask, clone=False)
        door_state = self._sm.compute(dt, self._dof_pos[:, self._dof_index_btn])
        self._door_state = door_state.to(self.device)
        sm_state = self._sm_state.to(self.device)
        self._door_pos_targets = (
            torch.Tensor([[1.0, -1.0, 1.0, -1.0]]).to(self.device) * torch.where(self._door_state[..., None], 0.8, 0.0)
        ).to(self.device)
        # print("dof_pos", self._dof_pos)
        # print("sm_state", sm_state)
        self.articulations.set_joint_positions(sm_state[:, -2:] * 3.14, self.all_mask, self._dof_index_light)
        self.articulations.set_joint_position_targets(self._door_pos_targets, self.all_mask, self._dof_index_door)
        self.articulations.set_joint_position_targets(
            self._dof_pos[:, self._dof_index_btn] + 1, self.all_mask, self._dof_index_btn
        )


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

        if("rgb" in self.modalities):
            camera_cfg = PinholeCameraCfg(
                sensor_tick=0,
                # height=480,
                # width=640,
                height=128,
                width=128,
                data_types=["rgb"],
                usd_params=PinholeCameraCfg.UsdCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                ),
            )
            self.camera = Camera(cfg=camera_cfg, device="cuda")
        else:
            self.camera = None

        # initialize the base class to setup the scene.
        super().__init__(self.cfg, **kwargs)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()

        assert (self.num_envs == 1 or self.camera is None), "ElevatorEnv only supports num_envs=1 Otherwise camera shape is wrong"

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
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self.reward_manager)

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
        print("[INFO] Observation Space: ", self.observation_space)

        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        print("[INFO]: Completed setting up the environment...")
        # Take an initial step to initialize the scene.
        self.sim.step()
        # -- fill up buffers
        self.robot.update_buffers(self.dt)
        self.elevator.update_buffers(self.dt)
        if(self.camera is not None):
            self.camera.update(dt=self.dt)

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

        # Spawn camera
        if(self.camera is not None):
            self.camera.spawn(
                self.template_env_ns + "/Robot/panda_hand" + "/CameraSensor",
                translation=(0.05, 0.005, -0.01),
                orientation=(0.0616284, 0.704416, 0.704416, 0.0616284),
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

    def _reset_idx(self, env_ids: VecEnvIndices):
        # randomize the MDP
        # -- robot DOF state
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        self.elevator.reset_idx(env_ids=env_ids)

        # -- init pose
        self._randomize_robot_initial_pose(env_ids=env_ids)
        self._randomize_elevator_initial_state(env_ids=env_ids)

        # --desire position
        self.robot_des_pose_w[env_ids, 0:2] =  torch.tensor([[1.53,-2.08]], device = self.device)

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
        # controller reset
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller.reset_idx(env_ids)

    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        self.actions = actions.clone().to(device=self.device)
        # transform actions based on controller
        if self.cfg.control.control_type == "inverse_kinematics":
            # set the controller commands
            ee_quad = self.robot.data.ee_state_w[:, 3:7]
            cmd_trans = self.actions[:, self.robot.base_num_dof : self.robot.base_num_dof + 3]
            cmd_quad = self.actions[:, self.robot.base_num_dof + 3 : self.robot.base_num_dof + 6]
            cmd_trans_rotated = quat_apply(ee_quad, cmd_trans)
            cmd_quad_rotated = quat_apply(ee_quad, cmd_quad)
            ik_cmd = torch.cat([cmd_trans_rotated, cmd_quad_rotated], 1).to(device=self.device)
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
            base_r = self.robot.data.base_dof_pos[:, 2]
            cmd_x = self.actions[:, 0] * torch.cos(base_r) - self.actions[:, 1] * torch.sin(base_r)
            cmd_y = self.actions[:, 0] * torch.sin(base_r) + self.actions[:, 1] * torch.cos(base_r)
            self.robot_actions[:, : self.robot.base_num_dof] = torch.cat(
                [cmd_x.unsqueeze(1), cmd_y.unsqueeze(1), self.actions[:, 2].unsqueeze(1)], 1
            )
            # we assume last command is tool action so don't change that
            self.robot_actions[:, -1] = self.actions[:, -1]
        elif self.cfg.control.control_type == "default":
            self.robot_actions[:, :] = self.actions
        elif self.cfg.control.control_type == "ohneHand":
            self.robot_actions[:, :-1] = self.actions
            self.robot_actions[:, -1] = -1.0
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
        self.elevator.update_buffers(self.dt)
        if(self.camera is not None):
            self.camera.update(dt=self.dt)
        # -- compute MDP signals
        # reward
        self.reward_buf = self.reward_manager.compute()
        # terminations
        self._check_termination()
        # -- store history
        self.previous_actions = self.actions.clone()

        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        robot_pos_error = torch.norm(self.robot.data.base_dof_pos[:,:2] - self.robot_des_pose_w[:,:2], dim=1)
        self.extras["is_success"] = torch.where(robot_pos_error < self.cfg.terminations.is_success_threshold, 1, 0)
        # -- update USD visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            self._debug_vis()


    def _get_observations(self) -> VecEnvObs:
        # compute observations
        obs = self._observation_manager.compute()
        # build return observation dict
        obs_dict = {}
        for k,v in self.cfg.observation_grouping.items():
            if(type(v) == list):
                obs_dict[k] = {k2: obs[k2] for k2 in v}
            elif(type(v) == str):
                obs_dict[k] = obs[v]
            else:
                obs_dict[k] = obs[k]
        return obs_dict

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

        # Create the contact views
        self.rigidContacts = RigidContactView(self.env_ns + "/.*/Elevator/.*", [], prepare_contact_sensors=False, apply_rigid_body_api=False)
        self.rigidContacts.initialize()

        if(self.camera is not None):
            self.camera.initialize()
        # self.camera.initialize(self.env_ns + "/.*/Robot/panda_hand/CameraSensor/Camera")

        # create controller
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller = DifferentialInverseKinematics(
                self.cfg.control.inverse_kinematics, self.robot.count, self.device
            )
            self.num_actions = self.robot.base_num_dof + self._ik_controller.num_actions + 1
        elif self.cfg.control.control_type == "default":
            self.num_actions = self.robot.base_num_dof + self.robot.arm_num_dof + 1
        elif self.cfg.control.control_type == "ohneHand":
            self.num_actions = self.robot.base_num_dof + self.robot.arm_num_dof

        # history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # robot joint actions
        self.robot_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)

        # commands
        self.robot_des_pose_w = torch.zeros((self.num_envs, 3), device=self.device)

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
            robot_pos_error = torch.norm(self.robot.data.base_dof_pos[:,:2] - self.robot_des_pose_w[:,:2], dim=1)
            self.reset_buf = torch.where(robot_pos_error < self.cfg.terminations.is_success_threshold, 1, self.reset_buf)
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)
        if self.cfg.terminations.collision:
            collid = self.rigidContacts.get_net_contact_forces().abs().sum(axis = -1).reshape(self.num_envs,-1).sum(axis = -1)
            self.reset_buf = torch.where(collid > 10., 1, self.reset_buf)

    def _randomize_robot_initial_pose(self, env_ids: torch.Tensor):
        if(self.cfg.initialization.robot.position_cat=="uniform"):
            dof, dof_vel = self.robot.get_default_dof_state(env_ids)
            dof[:, 0:3] = sample_uniform(
                torch.tensor([self.cfg.initialization.robot.position_uniform_min], device=self.device), 
                torch.tensor([self.cfg.initialization.robot.position_uniform_max], device=self.device), 
                (len(env_ids), 3), device=self.device
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
        return ((elevator_state[:,2,None]>0) | (elevator_state[:,3,None]>0)).to(dtype = torch.float32, device = env.device)

    def hand_camera_rgb(self, env: ElevatorEnv):
        """RGB camera observations.
        type uint8 and be stored in channel-last (H, W, C) format.
        """
        image_shape = env.camera.image_shape
        if env.camera.data.output["rgb"] is None:
            return torch.zeros((env.num_envs, image_shape[0], image_shape[1], 3), device=env.device)
        else:
            return (wp.torch.to_torch(env.camera.data.output["rgb"])[None, :, :, :3]).to(env.device)

    def actions(self, env: ElevatorEnv):
        """Last actions provided to env."""
        return env.actions


class ElevatorRewardManager(RewardManager):
    """Reward manager for single-arm reaching environment."""

    def penalizing_robot_dof_velocity_l2(self, env: ElevatorEnv):
        """Penalize large movements of the robot arm."""
        return torch.sum(torch.square(env.robot.data.arm_dof_vel), dim=1)

    def penalizing_robot_dof_acceleration_l2(self, env: ElevatorEnv):
        """Penalize fast movements of the robot arm."""
        return torch.sum(torch.square(env.robot.data.dof_acc), dim=1)

    def penalizing_action_rate_l2(self, env: ElevatorEnv):
        """Penalize large variations in action commands."""
        return torch.sum(torch.square(env.actions[:, :-1] - env.previous_actions[:, :-1]), dim=1)

    def penalizing_action_l2(self, env: ElevatorEnv):
        """Penalize large actions."""
        return torch.sum(torch.square(env.actions[:, :-1]), dim=1)
    
    def penalizing_collision(self, env:ElevatorEnv):
        """Penalize collision"""
        return (env.rigidContacts.get_net_contact_forces().abs().sum(axis = -1).reshape(env.num_envs,-1).sum(axis = -1)>0.1).to(torch.float32)

    def penalizing_camera_lin_vel_l2(self, env:ElevatorEnv):
        """Penalize camera movement"""
        lin_vel = env.robot.data.ee_state_w[:,7:10]
        return torch.sum(torch.square(lin_vel), dim=1)

    def penalizing_camera_ang_vel_l2(self, env:ElevatorEnv):
        """Penalize camera movement"""
        ang_vel = env.robot.data.ee_state_w[:,10:13]
        return torch.sum(torch.square(ang_vel), dim=1)

    def penalizing_nonflat_camera_l2(self, env:ElevatorEnv):
        axis_z = matrix_from_quat(env.robot.data.ee_state_w[:, 3:7])[:,2,:]
        w = torch.tensor([[0.5,0.,1.]], device = env.device)
        ref = torch.tensor([[1.,0.,0.]], device = env.device)
        return torch.sum(torch.square(axis_z - ref) * w, dim=1)

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
        reward[(elevator_state[:,0]==3) & ((elevator_state[:,2]>0) | (elevator_state[:,3]>0))] = 1.
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
        reward[(elevator_state[:,0]==3) & ((elevator_state[:,2]> 0) | (elevator_state[:,3]> 0))] = 1.
        return reward

    def tracking_reference_enter(self, env: ElevatorEnv, sigma):        
        robot_pos_error = torch.norm(env.robot.data.base_dof_pos[:,:2] - env.robot_des_pose_w[:,:2], dim=1)
        reward = torch.exp(-robot_pos_error / sigma)
        elevator_state = env.elevator._sm_state.to(env.device)
        reward[(elevator_state[:,0] != 1)] = 0.
        return reward

    def tracking_reference_waitin(self, env: ElevatorEnv, sigma):
        robot_pos_error = torch.norm(env.robot.data.base_dof_pos[:,:2] - env.robot_des_pose_w[:,:2], dim=1)
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
        reward[(elevator_state[:,2] == 0) & (elevator_state[:,3] == 0)] = 0. # Neigther btn is pressed
        return reward

