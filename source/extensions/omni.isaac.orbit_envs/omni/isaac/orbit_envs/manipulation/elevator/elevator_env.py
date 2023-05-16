# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gym.spaces
import math
import torch
from math import sqrt

# Modules for Elevator
from typing import Optional, Sequence, Union  # Dict, List, Tuple

import carb
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import ArticulationView
from pxr import Gf

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematics
from omni.isaac.orbit.markers import PointMarker, StaticMarker
from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulator
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import random_orientation, sample_uniform, scale_transform
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs

from .elevator_cfg import ElevatorEnvCfg, RandomizationCfg


import warp as wp

from omni.isaac.orbit.utils.timer import Timer
from enum import Enum
# import omni.isaac.orbit_envs  # noqa: F401
# from omni.isaac.orbit_envs.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


class DoorState(Enum):
    """States for the elevator door."""

    OPEN = wp.constant(1)
    CLOSE = wp.constant(0)


class ElevatorSmState(Enum):
    """States for the elevator state machine."""

    REST = wp.constant(0)
    DOOR_OPENING = wp.constant(1)
    DOOR_CLOSING = wp.constant(2)
    MOVE = wp.constant(3)
    

class ElevatorSmWaitTime(Enum):
    """Additional wait times (in s) for states for before switching."""

    DOOR_OPENING = wp.constant(20.0)
    DOOR_CLOSING = wp.constant(1.0)
    MOVE = wp.constant(0.5)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=wp.float32),
    sm_state: wp.array(dtype=wp.int32),
    sm_wait_time: wp.array(dtype=wp.float32),
    btn_pose: wp.array(dtype=wp.float32),
    door_state: wp.array(dtype=wp.int32)
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == ElevatorSmState.REST.value:
        door_state[tid] = DoorState.CLOSE.value
        # wait for a while
        if btn_pose[tid] < 0.0:
            # move to next state and reset wait time
            sm_state[tid] = ElevatorSmState.DOOR_OPENING.value
            sm_wait_time[tid] = 0.0
    elif state == ElevatorSmState.DOOR_OPENING.value:
        door_state[tid] = DoorState.OPEN.value
        if sm_wait_time[tid] >= ElevatorSmWaitTime.DOOR_OPENING.value:
            # move to next state and reset wait time
            sm_state[tid] = ElevatorSmState.DOOR_CLOSING.value
            sm_wait_time[tid] = 0.0
    elif state == ElevatorSmState.DOOR_CLOSING.value:
        door_state[tid] = DoorState.CLOSE.value
        if sm_wait_time[tid] >= ElevatorSmWaitTime.DOOR_CLOSING.value:
            # move to next state and reset wait time
            sm_state[tid] = ElevatorSmState.MOVE.value
            sm_wait_time[tid] = 0.0
    elif state == ElevatorSmState.MOVE.value:
        door_state[tid] = DoorState.CLOSE.value
        # wait for a while
        if sm_wait_time[tid] >= ElevatorSmWaitTime.MOVE.value:
            # move to next state and reset wait time
            sm_state[tid] = ElevatorSmState.REST.value
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class ElevatorSm:
    """A simple state machine for an elevator.

    The state machine is implemented as a warp kernel. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The elevator is at rest.
    2. DOOR_OPENING: The elevator opens the door when btn pushed.
    3. DOOR_CLOSING: The elevator close the door after waited certain time.
    4. MOVE: The elevator keep close the door.
    """

    def __init__(self, dt: float, num_envs: int, device: Union[torch.device, str] = "cpu"):
        """Initialize the state machine.

        Args:
            dt (float): The environment time step.
            num_envs (int): The number of environments to simulate.
            device (Union[torch.device, str], optional): The device to run the state machine on.
        """
        # save parameters
        self.dt = dt
        self.num_envs = num_envs
        self.device = device
        print("\n\n\nDEVICE:", self.device)
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, dtype=torch.float32, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        # desired state
        self.door_state = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        
        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.door_state_wp = wp.from_torch(self.door_state, wp.int32)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = ...
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, btn_pos: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert to warp
        btn_pos_wp = wp.from_torch(btn_pos.contiguous().to(self.device), wp.float32)
        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                btn_pos_wp,
                self.door_state_wp
            ],
        )
        wp.synchronize()
        # convert to torch
        return self.door_state.bool()

class Elevator:
    """
    simple class for elevator.
    """

    articulations: ArticulationView = None

    def __init__(self, dt: float):
        self._is_spawned = False
        self._door_state = None  # 0: closed, 1: open
        self._door_pos_targets = None
        self._dof_default_pos = None
        self._sm = None
        self._dt = dt

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
        self._door_state = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self._dof_default_targets = self.articulations._physics_view.get_dof_position_targets()
        self._dof_pos = self.articulations.get_joint_positions(indices=self.all_mask, clone=False)
        print("DoF Name", self.articulations.dof_names)
        # DoF Name ['PJoint_LO_Door', 'PJoint_RO_Door', 'PJoint_LI_Door', 'PJoint_RI_Door', 'PJoint_OU_Btn', 'PJoint_OD_Btn']
        print("ELEVATOR DEVICE", self.device)
        self._sm = ElevatorSm(self._dt, self.count, "cuda")

    def setDoorState(self, toopen=True, env_ids: Optional[Sequence[int]] = None):
        """
        This function can be replaced by the state machine. In update_buffers
        """
        if env_ids is None:
            env_ids = self.all_mask
        elif len(env_ids) == 0:
            return
        self._door_state[env_ids] = bool(toopen)
        self._door_pos_targets = (
            torch.Tensor([[1.0, -1.0, 1.0, -1.0]]).to(self.device) * torch.where(self._door_state[..., None], 0.8, 0.0)
        ).to(self.device)
        dof_targets = self._dof_default_targets.clone()
        dof_targets[:, :4] = self._door_pos_targets
        self.articulations._physics_view.set_dof_position_targets(dof_targets, self.all_mask)

    def update_buffers(self, dt: float):
        self._dof_pos[:] = self.articulations.get_joint_positions(indices=self.all_mask, clone=False)
        self._door_state = self._sm.compute(self._dof_pos[:,-1]).to(self.device)
        self._door_pos_targets = (
            torch.Tensor([[1.0, -1.0, 1.0, -1.0]]).to(self.device) * torch.where(self._door_state[..., None], 0.8, 0.0)
        ).to(self.device)
        dof_targets = self._dof_default_targets.clone()
        dof_targets[:, :4] = self._door_pos_targets
        self.articulations._physics_view.set_dof_position_targets(dof_targets, self.all_mask)


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
        self.elevator = Elevator(dt = 1.0 / 60.0) # TODO: get dt from cfg

        # initialize the base class to setup the scene.
        super().__init__(self.cfg, **kwargs)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()

        # prepare the observation manager
        self._observation_manager = ElevatorObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        # prepare the reward manager
        self._reward_manager = ElevatorRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self._reward_manager)

        # compute the observation space
        num_obs = self._observation_manager._group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        print("[INFO]: Completed setting up the environment...")
        # Take an initial step to initialize the scene.
        self.sim.step()
        # -- fill up buffers
        self.robot.update_buffers(self.dt)
        self.elevator.update_buffers(self.dt)

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
        self.elevator.setDoorState(toopen=False, env_ids=env_ids)
        # -- desired end-effector pose
        self._randomize_ee_desired_pose(env_ids, cfg=self.cfg.randomization.ee_desired_pose)

        # -- Reward logging
        # fill extras with episode information
        self.extras["episode"] = dict()
        # reset
        # -- rewards manager: fills the sums for terminated episodes
        self._reward_manager.reset_idx(env_ids, self.extras["episode"])
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
            self._ik_controller.set_command(self.actions[:, self.robot.base_num_dof : -1])
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
            # we assume the first is base command so don't change that
            self.robot_actions[:, : self.robot.base_num_dof] = self.actions[:, : self.robot.base_num_dof]
            # we assume last command is tool action so don't change that
            self.robot_actions[:, -1] = self.actions[:, -1]
        elif self.cfg.control.control_type == "default":
            self.robot_actions[:, :] = self.actions
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
        # -- compute MDP signals
        # reward
        self.reward_buf = self._reward_manager.compute()
        # terminations
        self._check_termination()
        # -- store history
        self.previous_actions = self.actions.clone()

        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        # -- update USD visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            self._debug_vis()

    def _get_observations(self) -> VecEnvObs:
        # compute observations
        return self._observation_manager.compute()

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

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)

        # convert configuration parameters to torch
        # randomization
        # -- desired pose
        config = self.cfg.randomization.ee_desired_pose
        for attr in ["position_uniform_min", "position_uniform_max", "position_default", "orientation_default"]:
            setattr(config, attr, torch.tensor(getattr(config, attr), device=self.device, requires_grad=False))

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()

        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")
        self.elevator.initialize(self.env_ns + "/.*/Elevator")

        # create controller
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller = DifferentialInverseKinematics(
                self.cfg.control.inverse_kinematics, self.robot.count, self.device
            )
            self.num_actions = self.robot.base_num_dof + self._ik_controller.num_actions + 1
        elif self.cfg.control.control_type == "default":
            self.num_actions = self.robot.base_num_dof + self.robot.arm_num_dof + 1

        # history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # robot joint actions
        self.robot_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        # commands
        self.ee_des_pose_w = torch.zeros((self.num_envs, 7), device=self.device)

    def _debug_vis(self):
        # compute error between end-effector and command
        error = torch.sum(torch.square(self.ee_des_pose_w[:, :3] - self.robot.data.ee_state_w[:, 0:3]), dim=1)
        # set indices of the prim based on error threshold
        goal_indices = torch.where(error < 0.002, 1, 0)
        # apply to instance manager
        # -- goal
        self._goal_markers.set_world_poses(self.ee_des_pose_w[:, :3], self.ee_des_pose_w[:, 3:7])
        self._goal_markers.set_status(goal_indices)
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
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)

    def _randomize_ee_desired_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.EndEffectorDesiredPoseCfg):
        """Randomize the desired pose of the end-effector."""
        # -- desired object root position
        if cfg.position_cat == "default":
            # constant command for position
            self.ee_des_pose_w[env_ids, 0:3] = cfg.position_default
        elif cfg.position_cat == "uniform":
            # sample uniformly from box
            # note: this should be within in the workspace of the robot
            self.ee_des_pose_w[env_ids, 0:3] = sample_uniform(
                cfg.position_uniform_min, cfg.position_uniform_max, (len(env_ids), 3), device=self.device
            )
        else:
            raise ValueError(f"Invalid category for randomizing the desired object positions '{cfg.position_cat}'.")
        # -- desired object root orientation
        if cfg.orientation_cat == "default":
            # constant position of the object
            self.ee_des_pose_w[env_ids, 3:7] = cfg.orientation_default
        elif cfg.orientation_cat == "uniform":
            self.ee_des_pose_w[env_ids, 3:7] = random_orientation(len(env_ids), self.device)
        else:
            raise ValueError(
                f"Invalid category for randomizing the desired object orientation '{cfg.orientation_cat}'."
            )
        # transform command from local env to world
        self.ee_des_pose_w[env_ids, 0:3] += self.envs_positions[env_ids]


class ElevatorObservationManager(ObservationManager):
    """Reward manager for single-arm reaching environment."""

    def arm_dof_pos_normalized(self, env: ElevatorEnv):
        """DOF positions for the arm normalized to its max and min ranges."""
        return scale_transform(
            env.robot.data.arm_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, :7, 0],
            env.robot.data.soft_dof_pos_limits[:, :7, 1],
        )

    def arm_dof_vel(self, env: ElevatorEnv):
        """DOF velocity of the arm."""
        return env.robot.data.arm_dof_vel

    def ee_position(self, env: ElevatorEnv):
        """Current end-effector position of the arm."""
        return env.robot.data.ee_state_w[:, :3] - env.envs_positions

    def ee_position_command(self, env: ElevatorEnv):
        """Desired end-effector position of the arm."""
        return env.ee_des_pose_w[:, :3] - env.envs_positions

    def actions(self, env: ElevatorEnv):
        """Last actions provided to env."""
        return env.actions


class ElevatorRewardManager(RewardManager):
    """Reward manager for single-arm reaching environment."""

    def tracking_robot_position_l2(self, env: ElevatorEnv):
        """Penalize tracking position error using L2-kernel."""
        # compute error
        return torch.sum(torch.square(env.ee_des_pose_w[:, :3] - env.robot.data.ee_state_w[:, 0:3]), dim=1)

    def tracking_robot_position_exp(self, env: ElevatorEnv, sigma: float):
        """Penalize tracking position error using exp-kernel."""
        # compute error
        error = torch.sum(torch.square(env.ee_des_pose_w[:, :3] - env.robot.data.ee_state_w[:, 0:3]), dim=1)
        return torch.exp(-error / sigma)

    def penalizing_robot_dof_velocity_l2(self, env: ElevatorEnv):
        """Penalize large movements of the robot arm."""
        return torch.sum(torch.square(env.robot.data.arm_dof_vel), dim=1)

    def penalizing_robot_dof_acceleration_l2(self, env: ElevatorEnv):
        """Penalize fast movements of the robot arm."""
        return torch.sum(torch.square(env.robot.data.dof_acc), dim=1)

    def penalizing_action_rate_l2(self, env: ElevatorEnv):
        """Penalize large variations in action commands."""
        return torch.sum(torch.square(env.actions[:, :-1] - env.previous_actions[:, :-1]), dim=1)
