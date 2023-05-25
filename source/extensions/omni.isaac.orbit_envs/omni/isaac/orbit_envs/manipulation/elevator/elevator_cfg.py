# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematicsCfg
from omni.isaac.orbit.robots.config.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG
from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulatorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, SimCfg, ViewerCfg

##
# Scene settings
##


@configclass
class TableCfg:
    """Properties for the table."""

    # note: we use instanceable asset since it consumes less memory
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"


@configclass
class MarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.1, 0.1, 0.1]  # x,y,z


##
# MDP settings
##


@configclass
class InitializationCfg:
    """configuration of the initialization of the Env."""

    @configclass
    class RobotPosCfg:
        """Initposition of the robot."""
        # category
        position_cat: str = "uniform"  # randomize position: "default", "uniform"
        # randomize position
        position_uniform_min = [0.9, 0.3, -3.1]  # position (x,y,z)
        position_uniform_max = [2.1, 2., 0.]  # position (x,y,z)

    # initialize
    robot: RobotPosCfg = RobotPosCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class LowDimCfg:
        """Observations for low dimension."""

        # global group settings
        enable_corruption: bool = True
        # observation terms
        dof_pos_normalized = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        dof_vel = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.1, "max": 0.1}}
        ee_position = {}
        actions = {}

    @configclass
    class PrivilegeCfg:
        """Observations for privileged information."""
        enable_corruption: bool = False
        # observation terms
        dof_pos_normalized = {"scale": 1.0}
        dof_vel = {"scale": 0.5}
        ee_position = {}
        # actions = {}
        elevator_state = {}
        elevator_waittime = {}

    @configclass
    class RGBCfg:
        hand_camera_rgb = {}


    # global observation settings
    policy: PrivilegeCfg = PrivilegeCfg()
    return_dict_obs_in_group = True
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    low_dim: LowDimCfg = LowDimCfg()
    # rgb: RGBCfg = RGBCfg()
    # privilege: PrivilegeCfg = PrivilegeCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    penalizing_robot_dof_velocity_l2 = {"weight": -0.02}  # -1e-4
    penalizing_robot_dof_acceleration_l2 = {"weight": -1e-5}
    # penalizing_action_rate_l2 = {"weight": -0.1}
    penalizing_action_l2 = {"weight": -0.5}
    penalizing_collision = {"weight": -1.}
    tracking_reference_points = {"weight": 2., "sigma": 0.5}


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    episode_timeout = True  # reset when episode length ended
    is_success = True  # reset when robot is in elevator
    collision = False  # reset when robot collides with the elevator


@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # action space
    control_type = "default"  # "default", "inverse_kinematics"
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 2

    # configuration loaded when control_type == "inverse_kinematics"
    inverse_kinematics: DifferentialInverseKinematicsCfg = DifferentialInverseKinematicsCfg(
        command_type="pose_rel",
        ik_method="dls",
        position_command_scale=(0.1, 0.1, 0.1),
        rotation_command_scale=(0.1, 0.1, 0.1),
    )


##
# Environment configuration
##


@configclass
class ElevatorEnvCfg(IsaacEnvCfg):
    """Configuration for the reach environment."""

    # General Settings
    # env: EnvCfg = EnvCfg(num_envs=2048, env_spacing=2.5, episode_length_s=4.0)
    env: EnvCfg = EnvCfg(num_envs=16, env_spacing=16, episode_length_s=50.0)
    viewer: ViewerCfg = ViewerCfg(debug_vis=False, eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
    # Physics settings
    sim: SimCfg = SimCfg(dt=1.0 / 60.0, substeps=1)

    # Scene Settings
    robot: MobileManipulatorCfg = RIDGEBACK_FRANKA_PANDA_CFG
    table: TableCfg = TableCfg()
    marker: MarkerCfg = MarkerCfg()

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()

    # Initialization settings
    initialization: InitializationCfg = InitializationCfg()
