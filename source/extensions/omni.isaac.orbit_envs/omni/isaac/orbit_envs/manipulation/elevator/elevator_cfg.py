# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematicsCfg
# from omni.isaac.orbit.robots.config.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG
from omni.isaac.orbit.robots.config.alma import ALMA_CFG
from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulatorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR, ASSET_NUCLEUS_DIR
from omni.isaac.assets import ASSETS_DATA_DIR
from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, SimCfg, ViewerCfg
import math
##
# Scene settings
##


@configclass
class MarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.1, 0.1, 0.1]  # x,y,z


@configclass
class ButtonPanelCfg:
    """Properties for the button panel in the scene
        cfg.buttonPanel
    """
    nx = 3
    ny = 4
    grid_size = 0.1
    usd_path = f"{ASSETS_DATA_DIR}/objects/elevator/button_obj.usd"
    usd_symbol_root =  f"{ASSETS_DATA_DIR}/objects/elevator/text_icons"
    symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "up", "down"]
    translation = (0, -0.75, 1.2)
    orientation = (0.0, 0.0, math.sqrt(1 / 2), math.sqrt(1 / 2))
    btn_light_cond = 5 # light on the button with continuous pushing of 5 frames

##
# MDP settings
##


@configclass
class InitializationCfg:
    """configuration of the initialization of the Env.
       cfg.initialization
    """

    @configclass
    class RobotPosCfg:
        """Initposition of the robot.
           cfg.initialization.robot
        """
        # category
        position_cat: str = "default"  # randomize position: "default", "uniform", "see-point"
        # default: use the default position
        # uniform: randomize position in the uniform range
        # see-point: randomize position in the uniform range and ensure the robot can see the point

        # randomize position
        position_uniform_min = [0.9, 0.3, -0.3, -3.1]  # position (x,y,z,yaw)
        position_uniform_max = [2.1, 2.,   0.1, 0.]  # position (x,y,z,yaw)
        # see point configs
        see_point_target = [1.5, 1.5]  # target point (x,y)
        see_point_FOV = 0.5  # half field of view of the robot (in radian)

    @configclass
    class ButtonPanelCfg:
        """Init settings for the buttons
            cfg.initialization.buttonPanel
        """
        num_target_max = 1 # the max number of buttons with semantic: button_target

    @configclass
    class ElevatorStateCfg:
        """Initial state of the elevator
            cfg.initialization.elevator
        """
        moving_elevator_prob = 0.4
        nonzero_floor_prob = 1.
        max_init_wait_time = 25.
        max_init_floor = 20
        hold_button_threshold = 10

    @configclass
    class SceneCfg:
        """Randomization with replicator and Randomization of obs frame
            cfg.initialization.scene
        """
        # the range of the random pose of the obs frame
        obs_frame_bias_range = [40.0, 40.0, math.pi]
        obs_frame_bias_use_init = True
        enable_replicator = True
        randomize_ground_materials = [
            f"{ASSET_NUCLEUS_DIR}/NVIDIA/Materials/Base/Masonry/Concrete_Rough.mdl",
            f"{ASSET_NUCLEUS_DIR}/NVIDIA/Materials/Base/Masonry/Concrete_Polished.mdl",
            f"{ASSET_NUCLEUS_DIR}/NVIDIA/Materials/Base/Stone/Slate.mdl",
            f"{ASSET_NUCLEUS_DIR}/NVIDIA/Materials/Base/Stone/Retaining_Block.mdl",
            f"{ASSET_NUCLEUS_DIR}/NVIDIA/Materials/Base/Stone/Retaining_Block.mdl",
            f"{ASSET_NUCLEUS_DIR}/NVIDIA/Materials/Base/Stone/Granite_Light.mdl"
        ]
        randomize_wall_materials = [
            f"{ASSET_NUCLEUS_DIR}/NVIDIA/Materials/Base/Masonry/Stucco.mdl",
            f"{ASSET_NUCLEUS_DIR}/NVIDIA/Materials/vMaterials_2/Concrete/Concrete_Wall_Aged.mdl",
            f"{ASSET_NUCLEUS_DIR}/NVIDIA/Materials/vMaterials_2/Concrete/Concrete_Wall_Even.mdl"
        ]
        randomize_door_materials = [
            f"{ASSET_NUCLEUS_DIR}/NVIDIA/Materials/vMaterials_2/Metal/Zinc_Brushed.mdl",
            f"{ASSET_NUCLEUS_DIR}/NVIDIA/Materials/vMaterials_2/Metal/Iron_Brushed.mdl"            
        ]

    # initialize
    robot: RobotPosCfg = RobotPosCfg()
    buttonPanel:ButtonPanelCfg = ButtonPanelCfg()
    elevator: ElevatorStateCfg = ElevatorStateCfg()
    scene: SceneCfg = SceneCfg()

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.
        cfg.observations
    """

    @configclass
    class LowDimCfg:
        """Observations for low dimension."""

        # global group settings
        enable_corruption: bool = False
        # observation terms
        dof_pos_obsframe = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.01, "max": 0.01},
                "normalizer": { # the normalizer of dof pos, should match robomimic conterparts
                    "mean": [0, 0, 0.0,   0, 0, 0.5,  0, 0, -0.5,  1.5],
                    "std":  [1, 1, 1.0, 1.0, 2,   2,  2, 2, 16.0, 16.0]
                }
        }
        dof_vel_obsframe = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.1, "max": 0.1}}
        ee_position_obsframe = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.1, "max": 0.1},
                    "normalizer": { # the normalizer of dof pos, should match robomimic conterparts
                        "mean": [0, 0, 0.6],
                        "std":  [1, 1, 1.0]
                    }
        }
        actions = {}

    @configclass
    class PrivilegeCfg:
        """Observations for privileged information."""
        enable_corruption: bool = False
        # observation terms
        dof_pos = {"scale": 1.0}
        dof_vel = {"scale": 0.5}
        ee_position = {}
        # elevator
        elevator_state = {}
        elevator_waittime = {}
        elevator_is_zerofloor = {}
        elevator_btn_pressed = {}
        # keypoints
        keypoint_outbtn_obsframe = {}
        keypoint_outdoor_obsframe = {}
        keypoint_inbtn_obsframe = {}

    @configclass
    class RGBCfg:
        hand_camera_rgb = {}
        base_camera_rgb = {}

    @configclass
    class SemanticCfg:
        """
        semantic
        cfg.observations.semantic
        """
        hand_camera_semantic = {"class_names":[
            ("button_target",), # first channel
            ("button_panel", "button", "button_target") # second channel
        ]}
        base_camera_semantic = {"class_names":[
            ("button_target",), # first channel
            ("button_panel", "button", "button_target") # second channel
        ]}

    @configclass
    class DebugCfg:
        debug_info = {}
        obs_shift_w = {}

    @configclass
    class GoalCfg:
        """
        This is quite different from the other observations.
        It is a fixed obs over the whole episode. And getting it takes heavy computation.
        By specifying the cfg here, it load the cached value from @random_goal_image every step .
        However, for the sake of storage, disable this and use the @random_goal_image in demoCollection.
        """
        goal_hand_rgb = {}
        goal_hand_semantic = {}
        goal_base_rgb = {}
        goal_base_semantic = {}

    @configclass
    class GoalLowdimCfg:
        goal_dof_pos = {}

    # global observation settings
    return_dict_obs_in_group = True
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    low_dim: LowDimCfg = LowDimCfg()
    debug: DebugCfg = DebugCfg()
    rgb: RGBCfg = RGBCfg()
    privilege: PrivilegeCfg = PrivilegeCfg()
    semantic: SemanticCfg = SemanticCfg()
    goal: GoalCfg = GoalCfg()
    goal_lowdim: GoalLowdimCfg = GoalLowdimCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    only_positive_rewards = True
    penalizing_robot_dof_velocity_l2 = {"weight": -0.1}  # -1e-4
    penalizing_robot_dof_acceleration_l2 = {"weight": -2e-5}
    # penalizing_action_rate_l2 = {"weight": -0.1}
    penalizing_action_l2 = {"weight": -0.1}
    penalizing_collision = {"weight": -1.}

    tracking_reference_button_pos = {"weight": 5., "sigma": 2.}
    tracking_reference_button_rot = {"weight": 3., "sigma": 0.1}
    tracking_reference_enter = {"weight": 8., "sigma": 3}
    tracking_reference_waitin = {"weight": 16., "sigma": 0.5}
    tracking_reference_waitout = {"weight": 6., "sigma": 2.}

    penalizing_camera_lin_vel_l2 = {"weight": -0.2}
    penalizing_camera_ang_vel_l2 = {"weight": -0.2} # Don't make it any larger
    penalizing_nonflat_camera_l2 = {"weight": -0.5}
    # look_at_moving_direction_exp = {"weight": -0.02, "sigma": 0.1}


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.
        cfg.terminations
    """
    episode_timeout = True  # reset when episode length ended
    is_success = True  # reset when `task_condition` is satisfied
    task_condition = "enter_elevator"  # the success condition of the task
    enter_elevator_threshold = 0.5  # distance to elevator center
    move_to_button_thresholds = [0.15, 0.05]  # distance to desired position and desired yaw
    collision = True  # reset when robot collides with the elevator
    extra_conditions = []
    hasdone_pushbtn_threshold = 8 # how long the button have to be hold
    hasdone_pushCorrect_threshold = 8 # how long the button have to be hold
    hasdone_pushWrong_threshold = 8 # how long the button have to be hold

@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # action space
    control_type = "default"  # "default", "inverse_kinematics", "base"
    substract_action_from_obs_frame = True
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 2
    # command types:
    #  "all_pos": all joins are p_abs control
    #  "xy_vel": only the base_xy is v_abs control, the rest are p_abs control
    command_type = "xy_vel"
    # command_type = "all_pos"


    # configuration loaded when control_type == "inverse_kinematics"
    inverse_kinematics: DifferentialInverseKinematicsCfg = DifferentialInverseKinematicsCfg(
        command_type="pose_abs",
        ik_method="dls",
    )


##
# Environment configuration
##


@configclass
class ElevatorEnvCfg(IsaacEnvCfg):
    """Configuration for the reach environment."""

    # General Settings
    # env: EnvCfg = EnvCfg(num_envs=2048, env_spacing=2.5, episode_length_s=4.0)
    env: EnvCfg = EnvCfg(num_envs=16, env_spacing=16, episode_length_s=30.0)
    viewer: ViewerCfg = ViewerCfg(debug_vis=False, eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
    # Physics settings
    sim: SimCfg = SimCfg(dt=1.0 / 60.0, substeps=1)

    # Scene Settings
    robot: MobileManipulatorCfg = ALMA_CFG
    marker: MarkerCfg = MarkerCfg()
    buttonPanel: ButtonPanelCfg = ButtonPanelCfg()

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()

    # Initialization settings
    initialization: InitializationCfg = InitializationCfg()

    # workflow settings
    # A list of tuples {group_name: [list of observation names]} to be used for observation grouping
    observation_grouping = {}
