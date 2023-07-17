from omni.isaac.orbit.utils import configclass
from dataclasses import MISSING
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

@configclass
class ButtonTaskPresetCfg:
    """The configuration that defines a set of parameters for a task. 
    """

    """The position of the button in the world frame."""
    button_pos = MISSING

    """The orientation of the button in the world frame."""
    button_yaw = MISSING

    """The init position range for push"""
    push_position_min = MISSING
    push_position_max = MISSING

    """The init position range for moveto"""
    moveto_position_min = MISSING
    moveto_position_max = MISSING

    """The viewer camera position"""
    viewer_eye = MISSING
    viewer_look = MISSING

BUTTON_TASK_OUT_RIGHT = ButtonTaskPresetCfg(
    button_pos = (0, -0.75, 1.2),
    button_yaw = -math.pi/2, # (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    push_position_min = [-0.3, -0.1, 0.4, -math.pi/2 - 0.25],
    push_position_max = [ 0.3,  0.3, 0.6, -math.pi/2 + 0.25],
    moveto_position_min = [-3,  1, 0.4],
    moveto_position_max = [ 3,  6, 0.6],
    viewer_eye = [4.0, 4.0, 4.0],
    viewer_look = [0.5, 0.0, 0.0]
)

BUTTON_TASK_OUT_LEFT = ButtonTaskPresetCfg(
    button_pos = (2.8, -0.75, 1.2),
    button_yaw = -math.pi/2, # (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    push_position_min = [3.1, -0.1, 0.4, -math.pi/2 - 0.25],
    push_position_max = [2.5,  0.3, 0.6, -math.pi/2 + 0.25],
    moveto_position_min = [-3,  1, 0.4],
    moveto_position_max = [ 3,  6, 0.6],
    viewer_eye =  [0.0, 4.0, 4.0],
    viewer_look = [2.5, 0.0, 0.0]
)

BUTTON_TASK_IN_RIGHT = ButtonTaskPresetCfg(
    button_pos = (0.34, -1.6, 1.3),
    button_yaw = -math.pi, # (0.5, 0.5,  0.5,  0.5)
    push_position_min = [0.8, -1.9,  0.4, -math.pi - 0.25],
    push_position_max = [1.5, -1.5, 0.6, -math.pi + 0.25],
    moveto_position_min = [0.9 ,  -1.5,  0.4],
    moveto_position_max = [2.15,  -2.7,  0.6],
    viewer_eye =  [2.45, -3.1, 3.2],
    viewer_look = [0.34, -1,   0.2]
)

BUTTON_TASK_IN_LEFT = ButtonTaskPresetCfg(
    button_pos = (2.68, -1.6, 1.2),
    button_yaw = 0, # (-0.5, -0.5,  0.5,  0.5)
    push_position_min = [1.5, -1.9, 0.4, 0.25],
    push_position_max = [2.15, -1.5, 0.6, 0.25],
    moveto_position_min = [0.9 ,  -1.5,  0.4],
    moveto_position_max = [2.15,  -2.7,  0.6],
    viewer_eye =  [0.35, -3.1, 3.2],
    viewer_look = [2.68, -1,   0.2]
)

@configclass
class ButtonPanelPresetCfg:
    """The configs for the button panel."""
    nx = MISSING
    ny = MISSING
    grid_size = MISSING
    symbols = MISSING


## Train set: Large button panel with a lot of symbols
BUTTON_PANEL_01 = ButtonPanelPresetCfg(
    nx = 3,
    ny = 4,
    grid_size = 0.1,
    symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "up", "down"]
)

## Train set: Small button panel with only 3 symbols
BUTTON_PANEL_02 = ButtonPanelPresetCfg(
    nx = 1,
    ny = 3,
    grid_size = 0.2,
    symbols = ["down", "up", "E"]
)

## Test set: Button panel inside the elevator
BUTTON_PANEL_03 = ButtonPanelPresetCfg(
    nx = 2,
    ny = 3,
    grid_size = 0.15,
    symbols = ["down", "1", "3", "up", "2", "4"]
)

## Test set: Button panel outside the elevator
BUTTON_PANEL_04 = ButtonPanelPresetCfg(
    nx = 1,
    ny = 2,
    grid_size = 0.2,
    symbols = ["down", "up"]
)

def panel_yaw_to_quat(yaw):
    base_rot = R.from_quat([-0.5, -0.5,  0.5,  0.5])
    base_rot_matrix = base_rot.as_matrix()
    y_rotation = R.from_euler('y', -yaw, degrees=False)
    y_rotation_matrix = y_rotation.as_matrix()
    return R.from_matrix(y_rotation_matrix @ base_rot_matrix).as_quat()

def modify_cfg_according_to_button_panel(cfg, button_panel):
    if button_panel.lower() == "panel01":
        panel_cfg = BUTTON_PANEL_01
    elif button_panel.lower() == "panel02":
        panel_cfg = BUTTON_PANEL_02
    elif button_panel.lower() == "panel03":
        panel_cfg = BUTTON_PANEL_03
    elif button_panel.lower() == "panel04":
        panel_cfg = BUTTON_PANEL_04
    else:
        raise NotImplementedError(f"Button panel {button_panel} not implemented.")

    cfg.buttonPanel.nx = panel_cfg.nx
    cfg.buttonPanel.ny = panel_cfg.ny
    cfg.buttonPanel.grid_size = panel_cfg.grid_size
    cfg.buttonPanel.symbols = panel_cfg.symbols
    return panel_cfg

def modify_cfg_according_to_button_task(cfg, button_task, task):
    if button_task.lower() == "out_left":
        button_cfg = BUTTON_TASK_OUT_LEFT
    elif button_task.lower() == "out_right":
        button_cfg = BUTTON_TASK_OUT_RIGHT
    elif button_task.lower() == "in_left":
        button_cfg = BUTTON_TASK_IN_LEFT
    elif button_task.lower() == "in_right":
        button_cfg = BUTTON_TASK_IN_RIGHT
    else:
        raise NotImplementedError(f"Button task {button_task} not implemented.")

    cfg.buttonPanel.translation = button_cfg.button_pos
    cfg.buttonPanel.orientation = panel_yaw_to_quat(button_cfg.button_yaw)
    cfg.viewer.eye = button_cfg.viewer_eye
    cfg.viewer.lookat = button_cfg.viewer_look
    
    if task.lower() == "pushbtn":
        cfg.initialization.robot.position_uniform_min = button_cfg.push_position_min
        cfg.initialization.robot.position_uniform_max = button_cfg.push_position_max
    elif task.lower() == "movetobtn":
        cfg.initialization.robot.position_uniform_min = button_cfg.moveto_position_min
        cfg.initialization.robot.position_uniform_max = button_cfg.moveto_position_max
        cfg.initialization.robot.see_point_target = button_cfg.button_pos[:2]
    else:
        raise NotImplementedError(f"Task {task} not implemented.")
    return button_cfg

def modify_cfg_according_to_task(cfg, task):
    if task.lower() == "pushbtn":
        modify_cfg_to_task_push_btn(cfg)
    elif task.lower() == "movetobtn":
        modify_cfg_to_task_move_to_btn(cfg)
    else:
        raise NotImplementedError(f"Task {task} not implemented.")

def modify_cfg_to_task_push_btn(cfg):
    cfg.initialization.robot.position_cat = "uniform"
    cfg.terminations.is_success = True
    cfg.terminations.task_condition = "pushed_perfect"
    cfg.terminations.extra_conditions = ["pushed_btn"]
    cfg.terminations.collision = True
    cfg.terminations.episode_timeout = True
    cfg.robot.rigid_props.disable_gravity = True # copied from ik related config
    cfg.control.control_type = "default"
    cfg.control.command_type = "all_pos"

def modify_cfg_to_task_move_to_btn(cfg):
    cfg.initialization.robot.position_cat = "see-point"
    cfg.initialization.robot.see_point_target = [0, -0.7]
    cfg.initialization.robot.see_point_FOV = 0.8
    cfg.terminations.is_success = False
    cfg.terminations.task_condition = "moveto_button"
    cfg.terminations.collision = True
    cfg.terminations.episode_timeout = True
    cfg.robot.rigid_props.disable_gravity = True # copied from ik related config
    cfg.control.control_type = "base"
    cfg.control.command_type = "xy_vel"
    cfg.observations.semantic.hand_camera_semantic = {
        "class_names":[("button_panel","button", "button_target")]
    }
    cfg.observations.semantic.base_camera_semantic = {
        "class_names":[("button_panel","button", "button_target")]
    }

def modify_cfg_to_robomimic(cfg):
    cfg.observations.return_dict_obs_in_group = True
    cfg.observation_grouping = {
        "policy":"privilege", 
        "rgb":None, 
        "low_dim":None, 
        "goal":["goal","goal_lowdim"],
        "semantic":None,
    }
    cfg.control.substract_action_from_obs_frame = True
    cfg.robot.rigid_props.disable_gravity = True # copied from ik related config

def modify_cfg_to_simServer(cfg):
    """
    The environment configuration for sim_robot_server
    Where we have raw obs and process obs in the tasks
    """
    modify_cfg_to_robomimic(cfg)

    cfg.initialization.robot.position_cat = "see-point"
    cfg.initialization.robot.position_uniform_min = [-3,  1, 0.4]
    cfg.initialization.robot.position_uniform_max = [ 3,  6,   0.6]
    cfg.initialization.robot.see_point_target = [0, -0.7]
    cfg.initialization.robot.see_point_FOV = 0.8

    cfg.control.control_type = "default"
    cfg.terminations.collision = False
    cfg.terminations.is_success = False
    cfg.terminations.episode_timeout = False

    cfg.observation_grouping.update({"debug":None})
    del cfg.observation_grouping["goal"]

    cfg.observations.low_dim.dof_pos_obsframe["normalizer"] = {
        "mean": [0.0]*10,
        "std":  [1.0]*10
    }

    cfg.observations.low_dim.ee_position_obsframe["normalizer"] = {
        "mean": [0.0]*3,
        "std":  [1.0]*3
    }

