

def modify_cfg_to_task_push_btn(cfg):
    cfg.initialization.robot.position_cat = "uniform"
    cfg.initialization.robot.position_uniform_min = [-0.3, -0.1, 0.4, -1.8]
    cfg.initialization.robot.position_uniform_max = [ 0.3,  0.3, 0.6, -1.2]
    cfg.terminations.is_success = True
    cfg.terminations.task_condition = "pushed_perfect"
    cfg.terminations.extra_conditions = ["pushed_btn"]
    cfg.terminations.collision = True
    cfg.terminations.episode_timeout = True
    cfg.robot.rigid_props.disable_gravity = True # copied from ik related config
    cfg.control.control_type = "default"

def modify_cfg_to_task_move_to_btn(cfg):
    cfg.initialization.robot.position_cat = "see-point"
    cfg.initialization.robot.position_uniform_min = [-3,  1, 0.4]
    cfg.initialization.robot.position_uniform_max = [ 3,  6,   0.6]
    cfg.initialization.robot.see_point_target = [0, -0.7]
    cfg.initialization.robot.see_point_FOV = 0.8

    cfg.terminations.is_success = False
    cfg.terminations.task_condition = "moveto_button"
    cfg.terminations.collision = True
    cfg.terminations.episode_timeout = True
    cfg.robot.rigid_props.disable_gravity = True # copied from ik related config
    cfg.control.control_type = "base"
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

