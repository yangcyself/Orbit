

def modify_cfg_to_task_push_btn(cfg):
    cfg.initialization.robot.position_cat = "uniform"
    cfg.initialization.robot.position_uniform_min = [-0.3, -0.1, 0.4, -1.8]
    cfg.initialization.robot.position_uniform_max = [ 0.3,  0.3, 0.6, -1.2]
    cfg.terminations.is_success = "pushed_perfect"
    cfg.terminations.extra_conditions = ["pushed_btn"]
    cfg.terminations.collision = True
    cfg.terminations.episode_timeout = True

def modify_cfg_to_robomimic(cfg):
    cfg.observations.return_dict_obs_in_group = True
    cfg.control.control_type = "default"
    cfg.observation_grouping = {
        "policy":"privilege", 
        "rgb":None, 
        "low_dim":None, 
        "goal":["goal","goal_lowdim"]
    }
    cfg.control.substract_action_from_obs_frame = True
