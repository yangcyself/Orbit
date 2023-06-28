

def modify_cfg_to_task_push_btn(cfg):
    cfg.initialization.robot.position_cat = "uniform"
    cfg.initialization.robot.position_uniform_min = [-0.3, -0.1, 0.4, -1.8]
    cfg.initialization.robot.position_uniform_max = [ 0.3,  0.3, 0.6, -1.2]

