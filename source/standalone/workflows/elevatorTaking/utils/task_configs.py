from omni.isaac.orbit.utils import configclass

@configclass 
class PolicyCfg:
    checkpoint: str = None
    mimic_update = {}

@configclass
class ObservationCfg:
    dof_pos_obsframe = {}
    ee_position_obsframe = {}
    goal_dof_pos = {}

@configclass
class PushBtnTaskCfg:
    observations: ObservationCfg = ObservationCfg(
        dof_pos_obsframe = {
            "normalizer": { # the normalizer of dof pos, should match robomimic conterparts
                "mean": [0, 0, 0.6,   0, 0, 0.5,  0, 0, -0.5,  1.5],
                "std":  [1, 1, 1.0, 1.0, 2,   2,  2, 2, 16.0, 16.0]
            }
        },
        ee_position_obsframe = {
            "normalizer": { # the normalizer of dof pos, should match robomimic conterparts
                "mean": [0, 0, 0.6],
                "std":  [1, 1, 1.0]
            }
        },
        goal_dof_pos = {}
    )
    policy: PolicyCfg = PolicyCfg()
    
@configclass
class MovetoButtonTaskCfg:
    observations: ObservationCfg = ObservationCfg(
        dof_pos_obsframe = {
            "normalizer": { # the normalizer of dof pos, should match robomimic conterparts
                "mean": [0, 0, 0.6,   0, 0, 0.5,  0, 0, -0.5,  1.5],
                "std":  [1, 1, 1.0, 1.0, 2,   2,  2, 2, 16.0, 16.0]
            }
        },
        ee_position_obsframe = {
            "normalizer": { # the normalizer of dof pos, should match robomimic conterparts
                "mean": [0, 0, 0.6],
                "std":  [1, 1, 1.0]
            }
        },
        goal_dof_pos = {}
    )
    policy: PolicyCfg = PolicyCfg()


@configclass
class SimRobotServerCfg:
    movetoBtn: MovetoButtonTaskCfg = MovetoButtonTaskCfg()
    pushBtn: PushBtnTaskCfg = PushBtnTaskCfg()

