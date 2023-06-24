# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the DynaArm robots.

The following configurations are available:

* :obj:`DYNAARM_CFG`: DynaArm arm without a gripper.
"""


import os
from pathlib import Path

from omni.isaac.orbit.actuators.config.dynadrive import BABOON_ACTUATOR_CFG, COYOTE_ACTUATOR_CFG
from omni.isaac.orbit.actuators.group import ActuatorGroupCfg
from omni.isaac.orbit.actuators.group.actuator_group_cfg import ActuatorControlCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg

from ..single_arm import SingleArmManipulatorCfg

_DYNAARM_INSTANCEABLE_USD = os.path.join(
    Path(__file__).parents[6], "omni.isaac.assets", "data", "robots", "anybotics", "dynaarm", "dynaarm.usd"
)

DYNAARM_CFG = SingleArmManipulatorCfg(
    meta_info=SingleArmManipulatorCfg.MetaInfoCfg(
        usd_path=_DYNAARM_INSTANCEABLE_USD,
        arm_num_dof=6,
        tool_num_dof=0,
        tool_sites_names=None,
    ),
    init_state=SingleArmManipulatorCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.01), what is this?
        dof_pos={
            "SH_ROT": 0.0,
            "SH_FLE": -0.7,
            "EL_FLE": 1.4,
            "FA_ROT": 0.0,
            "WRIST_1": 0.0,
            "WRIST_2": 0.0,
        },
        dof_vel={".*": 0.0},
    ),
    # TODO (kaixqu): Why I cannot use dynaarm_END_EFFECTOR here? I am using the second last frame
    ee_info=SingleArmManipulatorCfg.EndEffectorFrameCfg(body_name="gripper_base"),
    # TODO (kaixqu): Switch to the real actuator model
    actuator_groups={
        "dynaarm_arm": ActuatorGroupCfg(
            dof_names=["SH_ROT", "SH_FLE", "EL_FLE"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=87.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 800.0},
                damping={".*": 40.0},
            ),
        ),
        "dynaarm_wrist": ActuatorGroupCfg(
            dof_names=["FA_ROT", "WRIST_1", "WRIST_2"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=12.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 800.0},
                damping={".*": 40.0},
            ),
        ),
    },
    # actuator_groups={
    #     "dynaarm_arm": ActuatorGroupCfg(
    #         dof_names=["SH_ROT", "SH_FLE", "EL_FLE"],
    #         model_cfg=BABOON_ACTUATOR_CFG,
    #         control_cfg=ActuatorControlCfg(
    #             command_types=["v_abs"],
    #             stiffness={".*": 1e5},
    #             damping={".*": 1e3},
    #         ),
    #     ),
    #     "dynaarm_wrist": COYOTE_ACTUATOR_CFG(
    #         dof_names=["FA_ROT", "WRIST_1", "WRIST_2"],
    #         model_cfg=coyote_actuator,
    #         control_cfg=ActuatorControlCfg(
    #             command_types=["v_abs"],
    #             stiffness={".*": 1e5},
    #             damping={".*": 1e3},
    #         ),
    #     ),
    # },
)
"""Configuration of DynaArm."""
