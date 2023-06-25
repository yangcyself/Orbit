# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of Franka arm with Franka Hand on a Clearpath Ridgeback base using implicit actuator models.

The following control configuration is used:

* Base: position control base which uses actuator network to control the base.
* Arm: position control with damping (contains default position offsets)
* Hand: mimic control

"""

# python
import os
from scipy.spatial.transform import Rotation

from omni.isaac.assets import ASSETS_DATA_DIR

from omni.isaac.orbit.actuators.config.anydrive import ANYMAL_D_DEFAULT_GROUP_CFG
from omni.isaac.orbit.actuators.config.dynadrive import BABOON_ACTUATOR_CFG, COYOTE_ACTUATOR_CFG

# orbit
from omni.isaac.orbit.actuators.group import ActuatorControlCfg, ActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg

from ..mobile_manipulator import LeggedMobileManipulatorCfg
from .anymal import ANYMAL_D_CFG

##
# Helper functions
##


def quat_from_euler_rpy(roll, pitch, yaw, degrees=False):
    """Converts Euler XYZ to Quaternion (w, x, y, z)."""
    quat = Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=degrees).as_quat()
    return tuple(quat[[3, 0, 1, 2]].tolist())


def euler_rpy_apply(rpy, xyz, degrees=False):
    """Applies rotation from Euler XYZ on position vector."""
    rot = Rotation.from_euler("xyz", rpy, degrees=degrees)
    return tuple(rot.apply(xyz).tolist())


##
# Configuration
##


_ALMA_INSTANCEABLE_USD = os.path.join(ASSETS_DATA_DIR, "robots", "anybotics",  "alma", "alma_instanceable.usd")

ALMA_CFG = LeggedMobileManipulatorCfg(
    meta_info=LeggedMobileManipulatorCfg.MetaInfoCfg(
        usd_path=_ALMA_INSTANCEABLE_USD,
        soft_dof_pos_limit_factor=0.95,
        base_num_dof=4,
        arm_num_dof=6,
        tool_num_dof=0,
        tool_sites_names=None,
    ),
    feet_info=ANYMAL_D_CFG.feet_info,
    ee_info=LeggedMobileManipulatorCfg.EndEffectorFrameCfg(body_name="dynaarm_WRIST_2"),
    init_state=LeggedMobileManipulatorCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        dof_pos={
            # base
            "L[F,H]_HAA": -0.1,  # both left HAA
            "R[F,H]_HAA": 0.1,  # both right HAA
            ".*F_HFE": 0.7,  # both front HFE
            ".*H_HFE": -0.7,  # both hind HFE
            ".*F_KFE": -1.0,  # both front KFE
            ".*H_KFE": 1.0,  # both hind KFE
            # dynaarm
            "SH_ROT": 0.0,
            "SH_FLE": -0.7,
            "EL_FLE": 1.4,
            "FA_ROT": 0.0,
            "WRIST_1": 0.0,
            "WRIST_2": 0.0,
        },
        dof_vel={".*": 0.0},
    ),
    rigid_props=LeggedMobileManipulatorCfg.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    articulation_props=LeggedMobileManipulatorCfg.ArticulationRootPropertiesCfg(
        enable_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
    ),
    actuator_groups={
        # base
        # "base_legs": ANYMAL_D_DEFAULT_GROUP_CFG,
        "base": ActuatorGroupCfg(
            dof_names=["world_body.*"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=5e7, torque_limit=1e7),
            control_cfg=ActuatorControlCfg(command_types=["p_abs"], stiffness={".*": 5e4}, damping={".*": 1e4}),
        ),
        # arm
        "dynaarm_arm": ActuatorGroupCfg(
            dof_names=["SH_ROT", "SH_FLE", "EL_FLE"],
            # model_cfg=BABOON_ACTUATOR_CFG,  
            model_cfg = ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=27.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs", "v_abs", "t_abs"],
                stiffness={".*": 200.0},
                damping={".*": 40.0},
                dof_pos_offset={
                    "SH_ROT": 0.0,
                    "SH_FLE": -1.0, # 0.0,  # -0.7,
                    "EL_FLE": 2.4 # 0.0,  # 1.4,
                },
            ),
        ),
        "dynaarm_wrist": ActuatorGroupCfg(
            dof_names=["FA_ROT", "WRIST_1", "WRIST_2"],
            # model_cfg=COYOTE_ACTUATOR_CFG,  
            model_cfg = ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=14.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs", "v_abs", "t_abs"],
                stiffness={".*": 200.0},
                damping={".*": 40.0},
                dof_pos_offset={
                    "FA_ROT": 0.0,
                    "WRIST_1": 0.0,
                    "WRIST_2": 0.0,
                },
            ),
        ),
    },
)
