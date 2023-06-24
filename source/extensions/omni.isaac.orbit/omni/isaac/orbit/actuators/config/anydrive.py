# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration instances of actuation models for ANYmal robot.
"""

from omni.isaac.assets import ASSETS_RESOURCES_DIR

from omni.isaac.orbit.actuators.group import ActuatorControlCfg, ActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ActuatorNetLSTMCfg, ActuatorNetMLPCfg, DCMotorCfg
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

"""
Actuator Models.
"""

ANYDRIVE_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    peak_motor_torque=120.0, motor_torque_limit=80.0, motor_velocity_limit=7.5, gear_ratio=1.0
)
"""Configuration for ANYdrive 3.x with DC actuator model."""

ANYDRIVE_3_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    network_file=f"{ISAAC_ORBIT_NUCLEUS_DIR}/ActuatorNets/anydrive_3_lstm_jit.pt",
    peak_motor_torque=120.0,
    motor_torque_limit=89.0,
    motor_velocity_limit=7.5,
    gear_ratio=1.0,
)
"""Configuration for ANYdrive 3.0 (used on ANYmal-C) with LSTM actuator model."""

ANYDRIVE_3_1_ACTUATOR_CFG = ActuatorNetMLPCfg(
    network_file=f"{ASSETS_RESOURCES_DIR}/actuator_nets/anydrive_3_1_actuator.jit",
    peak_motor_torque=140.0,
    motor_torque_limit=89.0,
    motor_velocity_limit=8.5,
    gear_ratio=1.0,
    pos_scale=5.0,
    vel_scale=0.2,
    torque_scale=60.0,
    input_idx=(0, 2, 4),
)
"""Configuration for ANYdrive 3.1 (used on ANYmal-D) with MLP actuator model."""

"""
Actuator Groups.
"""

ANYMAL_C_DEFAULT_GROUP_CFG = ActuatorGroupCfg(
    dof_names=[".*HAA", ".*HFE", ".*KFE"],
    model_cfg=ANYDRIVE_3_ACTUATOR_CFG,
    control_cfg=ActuatorControlCfg(
        command_types=["p_abs"],
        dof_pos_offset={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,
            ".*H_KFE": 0.8,
        },
    ),
)
"""Configuration for default ANYmal-C quadruped with LSTM actuator network."""

ANYMAL_D_DEFAULT_GROUP_CFG = ActuatorGroupCfg(
    dof_names=[".*HAA", ".*HFE", ".*KFE"],
    model_cfg=ANYDRIVE_3_1_ACTUATOR_CFG,
    control_cfg=ActuatorControlCfg(
        command_types=["p_abs"],
        dof_pos_offset={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,
            ".*H_KFE": 0.8,
        },
    ),
)
"""Configuration for default ANYmal-D quadruped with MLP actuator network."""
