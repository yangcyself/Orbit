# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.actuators.model import DCMotorCfg

# 48V
BABOON_ACTUATOR_CFG = DCMotorCfg(
    gear_ratio=1.0,
    motor_torque_limit=27.0,
    peak_motor_torque=60.0,
    motor_velocity_limit=15.5,
)

# 48V
COYOTE_ACTUATOR_CFG = DCMotorCfg(
    gear_ratio=1.0,
    motor_torque_limit=14.0,
    peak_motor_torque=30.0,
    motor_velocity_limit=40.0,
)
