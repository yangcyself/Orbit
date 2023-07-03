# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass


@dataclass
class ButtonObjectData:
    """Data container for a robot."""

    ##
    # Frame states.
    ##

    dof_pos: torch.Tensor = None
    """Dof pos shape is ``(btn_count, dof)``."""
    dof_vel: torch.Tensor = None
    """Dof pos shape is ``(btn_count, dof)``."""
    btn_state: torch.Tensor = None
    """How long has this button been pushed. Shape is ``(btn_count, 1)``."""

    """
    Properties
    """
    @property
    def btn_isdown(self) -> torch.Tensor:
        """Whether the button is pushed or not. Shape is ``(btn_count, 1)``."""
        return self.dof_pos[:, 0:1]<0.0
