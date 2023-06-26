# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Optional, Tuple, List 

from omni.isaac.orbit.utils import configclass
from .button_cfg import ButtonObjectCfg

@configclass
class ButtonPanelCfg:
    """Configuration parameters for button Panel"""

    @configclass
    class RandomizerCfg:
        enabled: bool = False

    panel_size: Tuple[float, float] = (0.1, 0.1)
    """Size of the panel in meters. height, width Defaults to (0.1, 0.1)."""
    panel_grids: Tuple[int, int] = (0, 0)
    """Number of buttons in the panel, rows, columns. Defaults to (0, 0)."""
    btn_cfgs: List[ButtonObjectCfg] = []

    randomizer: RandomizerCfg = RandomizerCfg()
    