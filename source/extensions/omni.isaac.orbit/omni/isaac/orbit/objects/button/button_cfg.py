# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Optional, Tuple

from omni.isaac.orbit.utils import configclass


@configclass
class ButtonObjectCfg:
    """Configuration parameters for a robot."""
    usd_path: str = MISSING
    """USD file to spawn asset from."""
    symbol_usd_path: str = None
    """USD file for the symbol on the button. None for no symbol."""
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scale to spawn the object with. Defaults to (1.0, 1.0, 1.0)."""
