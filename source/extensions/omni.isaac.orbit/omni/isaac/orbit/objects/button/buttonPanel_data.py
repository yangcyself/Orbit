# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass


@dataclass
class ButtonPanelData:
    """Data container for a buttonPanel."""

    # Just a Nenv x Nbtn random perm matrix
    # Just to give some randomness to the choice of buttons
    buttonRanking = None