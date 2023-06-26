# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Optional, Sequence

import carb
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.orbit.utils.kit as kit_utils

from .buttonPanel_cfg import ButtonPanelCfg
from .buttonPanel_data import ButtonPanelData
from .button_object import ButtonObject

import math


class ButtonPanel:
    """Class for handling button panels.

    A button panel is a rectangular panel with multiple buttons.
    This class wraps around multiple buttons. 
    
    """

    cfg: ButtonPanelCfg
    """Configuration class for the button object."""
    button: ButtonObject
    """Button prim view for the button object."""

    def __init__(self, cfg: ButtonPanelCfg):
        """Initialize the button object.

        Args:
            cfg (ButtonPanelCfg): An instance of the configuration class.
        """
        # store inputs
        self.cfg = cfg
        # container for data access
        self._data = ButtonPanelData()
        # buffer variables (filled during spawn and initialize)
        self._spawn_prim_path: str = None
        if(len(self.cfg.btn_cfgs) > 0):
            self.button: ButtonObject = ButtonObject(self.cfg.btn_cfgs[0])

    """
    Properties
    """

    @property
    def count(self) -> int:
        """Number of prims encapsulated."""
        return self.articulations.count

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self.articulations._device

    @property
    def data(self) -> ButtonPanelData:
        """Data related to articulation."""
        return self._data

    """
    Operations.
    """

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        """Spawn a button object into the stage (loaded from its USD file).

        Note:
            If inputs `translation` or `orientation` are not :obj:`None`, then they override the initial root
            state specified through the configuration class at spawning.

        Args:
            prim_path (str): The prim path for spawning object at.
            translation (Sequence[float], optional): The local position of prim from its parent. Defaults to None.
            orientation (Sequence[float], optional): The local rotation (as quaternion `(w, x, y, z)`
                of the prim from its parent. Defaults to None.
        """
        # use default arguments
        if translation is None:
            translation = self.cfg.init_state.pos
        if orientation is None:
            orientation = self.cfg.init_state.rot

        # -- save prim path for later
        self._spawn_prim_path = prim_path
        # -- spawn asset if it doesn't exist.
        if not prim_utils.is_prim_path_valid(prim_path):
            # add prim as reference to stage
            prim_utils.create_prim(
                self._spawn_prim_path,
                "Xform",
                translation=translation,
                orientation=orientation,
                scale = (1.0,1.0,1.0)
            )
            prim_utils.create_prim(
                self._spawn_prim_path+"/Panel",
                "Cube",
                translation=(0, 0, 0),
                scale=(self.cfg.panel_size[0]/2, self.cfg.panel_size[1]/2, 0.005),

            )
            self.btn_count = 0
            button_spacing_x = self.cfg.panel_size[0]/self.cfg.panel_grids[0]
            button_spacing_y = self.cfg.panel_size[1]/self.cfg.panel_grids[1]
            button_spacing_xb = - self.cfg.panel_size[0]/2
            button_spacing_yb = - self.cfg.panel_size[1]/2
            for i in range(self.cfg.panel_grids[0]):
                for j in range(self.cfg.panel_grids[1]):
                    self.button.cfg.usd_path = self.cfg.btn_cfgs[self.btn_count].usd_path
                    self.button.cfg.symbol_usd_path = self.cfg.btn_cfgs[self.btn_count].symbol_usd_path
                    self.button.spawn(f"{prim_path}/button_{self.btn_count}", 
                        translation=(button_spacing_xb+(i+0.5)*button_spacing_x, 
                                     button_spacing_yb+(j+0.5)*button_spacing_y, 
                                     0.005)
                    )
                    self.btn_count += 1
        else:
            carb.log_warn(f"A prim already exists at prim path: '{prim_path}'. Skipping...")


    def initialize(self, prim_paths_expr: Optional[str] = None):
        """Initializes the PhysX handles and internal buffers.

        Note:
            PhysX handles are only enabled once the simulator starts playing. Hence, this function needs to be
            called whenever the simulator "plays" from a "stop" state.

        Args:
            prim_paths_expr (Optional[str], optional): The prim path expression for the prims. Defaults to None.

        Raises:
            RuntimeError: When input `prim_paths_expr` is :obj:`None`, the method defaults to using the last
                prim path set when calling the :meth:`spawn()` function. In case, the object was not spawned
                and no valid `prim_paths_expr` is provided, the function throws an error.
        """
        # default prim path if not cloned
        if prim_paths_expr is None:
            if self._is_spawned is not None:
                self._prim_paths_expr = self._spawn_prim_path
            else:
                raise RuntimeError(
                    "Initialize the object failed! Please provide a valid argument for `prim_paths_expr`."
                )
        else:
            self._prim_paths_expr = prim_paths_expr
        self.button.initialize(f"{self._prim_paths_expr}/button.*")

    def reset_buffers(self, env_ids: Optional[Sequence[int]] = None):
        """Resets all internal buffers.

        Args:
            env_ids (Optional[Sequence[int]], optional): The indices of the object to reset.
                Defaults to None (all instances).
        """
        self.button.reset_buffers(env_ids)

    def update_buffers(self, dt: float = None):
        """Update the internal buffers.

        The time step ``dt`` is used to compute numerical derivatives of quantities such as joint
        accelerations which are not provided by the simulator. Not used for this object.

        Args:
            dt (float, optional): The amount of time passed from last `update_buffers` call. Defaults to None.
        """
        self.button.update_buffers(dt)

