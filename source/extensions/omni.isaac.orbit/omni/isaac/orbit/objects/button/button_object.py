# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Optional, Sequence

import carb
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.articulations import ArticulationView 

import omni.isaac.orbit.utils.kit as kit_utils

from .button_cfg import ButtonObjectCfg
from .button_data import ButtonObjectData
import math


class ButtonObject:
    """Class for handling button objects.

    Button objects are spawned from USD files and are encapsulated by a single root prim.
    The root prim is used to apply physics material to the button body.

    This class wraps around :class:`ArticulationView` class from Isaac Sim to support the following:

    * Configuring using a single dataclass (struct).
    * Applying physics material to the button body.
    * Handling different button body views.
    * Storing data related to the button object.

    """

    cfg: ButtonObjectCfg
    """Configuration class for the button object."""
    objects: ArticulationView
    """Button prim view for the button object."""

    def __init__(self, cfg: ButtonObjectCfg):
        """Initialize the button object.

        Args:
            cfg (ButtonObjectCfg): An instance of the configuration class.
        """
        # store inputs
        self.cfg = cfg
        # container for data access
        self._data = ButtonObjectData()
        # buffer variables (filled during spawn and initialize)
        self._spawn_prim_path: str = None

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
    def data(self) -> ButtonObjectData:
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

        # -- save prim path for later
        self._spawn_prim_path = prim_path
        # -- spawn asset if it doesn't exist.
        if not prim_utils.is_prim_path_valid(prim_path):
            # add prim as reference to stage
            prim_utils.create_prim(
                self._spawn_prim_path,
                usd_path=self.cfg.usd_path,
                translation=translation,
                orientation=orientation,
                scale=self.cfg.scale,
            )
            if(self.cfg.symbol_usd_path is not None):
                prim_utils.create_prim(
                    self._spawn_prim_path + "/symbol/symbol",
                    usd_path=self.cfg.symbol_usd_path,
                    translation=(0,0,0),
                    scale=self.cfg.scale,
                )
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
        # create handles
        # -- object views
        self.articulations = ArticulationView(prim_paths_expr, reset_xform_properties=False)
        self.articulations.initialize()
        # set the default state
        self.articulations.post_reset()

        # create buffers
        # constants
        self._ALL_INDICES = torch.arange(self.count, dtype=torch.long, device=self.device)
        # -- frame states
        self.data.dof_pos = self.articulations.get_joint_positions(indices=self._ALL_INDICES, clone=False)
        self.data.dof_vel = self.articulations.get_joint_velocities(indices=self._ALL_INDICES, clone=False)
        self.data.btn_state = torch.zeros((self.count, 1), dtype=torch.bool, device=self.device)
        self.data.btn_state = torch.zeros((self.count, 1), dtype=torch.bool, device=self.device)
    def reset_buffers(self, env_ids: Optional[Sequence[int]] = None):
        """Resets all internal buffers.

        Args:
            env_ids (Optional[Sequence[int]], optional): The indices of the object to reset.
                Defaults to None (all instances).
        """
        pass

    def update_buffers(self, dt: float = None):
        """Update the internal buffers.

        The time step ``dt`` is used to compute numerical derivatives of quantities such as joint
        accelerations which are not provided by the simulator. Not used for this object.

        Args:
            dt (float, optional): The amount of time passed from last `update_buffers` call. Defaults to None.
        """
        # frame states
        self.data.dof_pos[:,:] = self.articulations.get_joint_positions(indices=self._ALL_INDICES, clone=False)
        self.data.dof_vel[:,:] = self.articulations.get_joint_velocities(indices=self._ALL_INDICES, clone=False)
        self.data.btn_state[:,:] = torch.where(self.data.btn_isdown, True, self.data.btn_state)

        # Flip the light if the button is pressed or have btn_state
        light_on = self.data.btn_state[:,[0]] | self.data.btn_isdown[:,[0]]
        self.articulations.set_joint_positions((~light_on) * math.pi, self._ALL_INDICES, torch.tensor([1]))

