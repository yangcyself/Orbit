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
    
    Provide helper functions for the buttons
    """

    def __init__(self, cfg: ButtonPanelCfg):
        """Initialize the button object.

        Args:
            cfg (ButtonPanelCfg): An instance of the configuration class.
        """
        # store inputs
        self.cfg = cfg
        # container for data access, Empty for now
        self._data = ButtonPanelData()
        # buffer variables (filled during spawn and initialize)
        self._spawn_prim_path: str = None
        if(len(self.cfg.btn_cfgs) > 0):
            self.button: ButtonObject = ButtonObject(self.cfg.btn_cfgs[0])

        """ The count of `spawn` calls. I assume each spawn call relates to a environment. """
        self.env_count: int = 0
        """ The count of buttons created. """
        self.btn_count: int = 0


    """
    Properties
    """
    @property
    def count(self) -> int:
        """Number of prims encapsulated."""
        return self.env_count

    @property
    def btn_per_env(self) -> int:
        """Number of buttons per environment."""
        return self.cfg.panel_grids[0] * self.cfg.panel_grids[1]

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self.button.device

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
        self.env_count += 1

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
        # sanity check
        assert self.btn_count == self.button.count, "Button count mismatch!"
        assert self.btn_per_env ==  int(self.btn_count / self.env_count), "Button per env count mismatch!"

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

    def get_button_pose_w(self):
        """Get button poses in world frame.

        Returns:
            torch.Tensor: n_envs x n_btns x 7
        """
        position_w, quat_w = self.button.articulations.get_world_poses(indices=None, clone=False)
        return torch.cat([position_w, quat_w], dim=-1).view(self.env_count, self.btn_per_env, 7)


    # functions for getting and setting button state
    def get_state_env_any(self, btn_ids: Optional[Sequence[int]] = None):
        """Get button state and reduce them within env (with any).
        btn_ids: The indices of the button to get state from. Defaults to None (all buttons).
                  The indices are counted from 0 for each environment.
        Returns:
            torch.tensor: The button state.
        """
        
        states = self.button.data.btn_state.view(self.env_count, self.btn_per_env)
        if btn_ids is None:
            btn_ids = ...
        return states[:, btn_ids].any(dim=1)
    
    def set_state_env_all(self, s:int = 0, btn_ids: Optional[Sequence[int]] = None, env_ids:Optional[Sequence[int]] = None):
        """Set button state and reduce them within env (with all).
        s: The state to set to. Defaults to 0 (off).
        btn_ids: The indices of the button to set state to. Defaults to None (all buttons).
                  The indices are counted from 0 for each environment.
        env_ids: The indices of the environment to set state to. Defaults to None (all environments).
        """
        if btn_ids is None:
            btn_ids = ...
        if env_ids is None:
            env_ids = ...
        self.button.data.btn_state.view(self.env_count, self.btn_per_env)[env_ids, btn_ids] = s
    
    # functions for getting and setting environment state (everything)
    @property
    def state_should_dims(self):
        btn_state_dim = self.button.state_should_dims[-1]
        state_should_dims = [btn_state_dim*i for i in range(self.btn_per_env+1)]
        return state_should_dims

    def get_state(self):
        # Return the underlying state of a simulated environment. Should be compatible with reset_to.
        return self.button.get_state().view(self.env_count, -1)
    
    def reset_to(self, state):
        # Reset the simulated environment to a given state. Useful for reproducing results
        # state: N x D tensor, where N is the number of environments and D is the dimension of the state
        self.button.reset_to(state.view(self.btn_count, -1))