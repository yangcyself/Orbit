# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Optional, Sequence

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.semantics as semantics_utils

import omni.isaac.orbit.utils.kit as kit_utils
import omni.replicator.core as rep

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
        # container for data access
        self._data = ButtonPanelData()
        # buffer variables (filled during spawn and initialize)
        self._spawn_prim_path: Sequence[str] = []
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
        self._spawn_prim_path.append(prim_path)
        # -- spawn asset if it doesn't exist.
        if not prim_utils.is_prim_path_valid(prim_path):
            # add prim as reference to stage
            prim_utils.create_prim(
                prim_path,
                "Xform",
                translation=translation,
                orientation=orientation,
                scale = (1.0,1.0,1.0)
            )
            prim_utils.create_prim(
                prim_path+"/Panel",
                "Cube",
                translation=(0, 0, 0),
                scale=(self.cfg.panel_size[0]/2, self.cfg.panel_size[1]/2, 0.005),
                semantic_label="button_panel"
            )
            
            button_spacing_x = self.cfg.panel_size[0]/self.cfg.panel_grids[0]
            button_spacing_y = self.cfg.panel_size[1]/self.cfg.panel_grids[1]
            button_spacing_xb = - self.cfg.panel_size[0]/2
            button_spacing_yb = - self.cfg.panel_size[1]/2
            btn_count = 0
            for i in range(self.cfg.panel_grids[0]):
                for j in range(self.cfg.panel_grids[1]):
                    self.button.cfg.usd_path = self.cfg.btn_cfgs[btn_count].usd_path
                    self.button.cfg.symbol_usd_path = self.cfg.btn_cfgs[btn_count].symbol_usd_path
                    self.button.spawn(f"{prim_path}/Button_{btn_count}", 
                        translation=(button_spacing_xb+(i+0.5)*button_spacing_x, 
                                     button_spacing_yb+(j+0.5)*button_spacing_y, 
                                     0.005)
                    )
                    btn_count += 1
            self.btn_count += btn_count
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
                self._prim_paths_expr = self._spawn_prim_path[-1]
            else:
                raise RuntimeError(
                    "Initialize the object failed! Please provide a valid argument for `prim_paths_expr`."
                )
        else:
            self._prim_paths_expr = prim_paths_expr
        self.button.initialize(f"{self._prim_paths_expr}/Button.*")
        # sanity check
        assert self.btn_count == self.button.count, "Button count mismatch!"
        assert self.btn_per_env ==  int(self.btn_count / self.env_count), "Button per env count mismatch!"
        assert all(self._spawn_prim_path[i] <= self._spawn_prim_path[i + 1] for i in range(len(self._spawn_prim_path) - 1))

        # data
        self._data.buttonRanking = torch.stack([
            torch.randperm(self.btn_per_env)
            for i in range(self.env_count)
        ])

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

    def get_all_button_pose_w(self):
        """Get all button poses in world frame.

        Returns:
            torch.Tensor: n_envs x n_btns x 7
        """
        position_w, quat_w = self.button.articulations.get_world_poses(indices=None, clone=False)
        return torch.cat([position_w, quat_w], dim=-1).view(self.env_count, self.btn_per_env, 7)

    def get_rank1_button_pose_w(self):
        """Get the button pose in world frame.
          The target button ranked the first in buttonRanking

        Returns:
            torch.Tensor: n_envs x 7
        """
        indices = self._data.buttonRanking[:,0] + torch.arange(0, self.btn_count, self.btn_per_env)
        position_w, quat_w = self.button.articulations.get_world_poses(indices=indices, clone=False)
        return torch.cat([position_w, quat_w], dim=-1).view(self.env_count, 7)

    # Functions for randomization in reset
    def random_reset_buttonRanking(self, env_ids:Optional[Sequence[int]] = None):
        if env_ids is None:
            env_ids = torch.arange(self.env_count, dtype=torch.long, device=self.device)
        for i in env_ids:
            self._data.buttonRanking[i,:] = torch.randperm(self.btn_per_env)

    def reset_semantics(self, numTarget):
        """reset the semantics of the buttons according to self.data.targetButton
            num_target buttons are labeled as `class: button_target`
            others are labeled as `class: button`
        """
        self._data.nTargets = numTarget
        for i, btnrank in enumerate(self._data.buttonRanking):
            for j in btnrank[:numTarget]:
                prim = prim_utils.get_prim_at_path(self._spawn_prim_path[i]+f"/Button_{j}")
                semantics_utils.add_update_semantics(prim, "button_target")
            for j in btnrank[numTarget:]:
                prim = prim_utils.get_prim_at_path(self._spawn_prim_path[i]+f"/Button_{j}")
                semantics_utils.add_update_semantics(prim, "button")


    # functions for getting and setting button state
    def get_state_env_any(self, c:int=1, btn_ids: Optional[Sequence[int]] = None):
        """Get button state and reduce them within env (with any).
        c:       The condition for state to be true: the button has to be pushed longer than this value
        btn_ids: The indices of the button to get state from. Defaults to None (all buttons).
                  The indices are counted from 0 for each environment.
                  If it is a 2D Matrix, it means each environment has different set of targets
        Returns:
            torch.tensor: The button state.
        """
        states = self.button.data.btn_state.view(self.env_count, self.btn_per_env)
        if(type(btn_ids) == torch.Tensor and len(btn_ids.shape) == 2):
            return (states.gather(1, btn_ids)>=c).any(dim=1)
        else:
            if btn_ids is None:
                btn_ids = ...
            return (states[:, btn_ids]>=c).any(dim=1)
    
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
        state_should_dims = [0]
        state_should_dims.append(state_should_dims[-1] + btn_state_dim * self.btn_per_env)
        state_should_dims.append(state_should_dims[-1] + 1) # self.data.nTargets
        state_should_dims.append(state_should_dims[-1] + self._data.buttonRanking.shape[1]) # self.data.buttonRanking
        return state_should_dims

    def get_state(self):
        # Return the underlying state of a simulated environment. Should be compatible with reset_to.
        button_states = self.button.get_state().view(self.env_count, -1)
        nTargets = torch.tensor([self._data.nTargets]).tile(self.env_count, 1)
        return torch.cat([button_states, nTargets, self._data.buttonRanking], dim=-1)
    
    def reset_to(self, state):
        # Reset the simulated environment to a given state. Useful for reproducing results
        # state: N x D tensor, where N is the number of environments and D is the dimension of the state
        state_should_dims = self.state_should_dims
        self.button.reset_to(state[:, state_should_dims[0]: state_should_dims[1]].view(self.btn_count, -1))
        self._data.nTargets = int(state[:, state_should_dims[1]: state_should_dims[2]][0,0].item())
        self._data.buttonRanking = state[:, state_should_dims[2]: state_should_dims[3]].to(self._data.buttonRanking)
        self.reset_semantics(self._data.nTargets)
