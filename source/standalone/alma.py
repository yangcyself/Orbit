import argparse
from omni.isaac.kit import SimulationApp

config = {"headless": False}
simulation_app = SimulationApp(config)

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import ArticulationView
from pxr import Gf

sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)

spawn_prim_path = "/Alma"

prim_utils.create_prim(
    spawn_prim_path,
    usd_path="/home/chenyu/opt/orbit/source/extensions/omni.isaac.assets/alma_no_merge/alma_no_merge.usd",
    translation=(0,0,0),
    orientation=(1,0,0,0),
)
sim.reset()

articulations = ArticulationView("/Alma", reset_xform_properties=False)

# # Play the simulator

articulations.initialize()
articulations.post_reset()
_dof_default_targets = articulations._physics_view.get_dof_position_targets()
_dof_pos = articulations.get_joint_positions(indices=[0,], clone=False)
print("DoF Name", articulations.dof_names)
print("Dof default targets", _dof_default_targets)
print("Dof pos", _dof_pos)


while simulation_app.is_running():
    # If simulation is stopped, then exit.
    if sim.is_stopped():
        pass
    else:
        sim.step()
    

simulation_app.close()
