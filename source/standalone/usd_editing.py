"""
Edit the usd file
references: https://developer.nvidia.com/usd/tutorials
            https://openusd.org/dev/api/class_usd_stage.html
"""

from omni.isaac.kit import SimulationApp

# Set the path below to your desired nucleus server
# Make sure you installed a local nucleus server before this
simulation_app = SimulationApp({"headless": False, "open_usd": f"source/standalone/elevator1.usd"})

from omni.isaac.core import World
from pxr import Sdf, Usd, UsdGeom

world = World()
world.reset()


prim = world.stage.GetPrimAtPath("/elevator")
# print("stageMembers",dir(world.stage))
"""
stageMembers ['ClearDefaultPrim', 'ClearMetadata', 'ClearMetadataByDictKey', 'CreateClassPrim', 'CreateInMemory',
'CreateNew', 'DefinePrim', 'ExpandPopulationMask', 'Export', 'ExportToString', 'FindLoadable', 'Flatten',
'GetAttributeAtPath', 'GetColorConfigFallbacks', 'GetColorConfiguration', 'GetColorManagementSystem', 'GetDefaultPrim',
'GetEditTarget', 'GetEditTargetForLocalLayer', 'GetEndTimeCode', 'GetFramesPerSecond', 'GetGlobalVariantFallbacks',
'GetInterpolationType', 'GetLayerStack', 'GetLoadRules', 'GetLoadSet', 'GetMasters', 'GetMetadata', 'GetMetadataByDictKey',
'GetMutedLayers', 'GetObjectAtPath', 'GetPathResolverContext', 'GetPopulationMask', 'GetPrimAtPath', 'GetPropertyAtPath',
'GetPseudoRoot', 'GetRelationshipAtPath', 'GetRootLayer', 'GetSessionLayer', 'GetStartTimeCode', 'GetTimeCodesPerSecond',
'GetUsedLayers', 'HasAuthoredMetadata', 'HasAuthoredMetadataDictKey', 'HasAuthoredTimeCodeRange', 'HasDefaultPrim',
'HasLocalLayer', 'HasMetadata', 'HasMetadataDictKey', 'InitialLoadSet', 'IsLayerMuted', 'IsSupportedFile', 'Load',
'LoadAll', 'LoadAndUnload', 'LoadNone', 'MuteAndUnmuteLayers', 'MuteLayer', 'Open', 'OpenMasked', 'OverridePrim', 'Reload',
 'RemovePrim', 'ResolveIdentifierToEditTarget', 'Save', 'SaveSessionLayers', 'SetColorConfigFallbacks',
 'SetColorConfiguration', 'SetColorManagementSystem', 'SetDefaultPrim', 'SetEditTarget', 'SetEndTimeCode',
 'SetFramesPerSecond', 'SetGlobalVariantFallbacks', 'SetInterpolationType', 'SetLoadRules', 'SetMetadata',
 'SetMetadataByDictKey', 'SetPopulationMask', 'SetStartTimeCode', 'SetTimeCodesPerSecond', 'Traverse', 'TraverseAll',
 'Unload', 'UnmuteLayer', 'WriteFallbackPrimTypes', '_GetPcpCache',
 '__bool__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'expired']
"""
# print("primName",prim.GetName()) # prints "elevator"
# print("primPath",prim.GetPrimPath()) # prints "/elevator"
# print("primmembers",dir(prim))
"""
AddAppliedSchema', 'ApplyAPI', 'ClearActive', 'ClearAssetInfo', 'ClearAssetInfoByKey', 'ClearCustomData',
'ClearCustomDataByKey', 'ClearDocumentation', 'ClearHidden', 'ClearInstanceable', 'ClearMetadata',
'ClearMetadataByDictKey', 'ClearPayload', 'ClearTypeName', 'ComputeExpandedPrimIndex', 'CreateAttribute',
'CreateRelationship', 'FindAllAttributeConnectionPaths', 'FindAllRelationshipTargetPaths', 'GetAllAuthoredMetadata',
'GetAllChildren', 'GetAllMetadata', 'GetAppliedSchemas', 'GetAssetInfo', 'GetAssetInfoByKey', 'GetAttribute',
'GetAttributeAtPath', 'GetAttributes', 'GetAuthoredAttributes', 'GetAuthoredProperties', 'GetAuthoredPropertiesInNamespace',
'GetAuthoredPropertyNames', 'GetAuthoredRelationships', 'GetChild', 'GetChildren', 'GetCustomData', 'GetCustomDataByKey',
'GetDescription', 'GetDocumentation', 'GetFilteredChildren', 'GetFilteredNextSibling', 'GetInherits', 'GetInstances',
'GetMaster', 'GetMetadata', 'GetMetadataByDictKey', 'GetName', 'GetNamespaceDelimiter', 'GetNextSibling', 'GetObjectAtPath',
'GetParent', 'GetPath', 'GetPayloads', 'GetPrim', 'GetPrimAtPath', 'GetPrimDefinition', 'GetPrimInMaster', 'GetPrimIndex',
'GetPrimPath', 'GetPrimStack', 'GetPrimTypeInfo', 'GetProperties', 'GetPropertiesInNamespace', 'GetProperty',
'GetPropertyAtPath', 'GetPropertyNames', 'GetPropertyOrder', 'GetReferences', 'GetRelationship', 'GetRelationshipAtPath',
'GetRelationships', 'GetSpecializes', 'GetSpecifier', 'GetStage', 'GetTypeName', 'GetVariantSet', 'GetVariantSets',
'HasAPI', 'HasAssetInfo', 'HasAssetInfoKey', 'HasAttribute', 'HasAuthoredActive', 'HasAuthoredAssetInfo',
'HasAuthoredAssetInfoKey', 'HasAuthoredCustomData', 'HasAuthoredCustomDataKey', 'HasAuthoredDocumentation',
'HasAuthoredHidden', 'HasAuthoredInherits', 'HasAuthoredInstanceable', 'HasAuthoredMetadata', 'HasAuthoredMetadataDictKey',
'HasAuthoredPayloads', 'HasAuthoredReferences', 'HasAuthoredSpecializes', 'HasAuthoredTypeName', 'HasCustomData',
'HasCustomDataKey', 'HasDefiningSpecifier', 'HasMetadata', 'HasMetadataDictKey', 'HasPayload', 'HasProperty',
'HasRelationship', 'HasVariantSets', 'IsA', 'IsAbstract', 'IsActive', 'IsDefined', 'IsGroup', 'IsHidden', 'IsInMaster',
'IsInstance', 'IsInstanceProxy', 'IsInstanceable', 'IsLoaded', 'IsMaster', 'IsModel', 'IsPseudoRoot', 'IsValid', 'Load',
'RemoveAPI', 'RemoveAppliedSchema', 'RemoveProperty', 'SetActive', 'SetAssetInfo', 'SetAssetInfoByKey', 'SetCustomData',
'SetCustomDataByKey', 'SetDocumentation', 'SetHidden', 'SetInstanceable', 'SetMetadata', 'SetMetadataByDictKey',
'SetPayload', 'SetPropertyOrder', 'SetSpecifier', 'SetTypeName', 'Unload', '_GetSourcePrimIndex',
"""

# print("primAttributes",prim.GetAttributes())
"""
primAttributes [Usd.Prim(</elevator>).GetAttribute('physxArticulation:articulationEnabled'),
    Usd.Prim(</elevator>).GetAttribute('physxArticulation:enabledSelfCollisions'),
    Usd.Prim(</elevator>).GetAttribute('physxArticulation:sleepThreshold'),
    Usd.Prim(</elevator>).GetAttribute('physxArticulation:solverPositionIterationCount'),
    Usd.Prim(</elevator>).GetAttribute('physxArticulation:solverVelocityIterationCount'),
    Usd.Prim(</elevator>).GetAttribute('physxArticulation:stabilizationThreshold'),
    Usd.Prim(</elevator>).GetAttribute('purpose'), Usd.Prim(</elevator>).GetAttribute('visibility'),
    Usd.Prim(</elevator>).GetAttribute('xformOp:orient'), Usd.Prim(</elevator>).GetAttribute('xformOp:scale'),
    Usd.Prim(</elevator>).GetAttribute('xformOp:translate'), Usd.Prim(</elevator>).GetAttribute('xformOpOrder')]
"""

# print("primChildren",prim.GetChildren())
"""
primChildren [Usd.Prim(</elevator/ElevatorOutsideArmature_3>), Usd.Prim(</elevator/ElevatorCage_11>),
    Usd.Prim(</elevator/ElevatorCallingButtons_12>), Usd.Prim(</elevator/ElevatorCallingButtons_007_13>),
    Usd.Prim(</elevator/ElevatorCallingButtons_006_14>), Usd.Prim(</elevator/ElevatorCallingButtons_003_15>),
    Usd.Prim(</elevator/FixedJoint>), Usd.Prim(</elevator/elevatorFrame>), Usd.Prim(</elevator/LeftOutsideDoorFrame>),
    Usd.Prim(</elevator/RightOutsideDoorFrame>), Usd.Prim(</elevator/LeftOutsideDoor>), Usd.Prim(</elevator/RightOutsideDoor>),
    Usd.Prim(</elevator/LeftInteriorDoorFrame>), Usd.Prim(</elevator/RightInteriorDoorFrame>),
    Usd.Prim(</elevator/LeftInteriorDoor>), Usd.Prim(</elevator/RightInteriorDoor>), Usd.Prim(</elevator/Materials>),
    Usd.Prim(</elevator/ElevatorShell>), Usd.Prim(</elevator/CageCeil>)]
"""
# print("primAllMatadata",prim.GetAllMetadata())
"""
{'apiSchemas': <pxr.Sdf.TokenListOp object at 0x7f83181a9730>,
'documentation': 'Concrete prim schema for a transform, which implements Xformable ',
'specifier': Sdf.SpecifierDef, 'typeName': 'Xform'}
"""


def move_prim(stage, src_path, dst_path):
    def copy_prim(src_prim, dst_prim):
        # Copy the attributes and metadata from the source prim to the new prim
        for attr in src_prim.GetAttributes():
            dst_attr = dst_prim.CreateAttribute(attr.GetName(), attr.GetTypeName())
            dst_attr.Set(attr.Get())

        for key in src_prim.GetAllMetadata():
            dst_prim.SetMetadata(key, src_prim.GetMetadata(key))

        # Copy the relationships from the source prim to the new prim
        for rel in src_prim.GetRelationships():
            dst_rel = dst_prim.CreateRelationship(rel.GetName())
            dst_rel.SetTargets(rel.GetTargets())

        # Recursively copy children
        for child in src_prim.GetChildren():
            child_dst_path = f"{dst_prim.GetPath()}/{child.GetName()}"
            child_dst_prim = stage.DefinePrim(child_dst_path, child.GetTypeName())
            copy_prim(child, child_dst_prim)

    # Locate the prim you want to move
    src_prim = stage.GetPrimAtPath(src_path)

    # Check if the prim exists
    if not src_prim:
        print(f"Prim not found at path: {src_path}")
        return

    # Create a new prim at the desired destination path
    dst_prim = stage.DefinePrim(dst_path, src_prim.GetTypeName())

    # Copy the source prim and its children to the destination prim
    copy_prim(src_prim, dst_prim)

    # Remove the source prim from the scene
    stage.RemovePrim(src_path)
    return dst_prim


def xform_to_fixed_joint(stage, xform_prim, fixed_joint_prim):
    # Check if the input prim is an Xform
    if not UsdGeom.Xformable(xform_prim):
        print(f"Input prim is not an Xform: {xform_prim.GetPath()}")
        return

    # Get the Xformable object from the input prim
    xform = UsdGeom.Xformable(xform_prim)

    # Get the local transformation matrix
    local_transform = xform.GetLocalTransformation()

    # Assuming the FixedJoint schema has attributes for local positions and rotations
    # Set the local position and rotation for body0
    fixed_joint_prim.CreateAttribute("body0:local_position", Sdf.ValueTypeNames.Float3).Set(
        local_transform.ExtractTranslation()
    )
    fixed_joint_prim.CreateAttribute("body0:local_rotation", Sdf.ValueTypeNames.Quatf).Set(
        local_transform.ExtractRotation()
    )

    # Set the local position and rotation for body1
    # In this example, we assume body1's local position and rotation to be the identity transform
    fixed_joint_prim.CreateAttribute("body1:local_position", Sdf.ValueTypeNames.Float3).Set((0, 0, 0))
    fixed_joint_prim.CreateAttribute("body1:local_rotation", Sdf.ValueTypeNames.Quatf).Set((1, 0, 0, 0))


def get_relative_transform(xform_prim_a, xform_prim_b):
    # Check if both input prims are Xform
    if not (UsdGeom.Xformable(xform_prim_a) and UsdGeom.Xformable(xform_prim_b)):
        print("One or both input prims are not Xform prims.")
        return

    # Get the Xformable objects from the input prims
    xform_a = UsdGeom.Xformable(xform_prim_a)
    xform_b = UsdGeom.Xformable(xform_prim_b)

    # Get the world transformation matrices for both prims
    world_transform_a = xform_a.ComputeWorldTransform()
    world_transform_b = xform_b.ComputeWorldTransform()

    # Calculate the relative transformation matrix between the two prims
    relative_transform = world_transform_a.GetInverse() * world_transform_b

    return relative_transform


# Get the Xformable object from the input prim
xform = UsdGeom.Xformable(prim)
# Get the local transformation matrix
local_transform = xform.GetLocalTransformation()
print("Translation", local_transform.ExtractTranslation())  # (0, 0, 0)
print("Rotation", local_transform.ExtractRotation())  # [(1, 0, 0) 0]
print("tuple Translation", tuple(local_transform.ExtractTranslation()))  # (0, 0, 0)
quat = local_transform.ExtractRotation().GetQuat()
print("tuple Rotation", (quat.real, *quat.imaginary))  # (1.0, 0.0, 0.0, 0.0)


mirror_prim = stage.GetPrimAtPath("/elevator/ElevatorCage_11/Mirror_9")
anchor_prim = stage.GetPrimAtPath("/elevator/elevatorFrame")
anchorpos = get_relative_transform(anchor_prim, mirror_prim)
mirror_prim = move_prim(world.stage, "/elevator/ElevatorCage_11/Mirror_9", "/elevator/Mirror_9")
UsdGeom.Xformable(mirror_prim).SetLocalTransformation(anchorpos)


import numpy as np

while simulation_app.is_running():
    # position, orientation = fancy_cube.get_world_pose()
    # linear_velocity = fancy_cube.get_linear_velocity()
    # # will be shown on terminal
    # print("Cube position is : " + str(position))
    # print("Cube's orientation is : " + str(orientation))
    # print("Cube's linear velocity is : " + str(linear_velocity))
    # we have control over stepping physics and rendering in this workflow
    # things run in sync
    world.step(render=True)  # execute one physics step and one rendering step
