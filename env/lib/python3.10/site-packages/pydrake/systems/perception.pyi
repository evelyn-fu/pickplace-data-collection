from pydrake.common.value import Value as Value
from pydrake.math import RigidTransform as RigidTransform
from pydrake.perception import BaseField as BaseField, Fields as Fields, PointCloud as PointCloud
from pydrake.systems.framework import LeafSystem as LeafSystem

class PointCloudConcatenation(LeafSystem):
    def __init__(self, id_list, default_rgb=...) -> None: ...
    def DoCalcOutput(self, context, output) -> None: ...
