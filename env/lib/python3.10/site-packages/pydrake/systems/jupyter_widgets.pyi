from _typeshed import Incomplete
from pydrake.common.jupyter import process_ipywidget_events as process_ipywidget_events
from pydrake.common.value import AbstractValue as AbstractValue
from pydrake.math import RigidTransform as RigidTransform, RollPitchYaw as RollPitchYaw
from pydrake.systems.framework import LeafSystem as LeafSystem, PublishEvent as PublishEvent
from typing import NamedTuple

class PoseSliders(LeafSystem):
    class Visible(NamedTuple):
        roll: Incomplete
        pitch: Incomplete
        yaw: Incomplete
        x: Incomplete
        y: Incomplete
        z: Incomplete

    class MinRange(NamedTuple):
        roll: Incomplete
        pitch: Incomplete
        yaw: Incomplete
        x: Incomplete
        y: Incomplete
        z: Incomplete

    class MaxRange(NamedTuple):
        roll: Incomplete
        pitch: Incomplete
        yaw: Incomplete
        x: Incomplete
        y: Incomplete
        z: Incomplete

    class Value(NamedTuple):
        roll: Incomplete
        pitch: Incomplete
        yaw: Incomplete
        x: Incomplete
        y: Incomplete
        z: Incomplete
    def __init__(self, visible=..., min_range=..., max_range=..., value=...) -> None: ...
    def SetPose(self, pose) -> None: ...
    def SetRpy(self, rpy) -> None: ...
    def SetXyz(self, xyz) -> None: ...
    def DoCalcOutput(self, context, output) -> None: ...

class WidgetSystem(LeafSystem):
    def __init__(self, *args, update_period_sec: float = ...) -> None: ...
    def DoCalcOutput(self, context, output, port_index) -> None: ...
