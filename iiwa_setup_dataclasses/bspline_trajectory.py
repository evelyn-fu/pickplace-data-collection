from dataclasses import dataclass
import numpy as np
from typing import Optional, List
from pathlib import Path
import os
from pydrake.all import (
    BsplineBasis,
    BsplineTrajectory,
    CompositeTrajectory
)

@dataclass
class BsplineTrajectoryAttributes:
    """A data class to hold the attributes of a B-spline trajectory."""

    spline_order: int
    """The order of the B-spline basis to use."""
    control_points: np.ndarray
    """The control points of the B-spline trajectory of shape
    (num_joints, num_control_points)."""
    knots: np.ndarray
    """The knots of the B-spline basis of shape (num_knots,)."""

    @classmethod
    def from_bspline_trajectory(cls, traj: BsplineTrajectory) -> None:
        """Sets the attributes from a B-spline trajectory."""
        assert traj.start_time() == 0.0, "Trajectory must start at time 0!"
        return cls(
            spline_order=traj.basis().order(),
            control_points=traj.control_points(),
            knots=np.array(traj.basis().knots()) * traj.end_time(),
        )

    def log(self, logging_path: Optional[Path] = None) -> None:
        """Logs the B-spline trajectory attributes to wandb. If logging_path is not
        None, then the attributes are also saved to disk."""
        np.save(logging_path / "spline_order.npy", np.array([self.spline_order]))
        np.save(logging_path / "control_points.npy", self.control_points)
        np.save(logging_path / "knots.npy", self.knots)

    @classmethod
    def load(cls, path: Path) -> "BsplineTrajectoryAttributes":
        """Loads the B-spline trajectory attributes from disk."""
        return cls(
            spline_order=int(np.load(path / "spline_order.npy")[0]),
            control_points=np.load(path / "control_points.npy"),
            knots=np.load(path / "knots.npy"),
        )

    def to_bspline_trajectory(self) -> BsplineTrajectory:
        """Converts the attributes to a B-spline trajectory."""
        return BsplineTrajectory(
            basis=BsplineBasis(order=self.spline_order, knots=self.knots),
            control_points=self.control_points,
        )
    
@dataclass
class CompositeBsplineTrajectoryAttributes:
    """A data class to hold the attributes of a Composite trajectory of B-spline trajectories."""

    spline_order: List[int]
    """List of the order of the B-spline basis to use for each B-spline trajectory"""
    control_points: List[np.ndarray]
    """List of the control points of the B-spline trajectory of shape
    (num_joints, num_control_points) for each B-spline trajectory."""
    knots: List[np.ndarray]
    """List of the knots of the B-spline basis of shape (num_knots,) for each B-spline trajectory."""

    @classmethod
    def from_composite_bspline_trajectory(cls, traj: CompositeTrajectory) -> None:
        """Sets the attributes from a B-spline trajectory."""
        spline_order = []
        control_points = []
        knots = []
        for i in range(traj.get_number_of_segments()):
            segment = traj.segment(i)
            assert segment.start_time() == 0.0, f"segment {i} must start at time 0!"
            spline_order.append(segment.basis().order())
            control_points.append(segment.control_points())
            knots.append(np.array(segment.basis().knots()) * segment.end_time())
        
        return cls(
            spline_order=spline_order,
            control_points=control_points,
            knots=knots
        )

    def log(self, logging_path: Optional[Path] = None) -> None:
        """Saves the B-spline trajectory attributes to disk"""
        for i in range(len(self.spline_order)):
            np.save(logging_path / f"spline_orders/{i}.npy", np.array([self.spline_order[i]]))
            np.save(logging_path / f"control_points/{i}.npy", self.control_points[i])
            np.save(logging_path / f"knots/{i}.npy", self.knots[i])

    @classmethod
    def load(cls, path: Path) -> "BsplineTrajectoryAttributes":
        """Loads the B-spline trajectory attributes from disk."""
        spline_order = []
        control_points = []
        knots = []
        subpath = os.path.join(path, "spline_orders")
        num_segments = len([entry for entry in os.listdir(subpath) if os.path.isfile(os.path.join(subpath, entry))])
        for i in range(num_segments):
            spline_order.append(int(np.load(path / f"spline_orders/{i}.npy")[0]))
            control_points.append(np.load(path / f"control_points/{i}.npy"))
            knots.append(np.load(path / f"knots/{i}.npy"))

        return cls(
            spline_order=spline_order,
            control_points=control_points,
            knots=knots
        )

    def to_composite_bspline_trajectory(self) -> CompositeTrajectory:
        """Converts the attributes to a Composite trajectory of Bsplne trajectories."""
        segments = []
        for i in range(len(self.spline_order)):
            segments.append(BsplineTrajectory(
                basis=BsplineBasis(order=self.spline_order[i], knots=self.knots[i]),
                control_points=self.control_points[i],
            ))
        
        return CompositeTrajectory(segments)