from dataclasses import dataclass
import numpy as np
import wandb
import os
from typing import Optional
from pathlib import Path
from pydrake.all import (
    BsplineBasis,
    BsplineTrajectory
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
        if wandb.run is not None:
            # NOTE: This overwrites the previous log
            np.save(
                os.path.join(wandb.run.dir, "spline_order.npy"),
                np.array([self.spline_order]),
            )
            np.save(
                os.path.join(wandb.run.dir, "control_points.npy"), self.control_points
            )
            np.save(os.path.join(wandb.run.dir, "knots.npy"), self.knots)
        if logging_path is not None:
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