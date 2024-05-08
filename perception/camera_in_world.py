from pydrake.all import AbstractValue, Context, LeafSystem, RigidTransform

class CameraPoseInWorldSource(LeafSystem):

    def __init__(self, X_ee_camera):
        super().__init__()

        # The current_cmd that should be passed to output when the current trajectory is invalid
        self._X_EE_input_port = self.DeclareAbstractInputPort(
            "X_EE", AbstractValue.Make(RigidTransform())
        )

        self.DeclareAbstractOutputPort(
            "X_WC",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcCameraPose,
        )

        self.X_ee_camera = X_ee_camera

    def CalcCameraPose(self, context: Context, output) -> None:
        X_EE = self.GetInputPort("X_EE").Eval(context)
        X_WC = X_EE @ self.X_ee_camera

        output.set_value(X_WC)