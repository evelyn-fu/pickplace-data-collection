import numpy as np
import copy
import json
import tqdm
import multiprocessing as mp
from functools import partial
from PIL import Image
from pydrake.systems.framework import LeafSystem
from pydrake.systems.sensors import CameraInfo
from pydrake.systems.sensors import (
    ImageRgba8U,
    ImageDepth16U,
    ImageLabel16I,
    ImageDepth32F,
    ConvertDepth32FTo16U
)
from pydrake.common.value import Value
from pydrake.common.value import (
    AbstractValue
)
from pydrake.math import (
    RigidTransform,
)

class MaskExtractor(LeafSystem):
    def __init__(
            self,
            gripper_mask_ind,
        ):
        super().__init__()

        self.DeclareAbstractInputPort("label_in",
                                    AbstractValue.Make(ImageLabel16I()))
    
        self.DeclareAbstractOutputPort(
            "mask_out",
            lambda: AbstractValue.Make(np.zeros((0, 0), dtype=np.uint8)),
            self.ExtractMask,
        )

        self.gripper_mask_ind = gripper_mask_ind
    
    def ExtractMask(self, context, output):
        label_image = copy.deepcopy(
            self.GetInputPort("label_in").Eval(context).data.squeeze()
        )
        object_labels = np.unique(label_image)
        masks = [
            np.uint8(np.where(label_image == label, 255, 0)) for label in object_labels
        ]
        if len(masks) <= self.gripper_mask_ind:
            print("no gripper seen in frame")
            output.set_value(np.zeros(label_image.shape, dtype=np.uint8))
        else:
            output.set_value(masks[self.gripper_mask_ind])