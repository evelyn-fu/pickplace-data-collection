import numpy as np
import copy
from PIL import Image
from pydrake.systems.framework import LeafSystem
from pydrake.systems.sensors import (
    ImageRgba8U,
    ImageDepth16U,
    ImageLabel16I,
)
from pydrake.common.value import Value
from pydrake.common.value import (
    Value
)

class ImageSaver(LeafSystem):
    def __init__(self, dirstr = "test2"):
        super().__init__()

        self.dirstr = dirstr
        self.DeclareAbstractInputPort(name="rgb_in",
                                      model_value=Value(ImageRgba8U()))
        self.DeclareAbstractInputPort(name="depth_in",
                                      model_value=Value(ImageDepth16U()))
        self.DeclareAbstractInputPort(name="label_in",
                                      model_value=Value(ImageLabel16I()))

        # Calling `ForcePublish()` will trigger the callback.
        self.DeclareForcedPublishEvent(self.Publish)

        # Publish at 33 fps
        self.DeclarePeriodicPublishEvent(period_sec=0.03,
                                         offset_sec=0,
                                         publish=self.Publish)
        
    def Publish(self, context):
        time_ms = int(context.get_time() * 1000)
        timestr = f"{time_ms:06d}"
        
        color = self.GetInputPort("rgb_in").Eval(context).data

        depth = copy.deepcopy(
            self.GetInputPort("depth_in").Eval(context).data.squeeze()
        )

        label_image = copy.deepcopy(
            self.GetInputPort("label_in").Eval(context).data.squeeze()
        )

        # remove alpha
        color = color[:, :, :3]
        color_pil = Image.fromarray(color)
        color_pil.save(self.dirstr+"/rgb/"+timestr+".png")
        
        # get mask for bottle
        object_labels = np.unique(label_image)
        masks = [
            np.uint8(np.where(label_image == label, 255, 0)) for label in object_labels
        ]
        mask_pil = Image.fromarray(masks[0])
        mask_pil.save(self.dirstr+"/masks/"+timestr+".png")

        # cap depth at 3000mm
        depth[depth > 3000] = 3000
        depth_pil = Image.fromarray(depth)
        depth_pil.save(self.dirstr+"/depth/"+timestr+".png")