import torch
from torch import nn
from transformers.utils.backbone_utils import load_backbone

from diffusers.models.diffusiondet.head import DiffusionDetHead


class DiffusionDet(nn.Module):
    """
    Implement DiffusionDet
    """

    def __init__(self, config):
        super(DiffusionDet, self).__init__()

        self.training = True

        self.preprocess_image = None
        self.backbone = None # load_backbone(config)

        roi_input_shape = {
            'p2': {'stride': 4},
            'p3': {'stride': 8},
            'p4': {'stride': 16},
            'p5': {'stride': 32},
            'p6': {'stride': 64}
        }
        self.head = DiffusionDetHead(config, roi_input_shape=roi_input_shape)
        self.criterion = None

        self.in_features = 0

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
        """
        if self.training:
            features = [torch.rand(1, 256, i, i) for i in [144, 72, 36, 18]]
            x_boxes = torch.rand(1, 300, 4)
            t = torch.rand(1)

            outputs_class, outputs_coord = self.head(features, x_boxes, t, None)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            # loss_dict = self.criterion(output, targets)

            # return loss_dict
