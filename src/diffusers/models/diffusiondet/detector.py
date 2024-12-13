from torch import nn
from transformers.utils.backbone_utils import load_backbone


class DiffusionDet(nn.Module):
    """
    Implement DiffusionDet
    """

    def __init__(self, config):
        super(DiffusionDet, self).__init__()

        self.preprocess_image = None
        self.backbone = load_backbone(config)

        self.head = None
        self.criterion = None

        self.in_features = 0

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
        """
        images = self.preprocess_image(batched_inputs)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        if self.training:
            targets, x_boxes, t = None, None, None
            outputs_class, outputs_coord = self.head(features, x_boxes, t, None)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            loss_dict = self.criterion(output, targets)

            return loss_dict
