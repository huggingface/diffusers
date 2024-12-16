import math
from dataclasses import astuple

import torch
from torch import nn
from torchvision.ops import RoIAlign


def convert_boxes_to_pooler_format(bboxes):
    B, N = bboxes.shape[:2]
    sizes = torch.full((B,), N)
    aggregated_bboxes = bboxes.view(B * N, -1)
    indices = torch.repeat_interleave(
        torch.arange(len(sizes), dtype=aggregated_bboxes.dtype, device=aggregated_bboxes.device), sizes
    )
    return torch.cat([indices[:, None], aggregated_bboxes], dim=1)


def assign_boxes_to_levels(
        bboxes,
        min_level,
        max_level,
        canonical_box_size,
        canonical_level,
):
    aggregated_bboxes = bboxes.view(bboxes.shape[0] * bboxes.shape[1], -1)
    area = (aggregated_bboxes[:, 2] - aggregated_bboxes[:, 0]) * (aggregated_bboxes[:, 3] - aggregated_bboxes[:, 1])
    box_sizes = torch.sqrt(area)
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8))
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


class DynamicHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, x_boxes, t, targets):
        pass


class DynamicConv(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.dim_dynamic = config.dim_dynamic
        self.num_dynamic = config.num_dynamic
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = config.pooler_resolution
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)


    def forward(self, pro_features, roi_features):
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


class DiffusionDetHead(nn.Module):
    def __init__(self, config, roi_input_shape, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1):
        super().__init__()

        in_features = config.roi_head_in_features
        pooler_resolution = config.pooler_resolution
        pooler_scales = tuple(1.0 / roi_input_shape[k]['stride'] for k in in_features)
        sampling_ratio = config.sampling_ratio

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
        )

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, features, bboxes, time_emb):
        B, N = bboxes.shape[:2]

        # roi_feature.
        roi_features = self.pooler(features, bboxes)

        pro_features = roi_features.view(B, N, self.d_model, -1).mean(-1)

        roi_features = roi_features.view(B * N, self.d_model, -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(B, N, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(N, B, self.d_model).permute(1, 0, 2).reshape(1, B * N,
                                                                                      self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(B * N, -1)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, N, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return class_logits.view(B, N, -1), pred_bboxes.view(B, N, -1), obj_features


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
            canonical_box_size=224,
            canonical_level=4,
    ):
        super().__init__()

        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2 and isinstance(output_size[0], int) and isinstance(output_size[1], int)
        assert math.isclose(min_level, int(min_level)) and math.isclose(max_level, int(max_level))
        assert (len(scales) == max_level - min_level + 1)
        assert 0 <= min_level <= max_level
        assert canonical_box_size > 0

        self.output_size = output_size
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        self.canonical_level = canonical_level
        self.canonical_box_size = canonical_box_size
        self.level_poolers = nn.ModuleList(
            RoIAlign(
                output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
            )
            for scale in scales
        )

    def forward(self, x, bboxes):
        num_level_assignments = len(self.level_poolers)
        assert len(x) == num_level_assignments and len(bboxes) == x[0].size(0)

        pooler_fmt_boxes = convert_boxes_to_pooler_format(bboxes)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            bboxes, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        batches = pooler_fmt_boxes.shape[0]
        channels = x[0].shape[1]
        output_size = self.output_size[0]
        sizes = (batches, channels, output_size, output_size)

        output = torch.zeros(sizes, dtype=x[0].dtype, device=x[0].device)

        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = (level_assignments == level).nonzero(as_tuple=True)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            output.index_put_((inds,), pooler(x_level, pooler_fmt_boxes_level))

        return output
