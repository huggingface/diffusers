import math

import torch
from torch import nn
from torchvision.ops import RoIAlign


class DynamicHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, x_boxes, t, targets):
        pass


class DiffusionDetHead(nn.Module):
    def __init__(self, roi_input_shape):
        super().__init__()

        in_features = ['p2', 'p3', 'p4', 'p5']
        pooler_resolution = 7
        pooler_scales = tuple(1.0 / roi_input_shape[k].stride for k in in_features)
        sampling_ratio = 2

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
        )

    def forward(self, features, bboxes, time_emb):
        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = self.pooler(features, proposal_boxes)

        pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes,
                                                                                             self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
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

        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features


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
        assert (len(scales) == self.max_level - self.min_level + 1)
        assert 0 <= self.min_level <= self.max_level
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

    def forward(self, x, box_lists):
        num_level_assignments = len(self.level_poolers)

        if not is_fx_tracing():
            torch._assert(
                isinstance(x, list) and isinstance(box_lists, list),
                "Arguments to pooler must be lists",
            )
        assert_fx_safe(
            len(x) == num_level_assignments,
            "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
                num_level_assignments, len(x)
            ),
        )
        assert_fx_safe(
            len(box_lists) == x[0].size(0),
            "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
                x[0].size(0), len(box_lists)
            ),
        )
        if len(box_lists) == 0:
            return _create_zeros(None, x[0].shape[1], *self.output_size, x[0])

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])

        for level, pooler in enumerate(self.level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            output.index_put_((inds,), pooler(x[level], pooler_fmt_boxes_level))

        return output

def _fmt_box_list(box_tensor, batch_index: int):
    repeated_index = torch.full(
        (len(box_tensor), 1),
        batch_index,
        dtype=box_tensor.dtype,
        device=box_tensor.device,
    )
    return torch.cat((repeated_index, box_tensor), dim=1)


def convert_boxes_to_pooler_format(box_lists):
    pooler_fmt_boxes = torch.cat(
        [_fmt_box_list(box_list, i) for i, box_list in enumerate(box_lists)],
        dim=0,
    )
    return pooler_fmt_boxes

def assign_boxes_to_levels(
    box_lists,
    min_level,
    max_level,
    canonical_box_size,
    canonical_level,
):
    box_sizes = torch.sqrt(torch.cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8))
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level