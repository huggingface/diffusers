# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import math

import torch
from torch import nn
from torch.nn import functional as F

from ...utils import is_comfy_kitchen_available


if is_comfy_kitchen_available():
    from comfy_kitchen.float_utils import from_blocked, to_blocked
    from comfy_kitchen.tensor import QuantizedTensor

    class ComfyQuantizedTensor(QuantizedTensor):
        """Packed Comfy weights whose sidecar tensors follow checkpoint row conversions."""

        def __new__(cls, qdata, layout_cls, params, metadata=None, sidecars=None, logical_shape=None):
            return torch.Tensor._make_wrapper_subclass(
                cls,
                logical_shape or params.orig_shape,
                device=qdata.device,
                dtype=params.orig_dtype,
                requires_grad=False,
            )

        def __init__(self, qdata, layout_cls, params, metadata=None, sidecars=None, logical_shape=None):
            super().__init__(qdata, layout_cls, params)
            self.metadata = metadata or {}
            self.sidecars = sidecars or {}

        def _copy_with(self, qdata=None, params=None, clone_params=True):
            if params is None:
                params = self.params.clone() if clone_params else self.params
            data = self._qdata if qdata is None else qdata
            return type(self)(
                data,
                self._layout_cls,
                params,
                self.metadata.copy(),
                {name: value.to(device=data.device).clone() for name, value in self.sidecars.items()},
                tuple(self.shape),
            )

        def dequantize(self):
            return super().dequantize().reshape(self.shape)

        def _row_components(self):
            rows, columns = self.params.orig_shape
            data = self._qdata
            fields = {}
            layout = self._layout_cls
            if layout == "TensorCoreNVFP4Layout":
                fields["block_scale"] = (from_blocked(self.params.block_scale, rows, columns // 16), 0)
            elif layout == "TensorCoreMXFP8Layout":
                fields["scale"] = (from_blocked(self.params.scale, rows, columns // 32), 0)
            elif layout in ("TensorWiseINT8Layout", "TensorCoreConvRotW4A4Layout"):
                fields["scale"] = (self.params.scale.reshape(-1, 1).expand(rows, 1), 0)
            elif layout in ("TensorCoreAWQW4A16Layout", "TensorCoreSVDQuantW4A4Layout"):
                scale = self.params.scale
                if layout == "TensorCoreSVDQuantW4A4Layout" and data.ndim == 4:
                    data = (
                        data.reshape(data.shape[0], data.shape[1], 32, 4, 32)
                        .permute(0, 2, 3, 1, 4)
                        .reshape(rows, columns // 2)
                    )
                    scale = scale.permute(1, 0, 2).reshape(columns // 64, rows)
                    fields["proj_up"] = (self.params.proj_up.permute(0, 2, 1).reshape(rows, -1), 0)
                elif layout == "TensorCoreSVDQuantW4A4Layout":
                    fields["proj_up"] = (self.params.proj_up, 0)
                fields["scale"] = (scale, 1)
                if layout == "TensorCoreAWQW4A16Layout":
                    fields["zeros"] = (self.params.zeros, 1)
            elif layout == "AsymW4A8Int8Layout":
                fields["scale"] = (self.params.scale, 0)
                fields["s_channel"] = (self.params.s_channel, 0)
                if self.params.correction is not None:
                    fields["correction"] = (self.params.correction, 1)
            return data[:rows], fields

        def _from_rows(self, data, fields, shape):
            values = {name: value for name, (value, _) in fields.items()}
            if self._layout_cls == "TensorCoreNVFP4Layout":
                values["block_scale"] = to_blocked(values["block_scale"])
            elif self._layout_cls == "TensorCoreMXFP8Layout":
                values["scale"] = to_blocked(values["scale"])
            elif self._layout_cls == "TensorCoreConvRotW4A4Layout":
                values["scale"] = values["scale"].flatten()
            params = dataclasses.replace(self.params, orig_shape=(math.prod(shape[:-1]), shape[-1]), **values)
            return type(self)(data, self._layout_cls, params, self.metadata.copy(), self.sidecars.copy(), tuple(shape))

        def _select_rows(self, indices, shape):
            data, fields = self._row_components()

            def select(tensor, dim):
                if tensor.element_size() == 1:
                    return tensor.view(torch.uint8).index_select(dim, indices).view(tensor.dtype)
                return tensor.index_select(dim, indices)

            return self._from_rows(
                select(data, 0), {name: (select(value, dim), dim) for name, (value, dim) in fields.items()}, shape
            )

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func in (torch.ops.aten.split.Tensor, torch.ops.aten.split_with_sizes.default):
                source, sizes = args[:2]
                dim = (args[2] if len(args) > 2 else kwargs.get("dim", 0)) % source.ndim
                if dim == source.ndim - 1:
                    raise NotImplementedError("Splitting quantized input features is not supported.")
                indices = torch.arange(math.prod(source.shape[:-1]), device=source.device).reshape(source.shape[:-1])
                return [
                    source._select_rows(part.flatten(), (*part.shape, source.shape[-1]))
                    for part in func(indices, sizes, dim)
                ]
            if func == torch.ops.aten.cat.default:
                tensors = args[0]
                dim = (args[1] if len(args) > 1 else kwargs.get("dim", 0)) % tensors[0].ndim
                if dim != 0 or any(not isinstance(t, cls) or t._layout_cls != tensors[0]._layout_cls for t in tensors):
                    raise NotImplementedError("Concatenation requires matching Comfy layouts along output rows.")
                source = tensors[0]
                components = [tensor._row_components() for tensor in tensors]
                row_fields = components[0][1]
                for tensor, (_, fields) in zip(tensors[1:], components[1:]):
                    if tensor.shape[1:] != source.shape[1:] or fields.keys() != row_fields.keys():
                        raise ValueError("Cannot concatenate incompatible quantized weight shapes or scales.")
                    if tensor.metadata != source.metadata or tensor.sidecars.keys() != source.sidecars.keys():
                        raise ValueError("Cannot concatenate weights with different quantization metadata.")
                    for name in source.sidecars:
                        if not torch.equal(source.sidecars[name], tensor.sidecars[name]):
                            raise ValueError(f"Cannot concatenate weights with different {name}.")
                    for field in dataclasses.fields(source.params):
                        if field.name in row_fields or field.name == "orig_shape":
                            continue
                        left, right = getattr(source.params, field.name), getattr(tensor.params, field.name)
                        equal = (
                            torch.equal(left, right)
                            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                            else left == right
                        )
                        if not equal:
                            raise ValueError(f"Cannot concatenate weights with different {field.name}.")
                data = torch.cat([part[0] for part in components], dim=0)
                fields = {
                    name: (torch.cat([part[1][name][0] for part in components], dim=axis), axis)
                    for name, (_, axis) in row_fields.items()
                }
                return source._from_rows(data, fields, (sum(t.shape[0] for t in tensors), *source.shape[1:]))
            if func in (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default):
                source, shape = args[:2]
                shape = list(shape)
                if -1 in shape:
                    shape[shape.index(-1)] = source.numel() // -math.prod(shape)
                if shape[-1] != source.shape[-1] or math.prod(shape) != source.numel():
                    raise NotImplementedError("Reshaping quantized input features is not supported.")
                return cls(
                    source._qdata, source._layout_cls, source.params, source.metadata, source.sidecars, tuple(shape)
                )
            if func in (torch.ops.aten.transpose.int, torch.ops.aten.unbind.int):
                source = args[0]
                indices = torch.arange(math.prod(source.shape[:-1]), device=source.device).reshape(source.shape[:-1])
                dims = [dim % source.ndim for dim in args[1:]]
                if any(dim == source.ndim - 1 for dim in dims):
                    raise NotImplementedError("Reordering quantized input features is not supported.")
                transformed = func(indices, *args[1:], **kwargs)
                if isinstance(transformed, torch.Tensor):
                    return source._select_rows(transformed.flatten(), (*transformed.shape, source.shape[-1]))
                return [source._select_rows(part.flatten(), (*part.shape, source.shape[-1])) for part in transformed]
            return super().__torch_dispatch__(func, types, args, kwargs)


class ComfyLinear(nn.Linear):
    """Linear inference dispatched through the weight's Comfy-Kitchen layout."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        inputs = inputs.to(weight.dtype)
        shape = inputs.shape
        inputs = inputs.reshape(-1, shape[-1])
        pre_quant_scale = weight.sidecars.get("pre_quant_scale")
        if pre_quant_scale is not None:
            inputs = inputs * pre_quant_scale.to(inputs.dtype)
        if weight.metadata.get("full_precision_matrix_mult", False):
            output = F.linear(inputs, weight.dequantize().to(weight.dtype), self.bias)
        else:
            if weight._layout_cls in ("TensorCoreFP8Layout", "TensorCoreNVFP4Layout", "TensorCoreMXFP8Layout"):
                quant_kwargs = {"scale": weight.sidecars.get("input_scale")}
                if weight._layout_cls == "TensorCoreFP8Layout":
                    quant_kwargs["dtype"] = weight.storage_dtype
                inputs = QuantizedTensor.from_float(inputs, weight._layout_cls, **quant_kwargs)
            output = F.linear(inputs, weight, self.bias)
        return output.to(weight.dtype).reshape(*shape[:-1], self.out_features)
