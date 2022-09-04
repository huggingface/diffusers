# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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

import torch
from typing import Tuple

class MPSWarmupMixin:
    r"""
    Temporary class to perform a 1-time warmup operation on some models, when they are moved to the `mps` device.

    It has been observed that the output of some models (`unet`, `vae`) is different the first time they run than the
    rest, for the same inputs. We are investigating the root cause of the problem, but meanwhile this class will be
    used to warmup those modules so their outputs are consistent.
    TODO: link to issue when we open it.

    Classes that require warmup need to adhere to [`MPSWarmupMixin`] and implement the following:

        - **warmup_inputs** -- A method that returns a suitable set of inputs to use during a forward pass.

    IMPORTANT:

    Warmup will be automatically performed when moving a pipeline to the `mps` device. If you move a single module, no
    warmup will be applied.
    """

    def warmup_inputs(self, batch_size=None) -> Tuple:
        r"""
        Return inputs suitable for the forward pass of this module.
        These will usually be a tuple of tensors. They will be automatically moved to the `mps` device on warmup.
        """
        raise NotImplementedError(
            """
            You declared conformance to `MPSWarmupMixin` but did not provide an implementation for `warmup_inputs`.

            Please, write a suitable implementation for `warmup_inputs` or remove conformance to `MPSWarmupMixin`
            if it's not needed.
            """
        )

    def warmup(self, batch_size=None):
        r"""
        Applies the warmup using `warmup_inputs`.
        Assumes this class implements `__call__` and has a `device` property.
        """
        if self.device.type != "mps":
            return

        with torch.no_grad():
            w_inputs = self.warmup_inputs(batch_size)
            w_inputs = [w.to("mps") for w in w_inputs]
            self.__call__(*w_inputs)
        