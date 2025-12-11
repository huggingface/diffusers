# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import logging
from .modeling_utils import ModelMixin
import json


logger = logging.get_logger(__name__)

# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import logging
from .modeling_utils import ModelMixin
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding


logger = logging.get_logger(__name__)

class GatedMultiAdapter(ModelMixin):
    r"""
    MultiAdapter is a wrapper model that contains multiple adapter models and merges their outputs according to
    user-assigned weighting.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for common methods such as downloading
    or saving.

    Args:
        adapters (`List[T2IAdapter]`, *optional*, defaults to None):
            A list of `T2IAdapter` model instances.
    """

    def __init__(self, adapters: List["T2IAdapter"], hidden_dim: int = 128, enable_film: bool = True, time_embed_dim=256):
        super(GatedMultiAdapter, self).__init__()

        self.time_embed_dim = time_embed_dim
        self.time_proj = GaussianFourierProjection()   # maps scalar → sin/cos features
        fourier_dim = self.time_proj.embedding_size * 2      # usually 256
        
        self.time_emb = TimestepEmbedding(
            in_channels=fourier_dim,
            time_embed_dim=time_embed_dim,  # final embedding dim
        )

        self.time_proj.requires_grad_(True)
        self.time_emb.requires_grad_(True)

        self.num_adapter = len(adapters)
        self.adapters = nn.ModuleList(adapters)
        self.enable_film = enable_film

        if len(adapters) == 0:
            raise ValueError("Expecting at least one adapter")

        if len(adapters) == 1:
            raise ValueError("For a single adapter, please use the `T2IAdapter` class instead of `MultiAdapter`")

        # The outputs from each adapter are added together with a weight.
        # This means that the change in dimensions from downsampling must
        # be the same for all adapters. Inductively, it also means the
        # downscale_factor and total_downscale_factor must be the same for all
        # adapters.
        first_adapter_total_downscale_factor = adapters[0].total_downscale_factor
        first_adapter_downscale_factor = adapters[0].downscale_factor
        for idx in range(1, len(adapters)):
            if (
                adapters[idx].total_downscale_factor != first_adapter_total_downscale_factor
                or adapters[idx].downscale_factor != first_adapter_downscale_factor
            ):
                raise ValueError(
                    f"Expecting all adapters to have the same downscaling behavior, but got:\n"
                    f"adapters[0].total_downscale_factor={first_adapter_total_downscale_factor}\n"
                    f"adapters[0].downscale_factor={first_adapter_downscale_factor}\n"
                    f"adapter[`{idx}`].total_downscale_factor={adapters[idx].total_downscale_factor}\n"
                    f"adapter[`{idx}`].downscale_factor={adapters[idx].downscale_factor}"
                )

        self.total_downscale_factor = first_adapter_total_downscale_factor
        self.downscale_factor = first_adapter_downscale_factor

        # ---------- NEW: FiLM + gating MLP ----------
        # FiLM: maps [adapter_feature, timestep] → (gamma, beta)
        # gating network: [adapter_feature, timestep] → scalar gate
        self.hidden_dim = hidden_dim
        self.film_mlps = nn.ModuleList()  # len = num_scales (default = 4 len of channels) channels: List[int] = [320, 640, 1280, 1280], seen in T2I-adapter
        self.gate_mlps = nn.ModuleList()
        self._init_mlps_from_features()
    
    def _init_mlps_from_features(self):
        # first_adapter_feats = adapter_outputs[0]  # list of 4 scales
        # num_scales = len(first_adapter_feats)
        channels = [320, 640, 1280, 1280]
        for k in range(4):
            C_k = channels[k]
            in_dim = C_k + self.time_embed_dim # pooled feature + timestep

            # FiLM_k
            film_mlp_k = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 2 * C_k )  # [gamma, beta]
            )

            # safe init: to let gamma ≈ 1, beta ≈ 0
            film_mlp_k[-1].weight.data.zero_()
            # bias: [ones for gamma | zeros for beta]
            film_mlp_k[-1].bias.data = torch.cat([
                torch.ones(C_k), 
                torch.zeros(C_k)
            ])

            # gate_k
            gate_mlp_k = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1), # scalar
            )

            self.film_mlps.append(film_mlp_k)
            self.gate_mlps.append(gate_mlp_k)


    def forward(self, xs: torch.Tensor, timestep: torch.Tensor) -> List[torch.Tensor]:
        r"""
        Args:
            xs (`torch.Tensor`):
                A tensor of shape (batch, channel, height, width) representing input images for multiple adapter
                models, concatenated along dimension 1(channel dimension). 
                The `channel` dimension should be equal to `num_adapter` * number of channel per image.

            timestep: diffusion timestep (batch or scalar)
        """
        assert isinstance(xs, (list, tuple)), "Adapter input must be a list of Tensors"
        assert len(xs) == self.num_adapter, f"Expected {self.num_adapter} inputs, got {len(xs)}"

        
        B = xs[0].shape[0]
        # =====Collect adapter outputs first=====
        adapter_outputs = []  # each element is a list of 4 feature maps

        for i, adapter in enumerate(self.adapters):
            x_i = xs[i]        # shape (B, C, H, W)
            ref_param = next(adapter.parameters())
            x_i = x_i.to(device=ref_param.device, dtype=ref_param.dtype)
            feats_i = adapter(x_i)   # list of 4 tensors (B, Ck, Hk, Wk)
            adapter_outputs.append(feats_i)

        device = xs[0].device
        
        self.film_mlps.to(device)
        self.gate_mlps.to(device)
        self.adapters.to(device)
        dtype = self.adapters[0].dtype
        for mlp in self.film_mlps:
            mlp.to(device=device, dtype=dtype)

        for mlp in self.gate_mlps:
            mlp.to(device=device, dtype=dtype)

        # ======prepare timestep embedding=====
        if timestep.dim() == 0:
            t = timestep[None].expand(B)
        else:
            t = timestep

        t = t.to(device=xs[0].device, dtype=xs[0].dtype)

        # ===== UNet timestep embedding =====
        t_proj = self.time_proj(t)     # (B, fourier_dim)
        t_embed = self.time_emb(t_proj)  # (B, time_embed_dim)


        # =====FiLM + gating========
        fused_outputs = []
        num_channels = len(adapter_outputs[0]) # 4

        all_weights = []
        for k in range(num_channels):
            # 对每一个channel，独立做FiLM和gating；
            # i: 不同adapter
            # k: 不同大小
            gamma_list = []
            beta_list = []
            gate_logit_list = []

            for i in range(self.num_adapter):
                m_i_k = adapter_outputs[i][k] # (B, C_k, H_k, W_k)
                C_k = m_i_k.shape[1] # channel size

                pooled = m_i_k.mean(dim=(2,3)) # (B, Ck)
                cond = torch.cat([pooled, t_embed], dim=1)
                cond = cond.to(dtype=self.film_mlps[k][0].weight.dtype)
                
                # film
                film_out = self.film_mlps[k](cond) # shape = (B, 2*C_k)
                # split to let: gamma_k_i.shape = (B, C_k); beta_k_i.shape  = (B, C_k)
                gamma_k_i, beta_k_i = torch.split(film_out, C_k, dim=-1) 
                gamma_k_i = gamma_k_i.view(B, C_k, 1, 1)
                beta_k_i = beta_k_i.view(B, C_k, 1, 1)
                gamma_list.append(gamma_k_i)
                beta_list.append(beta_k_i)

                # gate
                logit_i = self.gate_mlps[k](cond)          # (B,1)
                gate_logit_list.append(logit_i)

            # softmax over modalities i: (B, num_adapter)
            gate_logits_k = torch.cat(gate_logit_list, dim=1)
            alpha_k = torch.softmax(gate_logits_k, dim=1)  # (B, num_adapter)

            # ----- DEBUG PRINT GATE WEIGHTS -----
            if (not self.training) or (torch.rand(1).item() < 0.1):
                t_int = int(timestep[0].item()) if timestep.dim() > 0 else int(timestep.item())
                weights = alpha_k[0].detach().cpu().numpy().tolist()
                print(f"[GATE] t={t_int}, scale={k}, weights={weights}, sum={sum(weights):.4f}")
            # ------------------------------------
            all_weights.append(alpha_k.detach().cpu())


            # >> fuse modalities at this channel
            fused_k = 0
            for i in range(self.num_adapter):
                m_i_k = adapter_outputs[i][k]              # (B, C_k, H_k, W_k)
                gamma_k_i = gamma_list[i]                  # (B, C_k, 1, 1)
                beta_k_i = beta_list[i]                    # (B, C_k, 1, 1)
                w_i = alpha_k[:, i].view(B, 1, 1, 1)       # (B,1,1,1)

                if self.enable_film:
                    modulated = gamma_k_i * m_i_k + beta_k_i   # (B, C_k, H_k, W_k)
                else:
                    modulated = m_i_k
                fused_k = fused_k + w_i * modulated

            fused_outputs.append(fused_k)
            self.last_gate_weights = all_weights

        return fused_outputs

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
    ):
        """
        Save a model and its configuration file to a specified directory, allowing it to be re-loaded with the
        `[`~models.adapter.MultiAdapter.from_pretrained`]` class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                The directory where the model will be saved. If the directory does not exist, it will be created.
            is_main_process (`bool`, optional, defaults=True):
                Indicates whether current process is the main process or not. Useful for distributed training (e.g.,
                TPUs) and need to call this function on all processes. In this case, set `is_main_process=True` only
                for the main process to avoid race conditions.
            save_function (`Callable`):
                Function used to save the state dictionary. Useful for distributed training (e.g., TPUs) to replace
                `torch.save` with another method. Can also be configured using`DIFFUSERS_SAVE_MODE` environment
                variable.
            safe_serialization (`bool`, optional, defaults=True):
                If `True`, save the model using `safetensors`. If `False`, save the model with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
        """
        os.makedirs(save_directory, exist_ok=True)
        idx = 0
        model_path_to_save = save_directory
        for adapter in self.adapters:
            idx += 1
            model_path_to_save = save_directory + f"_{idx}"

            adapter.save_pretrained(
                model_path_to_save,
                is_main_process=is_main_process,
                save_function=save_function,
                safe_serialization=safe_serialization,
                variant=variant,
            )
        # Save FiLM + gate MLPs
        film_gate_state = {
            "film_mlps": self.film_mlps.state_dict(),
            "gate_mlps": self.gate_mlps.state_dict(),
            "hidden_dim": self.hidden_dim,
            "num_adapter": self.num_adapter,
        }
        torch.save(film_gate_state, os.path.join(save_directory, "pytorch_film_gate.bin"))

        # Save config (optional but recommended)
        config = {
            "hidden_dim": self.hidden_dim,
            "num_adapter": self.num_adapter,
            "total_downscale_factor": self.total_downscale_factor,
            "downscale_factor": self.downscale_factor,
            "enable_film": self.enable_film
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)


    @classmethod
    def from_pretrained(cls, pretrained_model_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a pretrained `MultiAdapter` model from multiple pre-trained adapter models.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, set it back to training mode using `model.train()`.

        Warnings:
            *Weights from XXX not initialized from pretrained model* means that the weights of XXX are not pretrained
            with the rest of the model. It is up to you to train those weights with a downstream fine-tuning. *Weights
            from XXX not used in YYY* means that the layer XXX is not used by YYY, so those weights are discarded.

        Args:
            pretrained_model_path (`os.PathLike`):
                A path to a *directory* containing model weights saved using
                [`~diffusers.models.adapter.MultiAdapter.save_pretrained`], e.g., `./my_model_directory/adapter`.
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary mapping device identifiers to their maximum memory. Default to the maximum memory
                available for each GPU and the available CPU RAM if unset.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading by not initializing the weights and only loading the pre-trained weights. This
                also tries to not use more than 1x model size in CPU memory (including peak memory) while loading the
                model. This is only supported when torch version >= 1.9.0. If you are using an older version of torch,
                setting this argument to `True` will raise an error.
            variant (`str`, *optional*):
                If specified, load weights from a `variant` file (*e.g.* pytorch_model.<variant>.bin). `variant` will
                be ignored when using `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If `None`, the `safetensors` weights will be downloaded if available **and** if`safetensors` library is
                installed. If `True`, the model will be forcibly loaded from`safetensors` weights. If `False`,
                `safetensors` is not used.
        """
        idx = 1
        adapters = []

        # load adapter and append to list until no adapter directory exists anymore
        # first adapter has to be saved under `./mydirectory/adapter` to be compliant with `DiffusionPipeline.from_pretrained`
        # second, third, ... adapters have to be saved under `./mydirectory/adapter_1`, `./mydirectory/adapter_2`, ...
        model_path_to_load = pretrained_model_path + f"_{idx}"
        while os.path.isdir(model_path_to_load):
            adapter = T2IAdapter.from_pretrained(model_path_to_load, **kwargs)
            adapters.append(adapter)

            idx += 1
            model_path_to_load = pretrained_model_path + f"_{idx}"

        logger.info(f"{len(adapters)} adapters loaded from {pretrained_model_path}.")

        if len(adapters) == 0:
            raise ValueError(
                f"No T2IAdapters found under {os.path.dirname(pretrained_model_path)}. Expected at least {pretrained_model_path + '_0'}."
            )

        # >> Instantiate MultiAdapter (MLPs not initialized yet)
        config_path = os.path.join(pretrained_model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            hidden_dim = cfg["hidden_dim"]
            enable_film = cfg["enable_film"]
        else:
            hidden_dim = 128   # fallback
            enable_film = False

        multi = cls(adapters, hidden_dim=hidden_dim,enable_film=enable_film)

        # >> Load FiLM + gating mlps
        film_gate_path = os.path.join(pretrained_model_path, "pytorch_film_gate.bin")
        if os.path.exists(film_gate_path):
            base_param = next(adapters[0].parameters())
            device = base_param.device
            state = torch.load(film_gate_path, map_location=device)

            multi.film_mlps.load_state_dict(state["film_mlps"])
            multi.gate_mlps.load_state_dict(state["gate_mlps"])
        else:
            print("⚠️ WARNING: No FiLM/gate state found; using fresh initialization")

        return multi


class GatedMultiAdapter1(ModelMixin):
    r"""
    MultiAdapter is a wrapper model that contains multiple adapter models and merges their outputs according to
    user-assigned weighting.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for common methods such as downloading
    or saving.

    Args:
        adapters (`List[T2IAdapter]`, *optional*, defaults to None):
            A list of `T2IAdapter` model instances.
    """

    def __init__(self, adapters: List["T2IAdapter"], hidden_dim: int = 128, enable_film: bool = True, temperature=5.0):
        super(GatedMultiAdapter, self).__init__()
        self.temperature = temperature
        self.num_adapter = len(adapters)
        self.adapters = nn.ModuleList(adapters)
        self.enable_film = enable_film
        if len(adapters) == 0:
            raise ValueError("Expecting at least one adapter")

        if len(adapters) == 1:
            raise ValueError("For a single adapter, please use the `T2IAdapter` class instead of `MultiAdapter`")

        # The outputs from each adapter are added together with a weight.
        # This means that the change in dimensions from downsampling must
        # be the same for all adapters. Inductively, it also means the
        # downscale_factor and total_downscale_factor must be the same for all
        # adapters.
        first_adapter_total_downscale_factor = adapters[0].total_downscale_factor
        first_adapter_downscale_factor = adapters[0].downscale_factor
        for idx in range(1, len(adapters)):
            if (
                adapters[idx].total_downscale_factor != first_adapter_total_downscale_factor
                or adapters[idx].downscale_factor != first_adapter_downscale_factor
            ):
                raise ValueError(
                    f"Expecting all adapters to have the same downscaling behavior, but got:\n"
                    f"adapters[0].total_downscale_factor={first_adapter_total_downscale_factor}\n"
                    f"adapters[0].downscale_factor={first_adapter_downscale_factor}\n"
                    f"adapter[`{idx}`].total_downscale_factor={adapters[idx].total_downscale_factor}\n"
                    f"adapter[`{idx}`].downscale_factor={adapters[idx].downscale_factor}"
                )

        self.total_downscale_factor = first_adapter_total_downscale_factor
        self.downscale_factor = first_adapter_downscale_factor

        # ---------- NEW: FiLM + gating MLP ----------
        # FiLM: maps [adapter_feature, timestep] → (gamma, beta)
        # gating network: [adapter_feature, timestep] → scalar gate
        self.hidden_dim = hidden_dim
        self.film_mlps = nn.ModuleList()  # len = num_scales (default = 4 len of channels) channels: List[int] = [320, 640, 1280, 1280], seen in T2I-adapter
        self.gate_mlps = nn.ModuleList()
        self._init_mlps_from_features()
    
    def _init_mlps_from_features(self, adapter_outputs):
        first_adapter_feats = adapter_outputs[0]  # list of 4 scales
        num_scales = len(first_adapter_feats)

        for k in range(num_scales):
            C_k = first_adapter_feats[k].shape[1]  # channel size, 如320 (4个channel分别是[320, 640, 1280, 1280])
            assert C_k in [320, 640, 1280, 1280]

            in_dim = C_k + 1  # pooled feature + timestep

            # FiLM_k
            film_mlp_k = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 2 * C_k )  # [gamma, beta]
            )

            # safe init: to let gamma ≈ 1, beta ≈ 0
            film_mlp_k[-1].weight.data.zero_()
            # bias: [ones for gamma | zeros for beta]
            film_mlp_k[-1].bias.data = torch.cat([
                torch.ones(C_k), 
                torch.zeros(C_k)
            ])

            # gate_k
            gate_mlp_k = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1), # scalar
            )

            self.film_mlps.append(film_mlp_k)
            self.gate_mlps.append(gate_mlp_k)

                # NEW: ensure newly created mlps go to correct device/dtype
        
        device = first_adapter_feats[0].device
        dtype = first_adapter_feats[0].dtype

        for mlp in self.film_mlps:
            mlp.to(device=device, dtype=dtype)

        for mlp in self.gate_mlps:
            mlp.to(device=device, dtype=dtype)

        self._mlp_inited = True

    def forward_static(self, xs: List[torch.Tensor]):
        """
        Only compute adapter backbone features (static across timesteps)
        xs: list of Tensor [(B, C, H, W), ...] length = num_adapter
        returns: List[num_adapter][num_scales], each scale is (B, Ck, Hk, Wk)
        """
        assert isinstance(xs, (list, tuple))
        assert len(xs) == self.num_adapter

        adapter_outputs = []
        for i, adapter in enumerate(self.adapters):
            feats_i = adapter(xs[i])    # list of 4 scales
            adapter_outputs.append(feats_i)

        if not self._mlp_inited:
            self._init_mlps_from_features(adapter_outputs)

        return adapter_outputs

    def forward_timestep(self, adapter_outputs, timestep: torch.Tensor):
        """
        Compute FiLM + gating using precomputed adapter_outputs
        adapter_outputs: List[num_adapter][num_scales]
        timestep: scalar or (B,)
        """
        if torch.isnan(timestep).any():
            raise ValueError("NaN in timestep !!!!!")

        for i, feat in enumerate(adapter_outputs):
            for f in feat:
                if torch.isnan(f).any():
                    raise ValueError(f"NaN in adapter_outputs[{i}] !!!!!")

        B = adapter_outputs[0][0].shape[0]
        ref_feat = adapter_outputs[0][0]  # for device/dtype

        # assert timestep.dim() == 1 and timestep.shape[0] == B # TODO: delete, true in training
        # Normalize timestep to tensor (B,1)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=ref_feat.device)
        if timestep.dim() == 0:
            t = timestep.view(1, 1).expand(B, 1).to(ref_feat.dtype) # scalar -> expand to (B,1)
        elif timestep.dim() == 1 and timestep.shape[0] == B:
            # batch -> (B,1)
            t = timestep.to(device=ref_feat.device, dtype=ref_feat.dtype).view(B, 1)
        else:
            raise ValueError(f"timestep shape must be () or (B,), got {timestep.shape}")

        fused_outputs = []
        num_scales = len(adapter_outputs[0])

        ## DEBUG
        _gate_logit_list = []
        for k in range(num_scales):
            gamma_list = []
            beta_list = []
            gate_logit_list = []

            for i in range(self.num_adapter):
                m_i_k = adapter_outputs[i][k]  # (B,C_k,H_k,W_k)
                C_k = m_i_k.shape[1]

                pooled = m_i_k.mean(dim=(2,3)) # (B, C_k)
                cond = torch.cat([pooled, t], dim=1) # (B, C_k + 1)

                film_out = self.film_mlps[k](cond)
                gamma_k_i, beta_k_i = torch.split(film_out, C_k, dim=-1)
                gamma_k_i = gamma_k_i.view(B, C_k, 1, 1)
                beta_k_i = beta_k_i.view(B, C_k, 1, 1)
                gamma_list.append(gamma_k_i)
                beta_list.append(beta_k_i)

                ## DEBUG
                if k == 0:
                    gate_logit_list.append(self.gate_mlps[k](cond))
                    _gate_logit_list.append(gate_logit_list[-1])
                else:
                    gate_logit_list.append(_gate_logit_list[i])
                ## DEBUG

            alpha_k = torch.softmax(torch.cat(gate_logit_list, dim=1)/ self.temperature, dim=1) # (B, num_adapter)

            ############## DEBUG: check sum == 1 ##############################
            alpha_sum = alpha_k.sum(dim=1)  # (B,)
            assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-3)
            
            ############## DEBUG: print gate weights by timestep ##############
            # Only print for specific timesteps (e.g. 5, 10, 20, 30)
            t_int = int(t[0].item())
            alpha_np = alpha_k[0].detach().float().cpu().numpy().tolist()
            print(f"[GATE] t={t_int}, scale={k}, weights={alpha_np}, sum={sum(alpha_np):.4f}")
            # ###################################################################

            fused_k = 0
            for i in range(self.num_adapter):
                # ######################### DEBUG #########################
                if self.enable_film:
                    modulated = gamma_list[i] * adapter_outputs[i][k] + beta_list[i]
                else:
                    modulated = adapter_outputs[i][k]
                fused_k = fused_k + modulated + alpha_k[:, i].view(B,1,1,1) * modulated

            fused_outputs.append(fused_k)

        for o in fused_outputs:
            if torch.isnan(o).any():
                raise ValueError("NaN in adapter fused_outputs !!!!!")

        return fused_outputs


    def forward(self, xs: torch.Tensor, timestep: torch.Tensor) -> List[torch.Tensor]:
        assert isinstance(xs, (list, tuple)), "Adapter input must be a list of Tensors"
        assert len(xs) == self.num_adapter, f"Expected {self.num_adapter} inputs, got {len(xs)}"

        device = xs[0].device
        dtype = self.adapters[0].dtype
        for a in self.adapters:
            a.to(device=device, dtype=dtype)
        for mlp in self.film_mlps:
            mlp.to(device=device, dtype=dtype)
        for mlp in self.gate_mlps:
            mlp.to(device=device, dtype=dtype)


        adapter_outputs = self.forward_static(xs)
        for aa in adapter_outputs:
            for a in aa:
                if torch.isnan(a).any():
                    raise ValueError("NaN in adapter adapter_outputs !!!!!")

        output = self.forward_timestep(adapter_outputs, timestep)
        return output


    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
    ):
        """
        Save a model and its configuration file to a specified directory, allowing it to be re-loaded with the
        `[`~models.adapter.MultiAdapter.from_pretrained`]` class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                The directory where the model will be saved. If the directory does not exist, it will be created.
            is_main_process (`bool`, optional, defaults=True):
                Indicates whether current process is the main process or not. Useful for distributed training (e.g.,
                TPUs) and need to call this function on all processes. In this case, set `is_main_process=True` only
                for the main process to avoid race conditions.
            save_function (`Callable`):
                Function used to save the state dictionary. Useful for distributed training (e.g., TPUs) to replace
                `torch.save` with another method. Can also be configured using`DIFFUSERS_SAVE_MODE` environment
                variable.
            safe_serialization (`bool`, optional, defaults=True):
                If `True`, save the model using `safetensors`. If `False`, save the model with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
        """
        os.makedirs(save_directory, exist_ok=True)
        idx = 0
        model_path_to_save = save_directory
        for adapter in self.adapters:
            idx += 1
            model_path_to_save = save_directory + f"_{idx}"

            adapter.save_pretrained(
                model_path_to_save,
                is_main_process=is_main_process,
                save_function=save_function,
                safe_serialization=safe_serialization,
                variant=variant,
            )
        
        # Save FiLM + gate MLPs
        film_gate_state = {
            "film_mlps": self.film_mlps.state_dict(),
            "gate_mlps": self.gate_mlps.state_dict(),
            "hidden_dim": self.hidden_dim,
            "num_adapter": self.num_adapter,
            "enable_film": self.enable_film,
            "temperature": self.temperature
        }
        torch.save(film_gate_state, os.path.join(save_directory, "pytorch_film_gate.bin"))

        # Save config (optional but recommended)
        config = {
            "hidden_dim": self.hidden_dim,
            "num_adapter": self.num_adapter,
            "total_downscale_factor": self.total_downscale_factor,
            "downscale_factor": self.downscale_factor,
            "enable_film": self.enable_film,
            "temperature": self.temperature
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)


    @classmethod
    def from_pretrained(cls, pretrained_model_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a pretrained `MultiAdapter` model from multiple pre-trained adapter models.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, set it back to training mode using `model.train()`.

        Warnings:
            *Weights from XXX not initialized from pretrained model* means that the weights of XXX are not pretrained
            with the rest of the model. It is up to you to train those weights with a downstream fine-tuning. *Weights
            from XXX not used in YYY* means that the layer XXX is not used by YYY, so those weights are discarded.

        Args:
            pretrained_model_path (`os.PathLike`):
                A path to a *directory* containing model weights saved using
                [`~diffusers.models.adapter.MultiAdapter.save_pretrained`], e.g., `./my_model_directory/adapter`.
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary mapping device identifiers to their maximum memory. Default to the maximum memory
                available for each GPU and the available CPU RAM if unset.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading by not initializing the weights and only loading the pre-trained weights. This
                also tries to not use more than 1x model size in CPU memory (including peak memory) while loading the
                model. This is only supported when torch version >= 1.9.0. If you are using an older version of torch,
                setting this argument to `True` will raise an error.
            variant (`str`, *optional*):
                If specified, load weights from a `variant` file (*e.g.* pytorch_model.<variant>.bin). `variant` will
                be ignored when using `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If `None`, the `safetensors` weights will be downloaded if available **and** if`safetensors` library is
                installed. If `True`, the model will be forcibly loaded from`safetensors` weights. If `False`,
                `safetensors` is not used.
        """
        idx = 1
        adapters = []

        # load adapter and append to list until no adapter directory exists anymore
        # first adapter has to be saved under `./mydirectory/adapter` to be compliant with `DiffusionPipeline.from_pretrained`
        # second, third, ... adapters have to be saved under `./mydirectory/adapter_1`, `./mydirectory/adapter_2`, ...
        model_path_to_load = pretrained_model_path + f"_{idx}"
        while os.path.isdir(model_path_to_load):
            adapter = T2IAdapter.from_pretrained(model_path_to_load, **kwargs)
            adapters.append(adapter)

            idx += 1
            model_path_to_load = pretrained_model_path + f"_{idx}"

        logger.info(f"{len(adapters)} adapters loaded from {pretrained_model_path}.")

        if len(adapters) == 0:
            raise ValueError(
                f"No T2IAdapters found under {os.path.dirname(pretrained_model_path)}. Expected at least {pretrained_model_path + '_0'}."
            )

        # >> Instantiate MultiAdapter (MLPs not initialized yet)
        config_path = os.path.join(pretrained_model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                hidden_dim = cfg["hidden_dim"]
                enable_film = cfg["enable_film"]
                temperature = cfg["temperature"]
            except:
                hidden_dim = 128   # fallback
                enable_film = False
                temperature = 5.0
        else:
            hidden_dim = 128   # fallback
            enable_film = False
            temperature = 5.0

        multi = cls(adapters, hidden_dim=hidden_dim, enable_film=enable_film, temperature = temperature)

        # >> Load FiLM + gating mlps
        film_gate_path = os.path.join(pretrained_model_path, "pytorch_film_gate.bin")
        if os.path.exists(film_gate_path):
            base_param = next(adapters[0].parameters())
            device = base_param.device
            state = torch.load(film_gate_path, map_location=device)

            # Figure out in_channels and downscale_factor robustly
            in_ch = getattr(adapters[0].config, "in_channels", None) if hasattr(adapters[0], "config") else None
            if in_ch is None:
                in_ch = getattr(adapters[0], "in_channels")

            downscale = getattr(adapters[0].config, "downscale_factor", None) if hasattr(adapters[0], "config") else None
            if downscale is None:
                downscale = getattr(adapters[0], "downscale_factor", 16)

            # Use a spatial size that is divisible by downscale_factor
            dummy_H = downscale
            dummy_W = downscale

            dummy = torch.zeros(
                1,
                in_ch,
                dummy_H,
                dummy_W,
                device=device,
                dtype=adapters[0].dtype,
            )

            # FiLM/MLP 的结构依赖于 C_k（实际 feature channel 数）
            # C_k 必须在 adapter_outputs[i][k] 经过一次前向推理后才能确定
            # Must initialize MLPs shape → run a dummy forward
            # We need adapter_outputs[k].shape to infer C_k
            # dummy = torch.zeros(1, multi.num_adapter * adapters[0].in_channels, 8, 8)
            # dummy_timestep = torch.tensor(1)
            # _ = multi(dummy, dummy_timestep)   # this triggers MLP init
            dummy_list = [dummy for _ in range(multi.num_adapter)]
            with torch.no_grad():
                _ = multi.forward_static(dummy_list)  # this triggers _init_mlps_from_features

            multi.film_mlps.load_state_dict(state["film_mlps"])
            multi.gate_mlps.load_state_dict(state["gate_mlps"])
        else:
            print("⚠️ WARNING: No FiLM/gate state found; using fresh initialization")

        return multi

class MultiAdapter(ModelMixin):
    r"""
    MultiAdapter is a wrapper model that contains multiple adapter models and merges their outputs according to
    user-assigned weighting.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for common methods such as downloading
    or saving.

    Args:
        adapters (`List[T2IAdapter]`, *optional*, defaults to None):
            A list of `T2IAdapter` model instances.
    """

    def __init__(self, adapters: List["T2IAdapter"]):
        super(MultiAdapter, self).__init__()

        self.num_adapter = len(adapters)
        self.adapters = nn.ModuleList(adapters)

        if len(adapters) == 0:
            raise ValueError("Expecting at least one adapter")

        if len(adapters) == 1:
            raise ValueError("For a single adapter, please use the `T2IAdapter` class instead of `MultiAdapter`")

        # The outputs from each adapter are added together with a weight.
        # This means that the change in dimensions from downsampling must
        # be the same for all adapters. Inductively, it also means the
        # downscale_factor and total_downscale_factor must be the same for all
        # adapters.
        first_adapter_total_downscale_factor = adapters[0].total_downscale_factor
        first_adapter_downscale_factor = adapters[0].downscale_factor
        for idx in range(1, len(adapters)):
            if (
                adapters[idx].total_downscale_factor != first_adapter_total_downscale_factor
                or adapters[idx].downscale_factor != first_adapter_downscale_factor
            ):
                raise ValueError(
                    f"Expecting all adapters to have the same downscaling behavior, but got:\n"
                    f"adapters[0].total_downscale_factor={first_adapter_total_downscale_factor}\n"
                    f"adapters[0].downscale_factor={first_adapter_downscale_factor}\n"
                    f"adapter[`{idx}`].total_downscale_factor={adapters[idx].total_downscale_factor}\n"
                    f"adapter[`{idx}`].downscale_factor={adapters[idx].downscale_factor}"
                )

        self.total_downscale_factor = first_adapter_total_downscale_factor
        self.downscale_factor = first_adapter_downscale_factor

    def forward(self, xs: torch.Tensor, adapter_weights: Optional[List[float]] = None) -> List[torch.Tensor]:
        r"""
        Args:
            xs (`torch.Tensor`):
                A tensor of shape (batch, channel, height, width) representing input images for multiple adapter
                models, concatenated along dimension 1(channel dimension). The `channel` dimension should be equal to
                `num_adapter` * number of channel per image.

            adapter_weights (`List[float]`, *optional*, defaults to None):
                A list of floats representing the weights which will be multiplied by each adapter's output before
                summing them together. If `None`, equal weights will be used for all adapters.
        """
        if adapter_weights is None:
            adapter_weights = torch.tensor([1 / self.num_adapter] * self.num_adapter)
        else:
            adapter_weights = torch.tensor(adapter_weights)

        accume_state = None
        for x, w, adapter in zip(xs, adapter_weights, self.adapters):
            features = adapter(x)
            if accume_state is None:
                accume_state = features
                for i in range(len(accume_state)):
                    accume_state[i] = w * accume_state[i]
            else:
                for i in range(len(features)):
                    accume_state[i] += w * features[i]
        return accume_state

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
    ):
        """
        Save a model and its configuration file to a specified directory, allowing it to be re-loaded with the
        `[`~models.adapter.MultiAdapter.from_pretrained`]` class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                The directory where the model will be saved. If the directory does not exist, it will be created.
            is_main_process (`bool`, optional, defaults=True):
                Indicates whether current process is the main process or not. Useful for distributed training (e.g.,
                TPUs) and need to call this function on all processes. In this case, set `is_main_process=True` only
                for the main process to avoid race conditions.
            save_function (`Callable`):
                Function used to save the state dictionary. Useful for distributed training (e.g., TPUs) to replace
                `torch.save` with another method. Can also be configured using`DIFFUSERS_SAVE_MODE` environment
                variable.
            safe_serialization (`bool`, optional, defaults=True):
                If `True`, save the model using `safetensors`. If `False`, save the model with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
        """
        idx = 0
        model_path_to_save = save_directory
        for adapter in self.adapters:
            adapter.save_pretrained(
                model_path_to_save,
                is_main_process=is_main_process,
                save_function=save_function,
                safe_serialization=safe_serialization,
                variant=variant,
            )

            idx += 1
            model_path_to_save = model_path_to_save + f"_{idx}"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a pretrained `MultiAdapter` model from multiple pre-trained adapter models.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, set it back to training mode using `model.train()`.

        Warnings:
            *Weights from XXX not initialized from pretrained model* means that the weights of XXX are not pretrained
            with the rest of the model. It is up to you to train those weights with a downstream fine-tuning. *Weights
            from XXX not used in YYY* means that the layer XXX is not used by YYY, so those weights are discarded.

        Args:
            pretrained_model_path (`os.PathLike`):
                A path to a *directory* containing model weights saved using
                [`~diffusers.models.adapter.MultiAdapter.save_pretrained`], e.g., `./my_model_directory/adapter`.
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary mapping device identifiers to their maximum memory. Default to the maximum memory
                available for each GPU and the available CPU RAM if unset.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading by not initializing the weights and only loading the pre-trained weights. This
                also tries to not use more than 1x model size in CPU memory (including peak memory) while loading the
                model. This is only supported when torch version >= 1.9.0. If you are using an older version of torch,
                setting this argument to `True` will raise an error.
            variant (`str`, *optional*):
                If specified, load weights from a `variant` file (*e.g.* pytorch_model.<variant>.bin). `variant` will
                be ignored when using `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If `None`, the `safetensors` weights will be downloaded if available **and** if`safetensors` library is
                installed. If `True`, the model will be forcibly loaded from`safetensors` weights. If `False`,
                `safetensors` is not used.
        """
        idx = 0
        adapters = []

        # load adapter and append to list until no adapter directory exists anymore
        # first adapter has to be saved under `./mydirectory/adapter` to be compliant with `DiffusionPipeline.from_pretrained`
        # second, third, ... adapters have to be saved under `./mydirectory/adapter_1`, `./mydirectory/adapter_2`, ...
        model_path_to_load = pretrained_model_path
        while os.path.isdir(model_path_to_load):
            adapter = T2IAdapter.from_pretrained(model_path_to_load, **kwargs)
            adapters.append(adapter)

            idx += 1
            model_path_to_load = pretrained_model_path + f"_{idx}"

        logger.info(f"{len(adapters)} adapters loaded from {pretrained_model_path}.")

        if len(adapters) == 0:
            raise ValueError(
                f"No T2IAdapters found under {os.path.dirname(pretrained_model_path)}. Expected at least {pretrained_model_path + '_0'}."
            )

        return cls(adapters)

class T2IAdapter(ModelMixin, ConfigMixin):
    r"""
    A simple ResNet-like model that accepts images containing control signals such as keyposes and depth. The model
    generates multiple feature maps that are used as additional conditioning in [`UNet2DConditionModel`]. The model's
    architecture follows the original implementation of
    [Adapter](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L97)
     and
     [AdapterLight](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L235).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the common methods, such as
    downloading or saving.

    Args:
        in_channels (`int`, *optional*, defaults to `3`):
            The number of channels in the adapter's input (*control image*). Set it to 1 if you're using a gray scale
            image.
        channels (`List[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The number of channels in each downsample block's output hidden state. The `len(block_out_channels)`
            determines the number of downsample blocks in the adapter.
        num_res_blocks (`int`, *optional*, defaults to `2`):
            Number of ResNet blocks in each downsample block.
        downscale_factor (`int`, *optional*, defaults to `8`):
            A factor that determines the total downscale factor of the Adapter.
        adapter_type (`str`, *optional*, defaults to `full_adapter`):
            Adapter type (`full_adapter` or `full_adapter_xl` or `light_adapter`) to use.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
        adapter_type: str = "full_adapter",
    ):
        super().__init__()

        if adapter_type == "full_adapter":
            self.adapter = FullAdapter(in_channels, channels, num_res_blocks, downscale_factor)
        elif adapter_type == "full_adapter_xl":
            self.adapter = FullAdapterXL(in_channels, channels, num_res_blocks, downscale_factor)
        elif adapter_type == "light_adapter":
            self.adapter = LightAdapter(in_channels, channels, num_res_blocks, downscale_factor)
        else:
            raise ValueError(
                f"Unsupported adapter_type: '{adapter_type}'. Choose either 'full_adapter' or "
                "'full_adapter_xl' or 'light_adapter'."
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        This function processes the input tensor `x` through the adapter model and returns a list of feature tensors,
        each representing information extracted at a different scale from the input. The length of the list is
        determined by the number of downsample blocks in the Adapter, as specified by the `channels` and
        `num_res_blocks` parameters during initialization.
        """
        return self.adapter(x)

    @property
    def total_downscale_factor(self):
        return self.adapter.total_downscale_factor

    @property
    def downscale_factor(self):
        """The downscale factor applied in the T2I-Adapter's initial pixel unshuffle operation. If an input image's dimensions are
        not evenly divisible by the downscale_factor then an exception will be raised.
        """
        return self.adapter.unshuffle.downscale_factor


# full adapter


class FullAdapter(nn.Module):
    r"""
    See [`T2IAdapter`] for more information.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.body = nn.ModuleList(
            [
                AdapterBlock(channels[0], channels[0], num_res_blocks),
                *[
                    AdapterBlock(channels[i - 1], channels[i], num_res_blocks, down=True)
                    for i in range(1, len(channels))
                ],
            ]
        )

        self.total_downscale_factor = downscale_factor * 2 ** (len(channels) - 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        This method processes the input tensor `x` through the FullAdapter model and performs operations including
        pixel unshuffling, convolution, and a stack of AdapterBlocks. It returns a list of feature tensors, each
        capturing information at a different stage of processing within the FullAdapter model. The number of feature
        tensors in the list is determined by the number of downsample blocks specified during initialization.
        """
        x = self.unshuffle(x)
        x = self.conv_in(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class FullAdapterXL(nn.Module):
    r"""
    See [`T2IAdapter`] for more information.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 16,
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.body = []
        # blocks to extract XL features with dimensions of [320, 64, 64], [640, 64, 64], [1280, 32, 32], [1280, 32, 32]
        for i in range(len(channels)):
            if i == 1:
                self.body.append(AdapterBlock(channels[i - 1], channels[i], num_res_blocks))
            elif i == 2:
                self.body.append(AdapterBlock(channels[i - 1], channels[i], num_res_blocks, down=True))
            else:
                self.body.append(AdapterBlock(channels[i], channels[i], num_res_blocks))

        self.body = nn.ModuleList(self.body)
        # XL has only one downsampling AdapterBlock.
        self.total_downscale_factor = downscale_factor * 2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        This method takes the tensor x as input and processes it through FullAdapterXL model. It consists of operations
        including unshuffling pixels, applying convolution layer and appending each block into list of feature tensors.
        """
        x = self.unshuffle(x)
        x = self.conv_in(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class AdapterBlock(nn.Module):
    r"""
    An AdapterBlock is a helper model that contains multiple ResNet-like blocks. It is used in the `FullAdapter` and
    `FullAdapterXL` models.

    Args:
        in_channels (`int`):
            Number of channels of AdapterBlock's input.
        out_channels (`int`):
            Number of channels of AdapterBlock's output.
        num_res_blocks (`int`):
            Number of ResNet blocks in the AdapterBlock.
        down (`bool`, *optional*, defaults to `False`):
            If `True`, perform downsampling on AdapterBlock's input.
    """

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int, down: bool = False):
        super().__init__()

        self.downsample = None
        if down:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.in_conv = None
        if in_channels != out_channels:
            self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.resnets = nn.Sequential(
            *[AdapterResnetBlock(out_channels) for _ in range(num_res_blocks)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This method takes tensor x as input and performs operations downsampling and convolutional layers if the
        self.downsample and self.in_conv properties of AdapterBlock model are specified. Then it applies a series of
        residual blocks to the input tensor.
        """
        if self.downsample is not None:
            x = self.downsample(x)

        if self.in_conv is not None:
            x = self.in_conv(x)

        x = self.resnets(x)

        return x


class AdapterResnetBlock(nn.Module):
    r"""
    An `AdapterResnetBlock` is a helper model that implements a ResNet-like block.

    Args:
        channels (`int`):
            Number of channels of AdapterResnetBlock's input and output.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This method takes input tensor x and applies a convolutional layer, ReLU activation, and another convolutional
        layer on the input tensor. It returns addition with the input tensor.
        """

        h = self.act(self.block1(x))
        h = self.block2(h)

        return h + x


# light adapter


class LightAdapter(nn.Module):
    r"""
    See [`T2IAdapter`] for more information.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280],
        num_res_blocks: int = 4,
        downscale_factor: int = 8,
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)

        self.body = nn.ModuleList(
            [
                LightAdapterBlock(in_channels, channels[0], num_res_blocks),
                *[
                    LightAdapterBlock(channels[i], channels[i + 1], num_res_blocks, down=True)
                    for i in range(len(channels) - 1)
                ],
                LightAdapterBlock(channels[-1], channels[-1], num_res_blocks, down=True),
            ]
        )

        self.total_downscale_factor = downscale_factor * (2 ** len(channels))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        This method takes the input tensor x and performs downscaling and appends it in list of feature tensors. Each
        feature tensor corresponds to a different level of processing within the LightAdapter.
        """
        x = self.unshuffle(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class LightAdapterBlock(nn.Module):
    r"""
    A `LightAdapterBlock` is a helper model that contains multiple `LightAdapterResnetBlocks`. It is used in the
    `LightAdapter` model.

    Args:
        in_channels (`int`):
            Number of channels of LightAdapterBlock's input.
        out_channels (`int`):
            Number of channels of LightAdapterBlock's output.
        num_res_blocks (`int`):
            Number of LightAdapterResnetBlocks in the LightAdapterBlock.
        down (`bool`, *optional*, defaults to `False`):
            If `True`, perform downsampling on LightAdapterBlock's input.
    """

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int, down: bool = False):
        super().__init__()
        mid_channels = out_channels // 4

        self.downsample = None
        if down:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.resnets = nn.Sequential(*[LightAdapterResnetBlock(mid_channels) for _ in range(num_res_blocks)])
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This method takes tensor x as input and performs downsampling if required. Then it applies in convolution
        layer, a sequence of residual blocks, and out convolutional layer.
        """
        if self.downsample is not None:
            x = self.downsample(x)

        x = self.in_conv(x)
        x = self.resnets(x)
        x = self.out_conv(x)

        return x


class LightAdapterResnetBlock(nn.Module):
    """
    A `LightAdapterResnetBlock` is a helper model that implements a ResNet-like block with a slightly different
    architecture than `AdapterResnetBlock`.

    Args:
        channels (`int`):
            Number of channels of LightAdapterResnetBlock's input and output.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This function takes input tensor x and processes it through one convolutional layer, ReLU activation, and
        another convolutional layer and adds it to input tensor.
        """

        h = self.act(self.block1(x))
        h = self.block2(h)

        return h + x
