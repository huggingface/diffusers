# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    _LazyModule,
    is_flax_available,
    is_torch_available,
)


_import_structure = {}

if is_torch_available():
    _import_structure["adapter"] = ["MultiAdapter", "T2IAdapter"]
    _import_structure["autoencoders.autoencoder_asym_kl"] = ["AsymmetricAutoencoderKL"]
    _import_structure["autoencoders.autoencoder_dc"] = ["AutoencoderDC"]
    _import_structure["autoencoders.autoencoder_kl"] = ["AutoencoderKL"]
    _import_structure["autoencoders.autoencoder_kl_allegro"] = ["AutoencoderKLAllegro"]
    _import_structure["autoencoders.autoencoder_kl_cogvideox"] = ["AutoencoderKLCogVideoX"]
    _import_structure["autoencoders.autoencoder_kl_hunyuan_video"] = ["AutoencoderKLHunyuanVideo"]
    _import_structure["autoencoders.autoencoder_kl_ltx"] = ["AutoencoderKLLTXVideo"]
    _import_structure["autoencoders.autoencoder_kl_mochi"] = ["AutoencoderKLMochi"]
    _import_structure["autoencoders.autoencoder_kl_temporal_decoder"] = ["AutoencoderKLTemporalDecoder"]
    _import_structure["autoencoders.autoencoder_oobleck"] = ["AutoencoderOobleck"]
    _import_structure["autoencoders.autoencoder_tiny"] = ["AutoencoderTiny"]
    _import_structure["autoencoders.consistency_decoder_vae"] = ["ConsistencyDecoderVAE"]
    _import_structure["autoencoders.vq_model"] = ["VQModel"]
    _import_structure["controlnets.controlnet"] = ["ControlNetModel"]
    _import_structure["controlnets.controlnet_flux"] = ["FluxControlNetModel", "FluxMultiControlNetModel"]
    _import_structure["controlnets.controlnet_hunyuan"] = [
        "HunyuanDiT2DControlNetModel",
        "HunyuanDiT2DMultiControlNetModel",
    ]
    _import_structure["controlnets.controlnet_sd3"] = ["SD3ControlNetModel", "SD3MultiControlNetModel"]
    _import_structure["controlnets.controlnet_sparsectrl"] = ["SparseControlNetModel"]
    _import_structure["controlnets.controlnet_union"] = ["ControlNetUnionModel"]
    _import_structure["controlnets.controlnet_xs"] = ["ControlNetXSAdapter", "UNetControlNetXSModel"]
    _import_structure["controlnets.multicontrolnet"] = ["MultiControlNetModel"]
    _import_structure["embeddings"] = ["ImageProjection"]
    _import_structure["modeling_utils"] = ["ModelMixin"]
    _import_structure["transformers.auraflow_transformer_2d"] = ["AuraFlowTransformer2DModel"]
    _import_structure["transformers.cogvideox_transformer_3d"] = ["CogVideoXTransformer3DModel"]
    _import_structure["transformers.dit_transformer_2d"] = ["DiTTransformer2DModel"]
    _import_structure["transformers.dual_transformer_2d"] = ["DualTransformer2DModel"]
    _import_structure["transformers.hunyuan_transformer_2d"] = ["HunyuanDiT2DModel"]
    _import_structure["transformers.latte_transformer_3d"] = ["LatteTransformer3DModel"]
    _import_structure["transformers.lumina_nextdit2d"] = ["LuminaNextDiT2DModel"]
    _import_structure["transformers.pixart_transformer_2d"] = ["PixArtTransformer2DModel"]
    _import_structure["transformers.prior_transformer"] = ["PriorTransformer"]
    _import_structure["transformers.sana_transformer"] = ["SanaTransformer2DModel"]
    _import_structure["transformers.stable_audio_transformer"] = ["StableAudioDiTModel"]
    _import_structure["transformers.t5_film_transformer"] = ["T5FilmDecoder"]
    _import_structure["transformers.transformer_2d"] = ["Transformer2DModel"]
    _import_structure["transformers.transformer_allegro"] = ["AllegroTransformer3DModel"]
    _import_structure["transformers.transformer_cogview3plus"] = ["CogView3PlusTransformer2DModel"]
    _import_structure["transformers.transformer_flux"] = ["FluxTransformer2DModel"]
    _import_structure["transformers.transformer_hunyuan_video"] = ["HunyuanVideoTransformer3DModel"]
    _import_structure["transformers.transformer_ltx"] = ["LTXVideoTransformer3DModel"]
    _import_structure["transformers.transformer_mochi"] = ["MochiTransformer3DModel"]
    _import_structure["transformers.transformer_sd3"] = ["SD3Transformer2DModel"]
    _import_structure["transformers.transformer_temporal"] = ["TransformerTemporalModel"]
    _import_structure["unets.unet_1d"] = ["UNet1DModel"]
    _import_structure["unets.unet_2d"] = ["UNet2DModel"]
    _import_structure["unets.unet_2d_condition"] = ["UNet2DConditionModel"]
    _import_structure["unets.unet_3d_condition"] = ["UNet3DConditionModel"]
    _import_structure["unets.unet_i2vgen_xl"] = ["I2VGenXLUNet"]
    _import_structure["unets.unet_kandinsky3"] = ["Kandinsky3UNet"]
    _import_structure["unets.unet_motion_model"] = ["MotionAdapter", "UNetMotionModel"]
    _import_structure["unets.unet_spatio_temporal_condition"] = ["UNetSpatioTemporalConditionModel"]
    _import_structure["unets.unet_stable_cascade"] = ["StableCascadeUNet"]
    _import_structure["unets.uvit_2d"] = ["UVit2DModel"]

if is_flax_available():
    _import_structure["controlnets.controlnet_flax"] = ["FlaxControlNetModel"]
    _import_structure["unets.unet_2d_condition_flax"] = ["FlaxUNet2DConditionModel"]
    _import_structure["vae_flax"] = ["FlaxAutoencoderKL"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        from .adapter import MultiAdapter, T2IAdapter
        from .autoencoders import (
            AsymmetricAutoencoderKL,
            AutoencoderDC,
            AutoencoderKL,
            AutoencoderKLAllegro,
            AutoencoderKLCogVideoX,
            AutoencoderKLHunyuanVideo,
            AutoencoderKLLTXVideo,
            AutoencoderKLMochi,
            AutoencoderKLTemporalDecoder,
            AutoencoderOobleck,
            AutoencoderTiny,
            ConsistencyDecoderVAE,
            VQModel,
        )
        from .controlnets import (
            ControlNetModel,
            ControlNetUnionModel,
            ControlNetXSAdapter,
            FluxControlNetModel,
            FluxMultiControlNetModel,
            HunyuanDiT2DControlNetModel,
            HunyuanDiT2DMultiControlNetModel,
            MultiControlNetModel,
            SD3ControlNetModel,
            SD3MultiControlNetModel,
            SparseControlNetModel,
            UNetControlNetXSModel,
        )
        from .embeddings import ImageProjection
        from .modeling_utils import ModelMixin
        from .transformers import (
            AllegroTransformer3DModel,
            AuraFlowTransformer2DModel,
            CogVideoXTransformer3DModel,
            CogView3PlusTransformer2DModel,
            DiTTransformer2DModel,
            DualTransformer2DModel,
            FluxTransformer2DModel,
            HunyuanDiT2DModel,
            HunyuanVideoTransformer3DModel,
            LatteTransformer3DModel,
            LTXVideoTransformer3DModel,
            LuminaNextDiT2DModel,
            MochiTransformer3DModel,
            PixArtTransformer2DModel,
            PriorTransformer,
            SanaTransformer2DModel,
            SD3Transformer2DModel,
            StableAudioDiTModel,
            T5FilmDecoder,
            Transformer2DModel,
            TransformerTemporalModel,
        )
        from .unets import (
            I2VGenXLUNet,
            Kandinsky3UNet,
            MotionAdapter,
            StableCascadeUNet,
            UNet1DModel,
            UNet2DConditionModel,
            UNet2DModel,
            UNet3DConditionModel,
            UNetMotionModel,
            UNetSpatioTemporalConditionModel,
            UVit2DModel,
        )

    if is_flax_available():
        from .controlnets import FlaxControlNetModel
        from .unets import FlaxUNet2DConditionModel
        from .vae_flax import FlaxAutoencoderKL

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
