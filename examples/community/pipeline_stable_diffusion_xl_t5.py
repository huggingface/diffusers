# Copyright Philip Brown, ppbrown@github
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

###########################################################################
# This pipeline attempts to use a model that has SDXL vae, T5 text encoder,
# and SDXL unet.
# At the present time, there are no pretrained models that give pleasing
# output. So as yet, (2025/06/10) this pipeline is somewhat of a tech
# demo proving that the pieces can at least be put together.
# Hopefully, it will encourage someone with the hardware available to
# throw enough resources into training one up.


from typing import Optional

import torch.nn as nn
from transformers import (
    CLIPImageProcessor,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
)

from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers


# Note: At this time, the intent is to use the T5 encoder mentioned
# below, with zero changes.
# Therefore, the model deliberately does not store the T5 encoder model bytes,
# (Since they are not unique!)
# but instead takes advantage of huggingface hub cache loading

T5_NAME = "mcmonkey/google_t5-v1_1-xxl_encoderonly"

# Caller is expected to load this, or equivalent, as model name for now
#   eg: pipe = StableDiffusionXL_T5Pipeline(SDXL_NAME)
SDXL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"


class LinearWithDtype(nn.Linear):
    @property
    def dtype(self):
        return self.weight.dtype


class StableDiffusionXL_T5Pipeline(StableDiffusionXLPipeline):
    _expected_modules = [
        "vae",
        "unet",
        "scheduler",
        "tokenizer",
        "image_encoder",
        "feature_extractor",
        "t5_encoder",
        "t5_projection",
        "t5_pooled_projection",
    ]

    _optional_components = [
        "image_encoder",
        "feature_extractor",
        "t5_encoder",
        "t5_projection",
        "t5_pooled_projection",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        tokenizer: CLIPTokenizer,
        t5_encoder=None,
        t5_projection=None,
        t5_pooled_projection=None,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        DiffusionPipeline.__init__(self)

        if t5_encoder is None:
            self.t5_encoder = T5EncoderModel.from_pretrained(T5_NAME, torch_dtype=unet.dtype)
        else:
            self.t5_encoder = t5_encoder

        # ----- build T5 4096 => 2048 dim projection -----
        if t5_projection is None:
            self.t5_projection = LinearWithDtype(4096, 2048)  # trainable
        else:
            self.t5_projection = t5_projection
        self.t5_projection.to(dtype=unet.dtype)
        # ----- build T5 4096 => 1280 dim projection -----
        if t5_pooled_projection is None:
            self.t5_pooled_projection = LinearWithDtype(4096, 1280)  # trainable
        else:
            self.t5_pooled_projection = t5_pooled_projection
        self.t5_pooled_projection.to(dtype=unet.dtype)

        print("dtype of Linear is ", self.t5_projection.dtype)

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            t5_encoder=self.t5_encoder,
            t5_projection=self.t5_projection,
            t5_pooled_projection=self.t5_pooled_projection,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.default_sample_size = (
            self.unet.config.sample_size
            if hasattr(self, "unet") and self.unet is not None and hasattr(self.unet.config, "sample_size")
            else 128
        )

        self.watermark = None

        # Parts of original SDXL class complain if these attributes are not
        # at least PRESENT
        self.text_encoder = self.text_encoder_2 = None

    # ------------------------------------------------------------------
    #  Encode a text prompt (T5-XXL + 4096→2048 projection)
    #  Returns exactly four tensors in the order SDXL’s __call__ expects.
    # ------------------------------------------------------------------
    def encode_prompt(
        self,
        prompt,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | None = None,
        **_,
    ):
        """
        Returns
        -------
        prompt_embeds                : Tensor [B, T, 2048]
        negative_prompt_embeds       : Tensor [B, T, 2048] | None
        pooled_prompt_embeds         : Tensor [B, 1280]
        negative_pooled_prompt_embeds: Tensor [B, 1280]    | None
        where B = batch * num_images_per_prompt
        """

        # --- helper to tokenize on the pipeline’s device ----------------
        def _tok(text: str):
            tok_out = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).to(self.device)
            return tok_out.input_ids, tok_out.attention_mask

        # ---------- positive stream -------------------------------------
        ids, mask = _tok(prompt)
        h_pos = self.t5_encoder(ids, attention_mask=mask).last_hidden_state  # [b, T, 4096]
        tok_pos = self.t5_projection(h_pos)  # [b, T, 2048]
        pool_pos = self.t5_pooled_projection(h_pos.mean(dim=1))  # [b, 1280]

        # expand for multiple images per prompt
        tok_pos = tok_pos.repeat_interleave(num_images_per_prompt, 0)
        pool_pos = pool_pos.repeat_interleave(num_images_per_prompt, 0)

        # ---------- negative / CFG stream --------------------------------
        if do_classifier_free_guidance:
            neg_text = "" if negative_prompt is None else negative_prompt
            ids_n, mask_n = _tok(neg_text)
            h_neg = self.t5_encoder(ids_n, attention_mask=mask_n).last_hidden_state
            tok_neg = self.t5_projection(h_neg)
            pool_neg = self.t5_pooled_projection(h_neg.mean(dim=1))

            tok_neg = tok_neg.repeat_interleave(num_images_per_prompt, 0)
            pool_neg = pool_neg.repeat_interleave(num_images_per_prompt, 0)
        else:
            tok_neg = pool_neg = None

        # ----------------- final ordered return --------------------------
        # 1) positive token embeddings
        # 2) negative token embeddings (or None)
        # 3) positive pooled embeddings
        # 4) negative pooled embeddings (or None)
        return tok_pos, tok_neg, pool_pos, pool_neg
