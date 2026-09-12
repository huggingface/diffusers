# Copyright (c) 2025 SandAI. All Rights Reserved.
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

import html
import re
import urllib.parse as ul

import torch
from transformers import AutoTokenizer, T5EncoderModel

from ...models import AutoencoderKLMagi
from ...utils import is_bs4_available, is_ftfy_available, requires_backends
from ..modular_pipeline import ModularPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam


if is_bs4_available():
    from bs4 import BeautifulSoup
if is_ftfy_available():
    import ftfy


class MagiTextEncoderStep(ModularPipelineBlocks):
    model_name = "magi"
    bad_punct_regex = re.compile(r"[#®•©™&@·º½¾¿¡§~\)\(\]\[\}\{\|\\\/\*]{1,}")

    @property
    def description(self):
        return "Encode prompts with T5 after the official two-pass caption cleaning."

    @property
    def expected_components(self):
        return [ComponentSpec("text_encoder", T5EncoderModel), ComponentSpec("tokenizer", AutoTokenizer)]

    @property
    def inputs(self):
        return [
            InputParam.template("prompt", required=True),
            InputParam("max_sequence_length", default=800, type_hint=int, description="Padded T5 caption length."),
            InputParam(
                "clean_caption", default=True, type_hint=bool, description="Apply the official two-pass text cleaning."
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "text_embeds",
                type_hint=torch.Tensor,
                description="Per-prompt FP32 T5 features before special-token insertion.",
            ),
            OutputParam("text_attention_mask", type_hint=torch.Tensor, description="Per-prompt boolean T5 keep-mask."),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        prompts = [block_state.prompt] if isinstance(block_state.prompt, str) else block_state.prompt
        if not isinstance(prompts, list) or not prompts or not all(isinstance(prompt, str) for prompt in prompts):
            raise ValueError("prompt must be a string or a nonempty list of strings.")
        if block_state.clean_caption:
            requires_backends(self, ["bs4", "ftfy"])
            prompts = [self.clean_caption(self.clean_caption(prompt)) for prompt in prompts]
        else:
            prompts = [prompt.lower().strip() for prompt in prompts]
        if not isinstance(block_state.max_sequence_length, int) or block_state.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be a positive integer.")
        embeddings, masks = [], []
        for prompt in prompts:
            tokens = components.tokenizer(
                [prompt],
                max_length=block_state.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            tokens = tokens.to(components.text_encoder.device)
            embeddings.append(
                components.text_encoder(
                    input_ids=tokens.input_ids, attention_mask=tokens.attention_mask
                ).last_hidden_state.float()
            )
            masks.append(tokens.attention_mask.bool())
        block_state.text_embeds = torch.cat(embeddings, dim=0)
        block_state.text_attention_mask = torch.cat(masks, dim=0)
        self.set_block_state(state, block_state)
        return components, state

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()


def encode_magi_prefix(vae, video, scaling_factor, device):
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.dtype != torch.uint8:
        raise ValueError("Prefix pixels must be a uint8 tensor shaped (batch, 3, frames, height, width).")
    if min(video.shape) <= 0 or video.shape[1] != 3:
        raise ValueError("Prefix pixels must be nonempty RGB frames.")
    if scaling_factor <= 0:
        raise ValueError("latent_scaling_factor must be positive.")
    pixels = (video.to(device=device, dtype=torch.float32) / 127.5 - 1).to(vae.dtype)
    return vae.encode(pixels).latent_dist.mode() * scaling_factor


class MagiVideoVaeEncoderStep(ModularPipelineBlocks):
    model_name = "magi"

    @property
    def description(self):
        return "Encode pre-resized uint8 RGB video frames as a deterministic, scaled VAE prefix."

    @property
    def expected_components(self):
        return [ComponentSpec("vae", AutoencoderKLMagi)]

    @property
    def expected_configs(self):
        return [ConfigSpec("latent_scaling_factor", 0.18215)]

    @property
    def inputs(self):
        return [
            InputParam(
                "video",
                required=True,
                type_hint=torch.Tensor,
                description="Pre-resized uint8 RGB prefix, shaped (batch, 3, frames, height, width).",
            )
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "conditioning_latents",
                type_hint=torch.Tensor,
                description="Per-prompt scaled VAE prefix, before video-batch expansion.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        s = self.get_block_state(state)
        s.conditioning_latents = encode_magi_prefix(
            components.vae, s.video, components.config.latent_scaling_factor, components._execution_device
        )
        self.set_block_state(state, s)
        return components, state


class MagiImageVaeEncoderStep(MagiVideoVaeEncoderStep):
    @property
    def description(self):
        return "Encode pre-resized uint8 RGB images as a one-latent-frame prefix."

    @property
    def inputs(self):
        return [
            InputParam(
                "image",
                required=True,
                type_hint=torch.Tensor,
                description="Pre-resized uint8 RGB images, shaped (batch, 3, height, width).",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        s = self.get_block_state(state)
        if not isinstance(s.image, torch.Tensor) or s.image.ndim != 4:
            raise ValueError("image must have shape (batch, 3, height, width).")
        s.conditioning_latents = encode_magi_prefix(
            components.vae, s.image.unsqueeze(2), components.config.latent_scaling_factor, components._execution_device
        )
        self.set_block_state(state, s)
        return components, state
