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

from ...utils import logging
from ..modular_pipeline import SequentialPipelineBlocks
from .before_denoise import DiffusionGemmaPrepareGenerationStep, DiffusionGemmaSetTimestepsStep
from .decoders import DiffusionGemmaDecodeStep
from .denoise import DiffusionGemmaDenoiseStep
from .encoders import DiffusionGemmaTextEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# text_encoder -> prepare_generation -> set_timesteps -> denoise -> decode
# auto_docstring
class DiffusionGemmaBlocks(SequentialPipelineBlocks):
    """
    Modular blocks for DiffusionGemma block-diffusion text generation.
      - `text_encoder` applies the chat template and tokenizes the prompt
      - `prepare_generation` sizes the canvases, creates the KV cache and resolves EOS
      - `set_timesteps` splits the forward budget and configures the scheduler
      - `denoise` generates the text canvas by canvas
      - `decode` trims at EOS and decodes the token IDs into text

      Components:
          processor (`ProcessorMixin`) model (`DiffusionGemmaForBlockDiffusion`) scheduler (`BlockRefinementScheduler`)

      Inputs:
          prompt (`str`, *optional*):
              Prompt text, wrapped in a chat template and tokenized
          messages (`list`, *optional*):
              A raw chat conversation to encode instead of `prompt`, e.g. `[{"role": "user", "content": "Hello"}]` or a
              multi-turn / multimodal conversation.
          image (`Image | ndarray | Tensor | list | list | list`, *optional*):
              Image(s) to pair with `prompt` for multimodal generation. For richer layouts, put the image content
              directly in `messages`.
          add_generation_prompt (`bool`, *optional*, defaults to True):
              Whether to add the generation prompt when applying the chat template.
          gen_length (`int`, *optional*, defaults to 256):
              Number of tokens to generate, rounded up to a multiple of the model's `canvas_length`.
          cache_implementation (`str`, *optional*):
              Set to `"static"` to use a fixed-shape `StaticCache` so the decoder can be compiled.
          eos_token_id (`int`, *optional*):
              EOS token ID for early stopping. Falls back to the processor's tokenizer.
          num_inference_steps (`int`, *optional*, defaults to 48):
              Number of denoising steps per canvas, i.e. the per-canvas budget of model forwards.
          eos_early_stop (`bool`, *optional*, defaults to True):
              Whether to stop generating further canvases once every sequence has emitted EOS.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          temperature (`float`, *optional*, defaults to 0.0):
              Sampling temperature (`0.0` is greedy). Other sampling knobs are scheduler config.
          stability_threshold (`int`, *optional*, defaults to 1):
              Consecutive steps the argmax prediction must be unchanged for a canvas to count as stable. Only used when
              `confidence_threshold` is set.
          confidence_threshold (`float`, *optional*, defaults to 0.005):
              Freeze each example once its prediction is stable and the mean per-token entropy is below this value, and
              leave the refinement loop once every example is frozen. Set to `None` to always run all steps.

      Outputs:
          prompt_ids (`LongTensor`):
              Tokenized prompt of shape `(batch_size, prompt_length)`.
          prompt_attention_mask (`LongTensor`):
              Attention mask for `prompt_ids`.
          multimodal_inputs (`dict`):
              Image tensors the processor produced for the encoder prefill.
          canvas_length (`int`):
              The model's canvas length, i.e. the number of tokens denoised per block.
          num_canvases (`int`):
              Number of canvases to generate.
          past_key_values (`object`):
              The encoder KV cache reused across canvases and denoising steps.
          eos_token_id (`int`):
              The resolved EOS token ID (user-provided or from the processor's tokenizer).
          finished (`Tensor`):
              Per-example flags marking sequences that already emitted EOS.
          predictor_steps (`int`):
              Predictor steps run per canvas.
          corrected_steps (`int`):
              Number of leading predictor steps that also run corrector sweeps.
          corrector_steps (`int`):
              Corrector sweeps run after each of the first `corrected_steps` predictor steps.
          decoder_position_ids (`LongTensor`):
              Position IDs of the canvas tokens, continuing the running sequence.
          decoder_attention_mask_mapping (`object`):
              The decoder attention mask mapping built over the populated cache plus the canvas.
          canvas (`LongTensor`):
              The noisy canvas of shape `(batch_size, canvas_length)` being denoised.
          sequences (`LongTensor`):
              The generated token IDs of shape `(batch_size, generated_length)`.
          texts (`list`):
              The decoded generated text, one string per prompt.
    """

    model_name = "diffusion-gemma"

    block_classes = [
        DiffusionGemmaTextEncoderStep,
        DiffusionGemmaPrepareGenerationStep,
        DiffusionGemmaSetTimestepsStep,
        DiffusionGemmaDenoiseStep,
        DiffusionGemmaDecodeStep,
    ]
    block_names = ["text_encoder", "prepare_generation", "set_timesteps", "denoise", "decode"]

    @property
    def description(self) -> str:
        return (
            "Modular blocks for DiffusionGemma block-diffusion text generation.\n"
            "- `text_encoder` applies the chat template and tokenizes the prompt\n"
            "- `prepare_generation` sizes the canvases, creates the KV cache and resolves EOS\n"
            "- `set_timesteps` splits the forward budget and configures the scheduler\n"
            "- `denoise` generates the text canvas by canvas\n"
            "- `decode` trims at EOS and decodes the token IDs into text"
        )
