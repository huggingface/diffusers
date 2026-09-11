# Reversible component conversions

Every definition is a function of the matching Diffusers component config that returns
`Conversion(mapping=..., rules=...)`. `mapping` contains exact renames; grouped tensor
operations declare both directions in a `Transform`. Definitions never instantiate models
or infer architecture from checkpoint tensors. `get_conversion` supplies constructor defaults.

```python
from diffusers.loaders.conversion import get_conversion

conversion = get_conversion("FluxTransformer2DModel", config)
original = conversion.to_original(diffusers_state)
restored = conversion.to_diffusers(original)
```

Use `scripts/convert_checkpoint.py --list-models` from the repository root to inspect the
registry. The generic CLI reads local tensor files or validated shard indexes and writes a
new safetensors component directory or a PyTorch checkpoint file. See the [usage guide](../../../../docs/source/en/using-diffusers/other-formats.md#bidirectional-component-conversion).

## Contract and format boundaries

- A config must describe the actual checkpoint variant, including layer counts, optional
  projections, and dimensions. The same config is needed when exporting a fresh Diffusers
  checkpoint; a previous original checkpoint is unnecessary.
- Exact input coverage is required. Prefixes, PyTorch wrappers, external assets, auxiliary
  training state, and tensor-parallel rank assembly belong outside a `Conversion`.
  `checkpoint.py` shares known component prefixes, auxiliary-state filtering and declared missing buffers
  between single-file loading and file conversion. `source.py` composes disjoint files or extracts module state;
  pipeline recipes handle tensor-parallel assembly and external runtime preparation. `configs/` supplies reusable
  original-config helpers and presets. None of these layers authors a second tensor mapping.
- Tensor operations preserve dtype/device and do not mutate inputs. Outputs may alias input
  storage. The file writer clones views for safetensors and operates on CPU one rule at a
  time, retaining at most one loaded PyTorch shard plus output shards and grouped tensors.
  Writing a single PyTorch file materializes the converted component state dict.
- `lossless=False` identifies normalization. In the gated LTX2 diffusion decoder profile,
  import folds linear gates into weights and export emits unit gates. Original gate values
  are not recoverable. Weight tensors round-trip exactly for FP32/FP16/BF16; non-unit gate
  folding follows the original importer's FP32 arithmetic.
- `MergeEqual` enforces equality when a source shared parameter was expanded into several
  target parameters. Independently fine-tuned unequal copies cannot be represented by one
  original tensor. `WithConstants` validates fixed buffers before removing/regenerating them.
- LoRA takes an explicit list of fully qualified `modules` in its config; underscore-joined
  Kohya keys are not parsed heuristically. `include_alpha=True` retains alpha tensors as
  `<module>.alpha`, separate from PEFT factors; callers must pass network alphas when loading
  adapters. DoRA is supported for Kohya/PEFT layouts. Adapter merging is not inverted.
- Some original formats already use Diffusers/Transformers names (for example Qwen Image,
  Motif Video, T5, and Qwen3). Their identity keys are explicitly enumerated from config;
  unknown tensors are still rejected. Transformers safetensors may omit tied T5/UMT5 or
  configured Qwen3 embedding copies; file conversion reconstructs only those declared ties.
- RAE conversion expects the assembled encoder, decoder, statistics, and processor state
  represented by its definition; a decoder-only training file does not contain all assets.
  ConsistencyDecoderVAE supports both the original Python module graph and `consistency_decoder_jit`
  archive keys. Use a TorchScript source manifest or the `consistency_decoder` pipeline recipe for archive extraction.
- CLAP conversion describes the complete component state, including persistent RoBERTa
  buffers. Shared source preparation supplies declared position/token-type buffers missing from historical checkpoints. Tensor-parallel MiniMax H3 uses `minimax_h3_shards` after rank
  assembly; the generic reader handles storage shards, not tensor-parallel concatenation.
- `Blip2QFormerModel` names the BLIP Diffusion composite in Diffusers' deprecated pipeline,
  including its vision encoder and projection. It does not refer to Transformers' bare
  Q-Former class with the same name. `ContextCLIPTextModel` includes its persistent position IDs.
- Output `conversion_config.json` records a Diffusers config, not original runtime settings.
  Source formats inferred during import (such as Cosmos 1 or gated LTX2 decoders) are persisted as `original_format`
  in the converted component config and single-file loaded model config, so a later export selects the same layout.
  Tokenizers, schedulers, training loss modules, optimizer/EMA history, inference-irrelevant
  original buffers, and full pipeline packaging are not reconstructed by tensor rules.
  The shared PyTorch exporter accepts nested output wrappers and regenerates fixed CogVideoX positional embeddings.
  Packed quantization, ONNX/TensorRT graph export and LoRA merging are separate operations.

## Verifying a conversion

For layout-only changes, compare original-importer and shared-converter outputs in the same target layout, including
key names, shapes, dtypes and exact tensor values. A SHA-256 digest of sorted tensor metadata and contiguous raw bytes
can verify the same equality without depending on serialization order or file metadata. Hashing the complete input and
output files is not meaningful when layouts, sharding or wrappers differ.

Test both directions and include source-shaped fixtures, bundle selection, auxiliary state and format persistence.
A round trip through one definition alone cannot detect a consistently incorrect rename or permutation. Full-size
GPU inference is not required for a pure layout refactor when independent tensor parity is established. For numerical
normalizations such as gate folding, compare the specified arithmetic and use tolerances where rounding is expected.

## Registered components

An empty format cell means the component has one config-driven canonical layout. For rows
with named variants, omit `original_format` to use the definition's documented default or set
it explicitly. Architecture options within a variant still come from the component config.

| Component class | Definition | Named `original_format` variants |
| --- | --- | --- |
| `AceStepAudioTokenDetokenizer` | [ace_step_detokenizer.py](ace_step_detokenizer.py) |  |
| `AceStepAudioTokenizer` | [ace_step_tokenizer.py](ace_step_tokenizer.py) |  |
| `AceStepConditionEncoder` | [ace_step_conditioner.py](ace_step_conditioner.py) |  |
| `AceStepTransformer1DModel` | [ace_step.py](ace_step.py) |  |
| `AnimaTextConditioner` | [anima_conditioner.py](anima_conditioner.py) |  |
| `AnyFlowFARTransformer3DModel` | [anyflow_far.py](anyflow_far.py) |  |
| `AnyFlowTransformer3DModel` | [anyflow.py](anyflow.py) |  |
| `AsymmetricAutoencoderKL` | [asymmetric_vae.py](asymmetric_vae.py) |  |
| `AudioLDM2ProjectionModel` | [audioldm2_projection.py](audioldm2_projection.py) |  |
| `AudioLDM2UNet2DConditionModel` | [audioldm2_unet.py](audioldm2_unet.py) |  |
| `AuraFlowTransformer2DModel` | [auraflow.py](auraflow.py) |  |
| `AutoencoderDC` | [autoencoder_dc.py](autoencoder_dc.py) |  |
| `AutoencoderKLCogVideoX` | [cogvideox.py](cogvideox.py) |  |
| `AutoencoderKLCosmos` | [cosmos_vae.py](cosmos_vae.py) |  |
| `AutoencoderKLFlux2` | [flux2_vae.py](flux2_vae.py) |  |
| `AutoencoderKLHunyuanImageRefiner` | [hunyuan_image_refiner_vae.py](hunyuan_image_refiner_vae.py) |  |
| `AutoencoderKLHunyuanImage` | [hunyuan_image_vae.py](hunyuan_image_vae.py) | `hunyuan_image_vae` (5D convolutions), `hunyuan_image_vae_2d` (pre-normalized 4D) |
| `AutoencoderKLHunyuanVideo15` | [hunyuan_video15_vae.py](hunyuan_video15_vae.py) |  |
| `AutoencoderKLHunyuanVideo` | [hunyuan_video_vae.py](hunyuan_video_vae.py) |  |
| `AutoencoderKLLTX2Audio` | [ltx2_audio_vae.py](ltx2_audio_vae.py) |  |
| `AutoencoderKLLTX2Video` | [ltx2_vae.py](ltx2_vae.py) |  |
| `AutoencoderKLLTXVideo` | [ltx_vae.py](ltx_vae.py) |  |
| `AutoencoderKLMiniMaxH3Audio` | [minimax_h3_audio_vae.py](minimax_h3_audio_vae.py) |  |
| `AutoencoderKLMiniMaxH3` | [minimax_h3_vae.py](minimax_h3_vae.py) |  |
| `AutoencoderKLMochi` | [mochi_vae.py](mochi_vae.py) |  |
| `AutoencoderKLQwenImage` | [qwen_image_vae.py](qwen_image_vae.py) |  |
| `AutoencoderKLTemporalDecoder` | [svd_vae.py](svd_vae.py) | `svd`, `temporal_vae` |
| `AutoencoderKLWan` | [wan_vae.py](wan_vae.py) |  |
| `AutoencoderKL` | [ldm_vae.py](ldm_vae.py) |  |
| `AutoencoderOobleck` | [oobleck.py](oobleck.py) |  |
| `AutoencoderRAE` | [rae.py](rae.py) |  |
| `AutoencoderSAME` | [same.py](same.py) |  |
| `AutoencoderTiny` | [tiny_vae.py](tiny_vae.py) |  |
| `Blip2QFormerModel` | [blip_qformer.py](blip_qformer.py) |  |
| `CCProjection` | [zero123_projection.py](zero123_projection.py) |  |
| `CLIPTextModelWithProjection` | [clip.py](clip.py) | `clip`, `openclip` |
| `CLIPTextModel` | [clip.py](clip.py) | `clip`, `openclip` |
| `CLIPVisionModelWithProjection` | [clip_vision.py](clip_vision.py) | `clip`, `openclip` |
| `CLIPVisionModel` | [clip_vision.py](clip_vision.py) | `clip`, `openclip` |
| `ChromaTransformer2DModel` | [chroma.py](chroma.py) |  |
| `ChronoEditTransformer3DModel` | [chronoedit.py](chronoedit.py) |  |
| `ClapAudioModelWithProjection` | [clap_audio.py](clap_audio.py) |  |
| `ClapAudioModel` | [clap_audio.py](clap_audio.py) |  |
| `ClapModel` | [clap_audio.py](clap_audio.py) |  |
| `ClapTextModelWithProjection` | [clap_text.py](clap_text.py) |  |
| `CogVideoXTransformer3DModel` | [cogvideox.py](cogvideox.py) |  |
| `CogView3PlusTransformer2DModel` | [cogview3plus.py](cogview3plus.py) |  |
| `CogView4Transformer2DModel` | [cogview4.py](cogview4.py) | `cogview4`, `megatron` |
| `ConsistencyDecoderVAE` | [consistency_decoder.py](consistency_decoder.py) | `consistency_decoder_jit` |
| `ContextCLIPTextModel` | [clip.py](clip.py) | `clip`, `openclip` |
| `ControlNetModel` | [controlnet.py](controlnet.py) |  |
| `CosmosControlNetModel` | [cosmos_controlnet.py](cosmos_controlnet.py) |  |
| `CosmosTransformer3DModel` | [cosmos.py](cosmos.py) | `cosmos1`, `cosmos2` |
| `DiTTransformer2DModel` | [dit.py](dit.py) |  |
| `ErnieImageTransformer2DModel` | [ernie_image.py](ernie_image.py) |  |
| `Flux2Transformer2DModel` | [flux2.py](flux2.py) |  |
| `FluxIPAdapter` | [flux_ip_adapter.py](flux_ip_adapter.py) |  |
| `FluxTransformer2DModel` | [flux.py](flux.py) |  |
| `HiDreamImageTransformer2DModel` | [hidream.py](hidream.py) |  |
| `HunyuanDiT2DControlNetModel` | [hunyuan_dit_controlnet.py](hunyuan_dit_controlnet.py) |  |
| `HunyuanDiT2DModel` | [hunyuan_dit.py](hunyuan_dit.py) |  |
| `HunyuanImageTransformer2DModel` | [hunyuan_image.py](hunyuan_image.py) | `hunyuan_image_fused`, `hunyuan_image_split` |
| `HunyuanVideo15Transformer3DModel` | [hunyuan_video15.py](hunyuan_video15.py) |  |
| `HunyuanVideoTransformer3DModel` | [hunyuan_video.py](hunyuan_video.py) |  |
| `I2VGenXLUNet` | [i2vgen_xl.py](i2vgen_xl.py) |  |
| `IFSafetyChecker` | [pipeline_components.py](pipeline_components.py) |  |
| `JoyImageEditPlusTransformer3DModel` | [joy_image.py](joy_image.py) |  |
| `JoyImageEditTransformer3DModel` | [joy_image.py](joy_image.py) |  |
| `Kandinsky3UNet` | [kandinsky3.py](kandinsky3.py) |  |
| `LDMBertModel` | [pipeline_components.py](pipeline_components.py) |  |
| `LTX2DurationHead` | [ltx2_duration.py](ltx2_duration.py) |  |
| `LTX2LatentUpsamplerModel` | [ltx2_upsampler.py](ltx2_upsampler.py) |  |
| `LTX2TextConnectors` | [ltx2_connectors.py](ltx2_connectors.py) |  |
| `LTX2VideoDiffusionDecoderModel` | [ltx2_diffusion_decoder.py](ltx2_diffusion_decoder.py) | `ltx2_diffusion_decoder`, `ltx2_diffusion_decoder_gated` |
| `LTX2VideoTransformer3DModel` | [ltx2.py](ltx2.py) |  |
| `LTX2VocoderWithBWE` | [ltx2_vocoder.py](ltx2_vocoder.py) |  |
| `LTX2Vocoder` | [ltx2_vocoder.py](ltx2_vocoder.py) |  |
| `LTXLatentUpsamplerModel` | [ltx_upsampler.py](ltx_upsampler.py) |  |
| `LTXVideoTransformer3DModel` | [ltx.py](ltx.py) |  |
| `LearnedClassifierFreeSamplingEmbeddings` | [pipeline_components.py](pipeline_components.py) |  |
| `LoRA` | [lora.py](lora.py) | `kohya`, `diffusers`, `diffusers_old`, `peft`, `animatediff` |
| `LongCatAudioDiTTransformer` | [longcat_audio.py](longcat_audio.py) |  |
| `LongCatAudioDiTVae` | [longcat_audio_vae.py](longcat_audio_vae.py) |  |
| `Lumina2Transformer2DModel` | [lumina2.py](lumina2.py) |  |
| `LuminaNextDiT2DModel` | [lumina.py](lumina.py) |  |
| `MiniMaxH3Transformer3DModel` | [minimax_h3.py](minimax_h3.py) | `minimax_h3`, `minimax_h3_shards` |
| `MiniMaxMusic3ConditionEncoder` | [minimax_music3_conditioner.py](minimax_music3_conditioner.py) |  |
| `MiniMaxMusic3RVQDepthDecoder` | [minimax_music3_rvq.py](minimax_music3_rvq.py) |  |
| `MiniMaxMusic3Transformer1DModel` | [minimax_music3.py](minimax_music3.py) |  |
| `MiniMaxMusic3Vocoder` | [minimax_music3_vocoder.py](minimax_music3_vocoder.py) |  |
| `MochiTransformer3DModel` | [mochi.py](mochi.py) |  |
| `MotifVideoTransformer3DModel` | [motif_video.py](motif_video.py) |  |
| `MotionAdapter` | [animatediff.py](animatediff.py) |  |
| `OmniGenTransformer2DModel` | [omnigen.py](omnigen.py) |  |
| `OvisImageTransformer2DModel` | [ovis_image.py](ovis_image.py) |  |
| `PRXTransformer2DModel` | [prx.py](prx.py) | `prx`, `prx_weight_norm` |
| `PaellaVQModel` | [paella.py](paella.py) |  |
| `PaintByExampleImageEncoder` | [pipeline_components.py](pipeline_components.py) |  |
| `PixArtTransformer2DModel` | [pixart.py](pixart.py) |  |
| `PriorTransformer` | [prior.py](prior.py) | `shap_e`, `unclip`, `kandinsky` |
| `Qwen3ForCausalLM` | [qwen3.py](qwen3.py) |  |
| `Qwen3Model` | [qwen3.py](qwen3.py) |  |
| `QwenImageTransformer2DModel` | [qwen_image.py](qwen_image.py) |  |
| `SD3ControlNetModel` | [sd3_controlnet.py](sd3_controlnet.py) |  |
| `SD3Transformer2DModel` | [sd3.py](sd3.py) |  |
| `SanaControlNetModel` | [sana_controlnet.py](sana_controlnet.py) |  |
| `SanaTransformer2DModel` | [sana.py](sana.py) |  |
| `SanaVideoTransformer3DModel` | [sana_video.py](sana_video.py) |  |
| `ShapERenderer` | [shap_e_renderer.py](shap_e_renderer.py) |  |
| `SkyReelsV2Transformer3DModel` | [skyreels_v2.py](skyreels_v2.py) |  |
| `SparseControlNetModel` | [sparse_controlnet.py](sparse_controlnet.py) |  |
| `SpectrogramContEncoder` | [spectrogram_continuous.py](spectrogram_continuous.py) |  |
| `SpectrogramNotesEncoder` | [spectrogram_notes.py](spectrogram_notes.py) |  |
| `SpeechT5HifiGan` | [hifigan.py](hifigan.py) |  |
| `StableAudio3DiTModel` | [stable_audio3.py](stable_audio3.py) |  |
| `StableAudio3DurationEmbedder` | [stable_audio3_duration.py](stable_audio3_duration.py) |  |
| `StableAudioDiTModel` | [stable_audio.py](stable_audio.py) |  |
| `StableAudioProjectionModel` | [stable_audio_projection.py](stable_audio_projection.py) |  |
| `StableCascadeUNet` | [stable_cascade.py](stable_cascade.py) |  |
| `T2IAdapter` | [t2i_adapter.py](t2i_adapter.py) |  |
| `T5EncoderModel` | [t5.py](t5.py) |  |
| `T5FilmDecoder` | [t5_film.py](t5_film.py) |  |
| `Transformer2DModel` | [vq_diffusion.py](vq_diffusion.py) |  |
| `UMT5EncoderModel` | [umt5.py](umt5.py) |  |
| `UNet1DModel` | [unet_1d.py](unet_1d.py) | `diffuser_rl`, `diffuser_rl_legacy` |
| `UNet2DConditionModel` | [ldm_unet.py](ldm_unet.py) | `ldm`, `versatile_image` |
| `UNet2DModel` | [unet_2d.py](unet_2d.py) | `ddpm`, `ldm`, `consistency`, `ncsnpp` |
| `UNet3DConditionModel` | [unet_3d.py](unet_3d.py) |  |
| `UNetFlatConditionModel` | [versatile_text_unet.py](versatile_text_unet.py) |  |
| `UNetSpatioTemporalConditionModel` | [svd.py](svd.py) |  |
| `UVit2DModel` | [uvit.py](uvit.py) |  |
| `UnCLIPTextProjModel` | [unclip_text_projection.py](unclip_text_projection.py) |  |
| `UniDiffuserModel` | [unidiffuser.py](unidiffuser.py) |  |
| `UniDiffuserTextDecoder` | [unidiffuser_text.py](unidiffuser_text.py) |  |
| `VQModel` | [vq_model.py](vq_model.py) |  |
| `WanAnimate2Transformer3DModel` | [wan_animate2.py](wan_animate2.py) |  |
| `WanAnimateTransformer3DModel` | [wan_animate.py](wan_animate.py) |  |
| `WanTransformer3DModel` | [wan.py](wan.py) |  |
| `WanVACETransformer3DModel` | [wan_vace.py](wan_vace.py) |  |
| `WuerstchenDiffNeXt` | [wuerstchen_decoder.py](wuerstchen_decoder.py) |  |
| `WuerstchenPrior` | [wuerstchen_prior.py](wuerstchen_prior.py) |  |
| `ZImageControlNetModel` | [z_image_controlnet.py](z_image_controlnet.py) |  |
| `ZImageTransformer2DModel` | [z_image.py](z_image.py) |  |
