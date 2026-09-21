# Checkpoint conversion and pipeline assembly

[`convert_checkpoint.py`](convert_checkpoint.py) is the single component tensor-conversion command. Every model uses
the same reversible `Conversion(mapping=..., rules=...)` definition in the library, single-file loader, and recipes.
Use the matching Diffusers component configuration; tensor conversion preserves dtypes and does not construct a model.

```bash
python scripts/convert_checkpoint.py --list-models
python scripts/convert_checkpoint.py --direction to-diffusers \
    --input ./original.safetensors --config ./transformer/config.json --output ./converted
python scripts/convert_checkpoint.py --direction to-original --input ./converted --output ./original-export
```

The command accepts local tensor files, component directories, and shard indexes. Use `--input-prefix` to select a
component (for example `model.diffusion_model.`), and repeat `--input-wrapper` for nested PyTorch dictionaries.
Known component-specific auxiliary state is handled by shared source preparation. Unknown component tensors fail
coverage validation. Download Hub inputs before invoking this local file command.

## Configuration and source preparation

Configuration helpers and presets live in [`diffusers.loaders.conversion.configs`](../src/diffusers/loaders/conversion/configs).
List them with `--list-presets`. Select a JSON preset, or call a named helper with JSON keyword arguments. Use
`--preset-component` to select one entry from a variant collection. Supply `--model-class` if the config has no `_class_name`.

```bash
python scripts/convert_checkpoint.py --direction to-diffusers \
    --input ./wan.safetensors --output ./wan-transformer --model-class WanTransformer3DModel \
    --preset wan.get_transformer_config --preset-args '{"model_type":"Wan-T2V-1.3B"}'
```

Use `--input-manifest sources.json` when a component spans several files. Paths are relative to the manifest.
Each source can select an `input_prefix`, add an `output_prefix`, and specify a list of dictionary `wrapper` keys.
Duplicate resulting keys are rejected. For example, a tiny VAE's separate encoder and decoder files use:

```json
{
  "sources": [
    {"path": "encoder.pth", "output_prefix": "encoder."},
    {"path": "decoder.pth", "output_prefix": "decoder."}
  ]
}
```

```bash
python scripts/convert_checkpoint.py --direction to-diffusers \
    --input-manifest sources.json --preset tiny-vae --output ./tiny-vae
```

The default source `format` is `tensors`. `torchscript` extracts an archive's state dict. The explicit `python-model`
format extracts a pickled module's state dict (including wrapped modules, such as legacy Diffuser RL checkpoints).
Use `python-model` only with trusted files: it executes pickle code and requires the original Python model package.
Tensor-parallel rank concatenation and external JAX/TensorFlow/reference-model preparation remain in the relevant
pipeline recipe; the manifest combines disjoint keys, not tensor-parallel slices.

## Containers, pipelines, adapters, and graph export

Safetensors output is a new component directory, optionally sharded with `--max-shard-size`. PyTorch output is a
single file: use `--output-format pytorch`, repeated `--output-wrapper`, and `--output-prefix` for its container.
The original SAT transformer container, including regenerated fixed CogVideoX positional embeddings, uses:

```bash
python scripts/convert_checkpoint.py --direction to-original --input ./cogvideox/transformer \
    --output ./mp_rank_00_model_states.pt --output-format pytorch \
    --output-wrapper module --output-prefix model.diffusion_model.
```

A SAT VAE uses its VAE component directory, `--output-wrapper state_dict`, and no prefix. Original runtime settings
and separately loaded assets still come from the original implementation. `conversion_config.json` records the
Diffusers component configuration for repeat conversion.

[`build_pipeline.py`](build_pipeline.py) runs the preserved assembly recipes. Recipes prepare original containers,
resolve model configurations, gather encoders/tokenizers, and construct pipelines; component tensor layouts come from
the shared conversion package. Existing recipe-specific arguments are retained. Some recipes require their original
runtime dependencies or GPU hardware. See the [recipe dependency guide](recipes/README.md) and the recipe's help
for requirements and the scope of startup checks.

```bash
python scripts/build_pipeline.py --list
python scripts/build_pipeline.py cogvideox --help
```

[`export_pipeline_checkpoint.py`](export_pipeline_checkpoint.py) packages a local SD/SDXL pipeline's shared component
exports into one original checkpoint. [`merge_lora.py`](merge_lora.py) loads and fuses adapters through the pipeline's
public LoRA API. LoRA factor-format conversion itself uses `convert_checkpoint.py` with an explicit module list;
merging changes base weights and is not reversible. Deployment graph commands are named `export_*`.

```bash
python scripts/export_pipeline_checkpoint.py --input ./sdxl --output ./sdxl.safetensors
python scripts/merge_lora.py --model ./base --adapter ./adapter.safetensors --output ./merged
```

## Migration table

For component rows, provide `--config` or the listed preset and `--model-class`. Convert each component separately
when several classes are listed. Pipeline rows retain the original assembly behavior through the named recipe.

| Former command | Replacement |
| --- | --- |
| `convert_ace_step_to_diffusers.py` | `build_pipeline.py ace_step` ([recipe](recipes/ace_step.py)) |
| `convert_amused.py` | `build_pipeline.py amused` ([recipe](recipes/amused.py)) |
| `convert_anima_to_diffusers.py` | `build_pipeline.py anima` ([recipe](recipes/anima.py)) |
| `convert_animatediff_motion_lora_to_diffusers.py` | `convert_checkpoint.py`: LoRA (`original_format=animatediff`) |
| `convert_animatediff_motion_module_to_diffusers.py` | `convert_checkpoint.py`: MotionAdapter |
| `convert_animatediff_sparsectrl_to_diffusers.py` | `convert_checkpoint.py`: SparseControlNetModel |
| `convert_anyflow_to_diffusers.py` | `build_pipeline.py anyflow` ([recipe](recipes/anyflow.py)) |
| `convert_asymmetric_vqgan_to_diffusers.py` | `convert_checkpoint.py`: AsymmetricAutoencoderKL; preset `asymmetric-vae-1.5` or `asymmetric-vae-2` |
| `convert_aura_flow_to_diffusers.py` | `convert_checkpoint.py`: AuraFlowTransformer2DModel |
| `convert_blipdiffusion_to_diffusers.py` | `build_pipeline.py blip_diffusion` ([recipe](recipes/blip_diffusion.py)) |
| `convert_cogvideox_to_diffusers.py` | `build_pipeline.py cogvideox` ([recipe](recipes/cogvideox.py)) |
| `convert_cogview3_to_diffusers.py` | `build_pipeline.py cogview3` ([recipe](recipes/cogview3.py)) |
| `convert_cogview4_to_diffusers.py` | `build_pipeline.py cogview4` ([recipe](recipes/cogview4.py)) |
| `convert_cogview4_to_diffusers_megatron.py` | `build_pipeline.py cogview4_megatron` ([recipe](recipes/cogview4_megatron.py)) |
| `convert_consistency_decoder.py` | `build_pipeline.py consistency_decoder` ([recipe](recipes/consistency_decoder.py)) |
| `convert_consistency_to_diffusers.py` | `build_pipeline.py consistency` ([recipe](recipes/consistency.py)) |
| `convert_cosmos_to_diffusers.py` | `build_pipeline.py cosmos` ([recipe](recipes/cosmos.py)) |
| `convert_dance_diffusion_to_diffusers.py` | `build_pipeline.py dance_diffusion` ([recipe](recipes/dance_diffusion.py)) |
| `convert_dcae_to_diffusers.py` | `convert_checkpoint.py`: AutoencoderDC |
| `convert_ddpm_original_checkpoint_to_diffusers.py` | `build_pipeline.py ddpm` ([recipe](recipes/ddpm.py)) |
| `convert_diffusers_sdxl_lora_to_webui.py` | `convert_checkpoint.py`: LoRA (`original_format=kohya`) |
| `convert_diffusers_to_original_sdxl.py` | `export_pipeline_checkpoint.py` |
| `convert_diffusers_to_original_stable_diffusion.py` | `export_pipeline_checkpoint.py` |
| `convert_dit_to_diffusers.py` | `build_pipeline.py dit` ([recipe](recipes/dit.py)) |
| `convert_flux2_to_diffusers.py` | `build_pipeline.py flux2` ([recipe](recipes/flux2.py)) |
| `convert_flux_to_diffusers.py` | `convert_checkpoint.py`: FluxTransformer2DModel / AutoencoderKL |
| `convert_flux_xlabs_ipadapter_to_diffusers.py` | `build_pipeline.py flux_ip_adapter` ([recipe](recipes/flux_ip_adapter.py)) |
| `convert_gligen_to_diffusers.py` | `build_pipeline.py gligen` ([recipe](recipes/gligen.py)) |
| `convert_hunyuan_image_to_diffusers.py` | `build_pipeline.py hunyuan_image` ([recipe](recipes/hunyuan_image.py)) |
| `convert_hunyuan_video1_5_to_diffusers.py` | `build_pipeline.py hunyuan_video15` ([recipe](recipes/hunyuan_video15.py)) |
| `convert_hunyuan_video_to_diffusers.py` | `build_pipeline.py hunyuan_video` ([recipe](recipes/hunyuan_video.py)) |
| `convert_hunyuandit_controlnet_to_diffusers.py` | `convert_checkpoint.py`: HunyuanDiT2DControlNetModel |
| `convert_hunyuandit_to_diffusers.py` | `build_pipeline.py hunyuan_dit` ([recipe](recipes/hunyuan_dit.py)) |
| `convert_i2vgen_to_diffusers.py` | `build_pipeline.py i2vgen` ([recipe](recipes/i2vgen.py)) |
| `convert_if.py` | `build_pipeline.py deepfloyd_if` ([recipe](recipes/deepfloyd_if.py)) |
| `convert_joyimage_edit_to_diffusers.py` | `build_pipeline.py joyimage` ([recipe](recipes/joyimage.py)) |
| `convert_k_upscaler_to_diffusers.py` | `build_pipeline.py k_upscaler` ([recipe](recipes/k_upscaler.py)) |
| `convert_kakao_brain_unclip_to_diffusers.py` | `build_pipeline.py unclip` ([recipe](recipes/unclip.py)) |
| `convert_kandinsky3_unet.py` | `convert_checkpoint.py`: Kandinsky3UNet |
| `convert_kandinsky_to_diffusers.py` | `build_pipeline.py kandinsky` ([recipe](recipes/kandinsky.py)) |
| `convert_ldm_original_checkpoint_to_diffusers.py` | `build_pipeline.py ldm` ([recipe](recipes/ldm.py)) |
| `convert_longcat_audio_dit_to_diffusers.py` | `build_pipeline.py longcat_audio` ([recipe](recipes/longcat_audio.py)) |
| `convert_lora_safetensor_to_diffusers.py` | `merge_lora.py` |
| `convert_ltx2_to_diffusers.py` | `build_pipeline.py ltx2` ([recipe](recipes/ltx2.py)) |
| `convert_ltx_to_diffusers.py` | `build_pipeline.py ltx` ([recipe](recipes/ltx.py)) |
| `convert_lumina_to_diffusers.py` | `build_pipeline.py lumina` ([recipe](recipes/lumina.py)) |
| `convert_minimax_h3_to_diffusers.py` | `build_pipeline.py minimax_h3` ([recipe](recipes/minimax_h3.py)) |
| `convert_minimax_music3_to_diffusers.py` | `build_pipeline.py minimax_music3` ([recipe](recipes/minimax_music3.py)) |
| `convert_mochi_to_diffusers.py` | `build_pipeline.py mochi` ([recipe](recipes/mochi.py)) |
| `convert_models_diffuser_to_diffusers.py` | `convert_checkpoint.py`: UNet1DModel; preset `diffuser-rl-32`, `diffuser-rl-128`, or `diffuser-rl-value` |
| `convert_ms_text_to_video_to_diffusers.py` | `convert_checkpoint.py`: UNet3DConditionModel; preset `modelscope-text-to-video` |
| `convert_music_spectrogram_to_diffusers.py` | `build_pipeline.py music_spectrogram` ([recipe](recipes/music_spectrogram.py)) |
| `convert_ncsnpp_original_checkpoint_to_diffusers.py` | `build_pipeline.py ncsnpp` ([recipe](recipes/ncsnpp.py)) |
| `convert_omnigen_to_diffusers.py` | `build_pipeline.py omnigen` ([recipe](recipes/omnigen.py)) |
| `convert_original_audioldm2_to_diffusers.py` | `build_pipeline.py audioldm2` ([recipe](recipes/audioldm2.py)) |
| `convert_original_audioldm_to_diffusers.py` | `build_pipeline.py audioldm` ([recipe](recipes/audioldm.py)) |
| `convert_original_controlnet_to_diffusers.py` | `build_pipeline.py controlnet` ([recipe](recipes/controlnet.py)) |
| `convert_original_musicldm_to_diffusers.py` | `build_pipeline.py musicldm` ([recipe](recipes/musicldm.py)) |
| `convert_original_stable_diffusion_to_diffusers.py` | `build_pipeline.py stable_diffusion` ([recipe](recipes/stable_diffusion.py)) |
| `convert_original_t2i_adapter.py` | `convert_checkpoint.py`: T2IAdapter |
| `convert_ovis_image_to_diffusers.py` | `convert_checkpoint.py`: OvisImageTransformer2DModel |
| `convert_pixart_alpha_to_diffusers.py` | `build_pipeline.py pixart_alpha` ([recipe](recipes/pixart_alpha.py)) |
| `convert_pixart_sigma_to_diffusers.py` | `build_pipeline.py pixart_sigma` ([recipe](recipes/pixart_sigma.py)) |
| `convert_prx_to_diffusers.py` | `build_pipeline.py prx` ([recipe](recipes/prx.py)) |
| `convert_rae_to_diffusers.py` | `build_pipeline.py rae` ([recipe](recipes/rae.py)) |
| `convert_sana_controlnet_to_diffusers.py` | `build_pipeline.py sana_controlnet` ([recipe](recipes/sana_controlnet.py)) |
| `convert_sana_to_diffusers.py` | `build_pipeline.py sana` ([recipe](recipes/sana.py)) |
| `convert_sana_video_to_diffusers.py` | `build_pipeline.py sana_video` ([recipe](recipes/sana_video.py)) |
| `convert_sd3_controlnet_to_diffusers.py` | `convert_checkpoint.py`: SD3ControlNetModel |
| `convert_sd3_to_diffusers.py` | `convert_checkpoint.py`: SD3Transformer2DModel / AutoencoderKL |
| `convert_shap_e_to_diffusers.py` | `build_pipeline.py shap_e` ([recipe](recipes/shap_e.py)) |
| `convert_skyreelsv2_to_diffusers.py` | `build_pipeline.py skyreels_v2` ([recipe](recipes/skyreels_v2.py)) |
| `convert_stable_audio.py` | `build_pipeline.py stable_audio` ([recipe](recipes/stable_audio.py)) |
| `convert_stable_audio_3_to_diffusers.py` | `build_pipeline.py stable_audio3` ([recipe](recipes/stable_audio3.py)) |
| `convert_stable_cascade.py` | `build_pipeline.py stable_cascade` ([recipe](recipes/stable_cascade.py)) |
| `convert_stable_cascade_lite.py` | `build_pipeline.py stable_cascade_lite` ([recipe](recipes/stable_cascade_lite.py)) |
| `convert_stable_diffusion_checkpoint_to_onnx.py` | `export_stable_diffusion_checkpoint_to_onnx.py` |
| `convert_stable_diffusion_controlnet_to_onnx.py` | `export_stable_diffusion_controlnet_to_onnx.py` |
| `convert_stable_diffusion_controlnet_to_tensorrt.py` | `export_stable_diffusion_controlnet_to_tensorrt.py` |
| `convert_svd_to_diffusers.py` | `convert_checkpoint.py`: UNetSpatioTemporalConditionModel / AutoencoderKLTemporalDecoder |
| `convert_tiny_autoencoder_to_diffusers.py` | `convert_checkpoint.py`: AutoencoderTiny; preset `tiny-vae` and source manifest |
| `convert_unclip_txt2img_to_image_variation.py` | `build_pipeline.py unclip_image_variation` ([recipe](recipes/unclip_image_variation.py)) |
| `convert_unidiffuser_to_diffusers.py` | `build_pipeline.py unidiffuser` ([recipe](recipes/unidiffuser.py)) |
| `convert_vae_diff_to_onnx.py` | `export_vae_diff_to_onnx.py` |
| `convert_vae_pt_to_diffusers.py` | `convert_checkpoint.py`: AutoencoderKL; preset `sd-vae` |
| `convert_versatile_diffusion_to_diffusers.py` | `build_pipeline.py versatile_diffusion` ([recipe](recipes/versatile_diffusion.py)) |
| `convert_vq_diffusion_to_diffusers.py` | `build_pipeline.py vq_diffusion` ([recipe](recipes/vq_diffusion.py)) |
| `convert_wan_to_diffusers.py` | `build_pipeline.py wan` ([recipe](recipes/wan.py)) |
| `convert_wuerstchen.py` | `build_pipeline.py wuerstchen` ([recipe](recipes/wuerstchen.py)) |
| `convert_zero123_to_diffusers.py` | `build_pipeline.py zero123` ([recipe](recipes/zero123.py)) |
