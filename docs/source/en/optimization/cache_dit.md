**[cache-dit](https://github.com/vipshop/cache-dit)** is a **Unified**, **Flexible**, and **Training-free** cache acceleration framework designed for 🤗 Diffusers, enabling cache acceleration with just **one line** of code. It encompasses a range of key features including **Unified Cache APIs**, **Forward Pattern Matching**, **Automatic Block Adapter**, **Hybrid Forward Pattern**, **DBCache**, **TaylorSeer Calibrator**, and **Cache CFG**.   

Notably, **[cache-dit](https://github.com/vipshop/cache-dit)** now supports nearly **all** of Diffusers' **DiT-based** pipelines, such as Qwen-Image, FLUX.1, Qwen-Image-Lightning, Wan 2.1/2.2, HunyuanImage-2.1, HunyuanVideo, HunyuanDiT, HiDream, AuraFlow, CogView3Plus, CogView4, LTXVideo, CogVideoX/X 1.5, ConsisID, Cosmos, SkyReelsV2, VisualCloze, OmniGen 1/2, Lumina 1/2, PixArt, Chroma, Sana, Allegro, Mochi, SD 3/3.5, Amused, and DiT-XL, with relevant benchmarks available for **Text2Image DrawBench** and **Text2Image Distillation DrawBench**.

## 📖Contents 

<div id="contents"></div>  

- [⚙️Installation](#️installation)
- [🔥Benchmarks](#benchmarks)
- [🔥Supported Pipelines](#supported)
- [🎉Unified Cache APIs](#unified)
- [⚡️Dual Block Cache](#dbcache)
- [🔥TaylorSeer Calibrator](#taylorseer)
- [⚡️Hybrid Cache CFG](#cfg)
- [⚙️Torch Compile](#compile)

## ⚙️Installation  

<div id="installation"></div>

You can install the stable release of `cache-dit` from PyPI:

```bash
pip3 install -U cache-dit
```
Or you can install the latest develop version from GitHub:

```bash
pip3 install git+https://github.com/vipshop/cache-dit.git
```

## 🔥Supported Pipelines  

<div id="supported"></div>

Currently, **cache-dit** library supports almost **Any** Diffusion Transformers (with **Transformer Blocks** that match the specific Input and Output **patterns**). Please check [🎉Examples](https://github.com/vipshop/cache-dit/raw/main/examples/pipeline) for more details. Here are just some of the tested models listed.

```python
>>> import cache_dit
>>> cache_dit.supported_pipelines()
(30, ['Flux*', 'Mochi*', 'CogVideoX*', 'Wan*', 'HunyuanVideo*', 'QwenImage*', 'LTX*', 'Allegro*',
'CogView3Plus*', 'CogView4*', 'Cosmos*', 'EasyAnimate*', 'SkyReelsV2*', 'StableDiffusion3*',
'ConsisID*', 'DiT*', 'Amused*', 'Bria*', 'Lumina*', 'OmniGen*', 'PixArt*', 'Sana*', 'StableAudio*',
'VisualCloze*', 'AuraFlow*', 'Chroma*', 'ShapE*', 'HiDream*', 'HunyuanDiT*', 'HunyuanDiTPAG*'])
```

<details>
<summary> Show all pipelines </summary>  

- [🚀HunyuanImage-2.1](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀Qwen-Image-Lightning](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Qwen-Image-Edit](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀Qwen-Image](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀FLUX.1-dev](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀FLUX.1-Fill-dev](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀FLUX.1-Kontext-dev](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogView4](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Wan2.2-T2V](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀HunyuanVideo](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀HiDream-I1-Full](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀HunyuanDiT](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Wan2.1-T2V](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Wan2.1-FLF2V](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀SkyReelsV2](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀Chroma1-HD](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀CogVideoX1.5](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogView3-Plus](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogVideoX](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀VisualCloze](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀LTXVideo](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀OmniGen](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀Lumina2](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀mochi-1-preview](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀AuraFlow-v0.3](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀PixArt-Alpha](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀PixArt-Sigma](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀NVIDIA Sana](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀SD-3/3.5](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀ConsisID](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Allegro](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Amused](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀DiT-XL](https://github.com/vipshop/cache-dit/raw/main/examples)
- ...

</details>

## 🔥Benchmarks

<div id="benchmarks"></div>

cache-dit will support more mainstream Cache acceleration algorithms in the future. More benchmarks will be released, please stay tuned for update. Here, only the results of some precision and performance benchmarks are presented. The test dataset is **DrawBench**. For a complete benchmark, please refer to [📚Benchmarks](https://github.com/vipshop/cache-dit/raw/main/bench/).

### 📚Text2Image DrawBench: FLUX.1-dev

Comparisons between different FnBn compute block configurations show that **more compute blocks result in higher precision**. For example, the F8B0_W8MC0 configuration achieves the best Clip Score (33.007) and ImageReward (1.0333). **Device**: NVIDIA L20. **F**: Fn_compute_blocks, **B**: Bn_compute_blocks, 50 steps.


| Config | Clip Score(↑) | ImageReward(↑) | PSNR(↑) | TFLOPs(↓) | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| [**FLUX.1**-dev]: 50 steps | 32.9217 | 1.0412 | INF | 3726.87 | 1.00x |
| F8B0_W4MC0_R0.08 | 32.9871 | 1.0370 | 33.8317 | 2064.81 | 1.80x |
| F8B0_W4MC2_R0.12 | 32.9535 | 1.0185 | 32.7346 | 1935.73 | 1.93x |
| F8B0_W4MC3_R0.12 | 32.9234 | 1.0085 | 32.5385 | 1816.58 | 2.05x |
| F4B0_W4MC3_R0.12 | 32.8981 | 1.0130 | 31.8031 | 1507.83 | 2.47x |
| F4B0_W4MC4_R0.12 | 32.8384 | 1.0065 | 31.5292 | 1400.08 | 2.66x |

The comparison between **cache-dit: DBCache** and algorithms such as Δ-DiT, Chipmunk, FORA, DuCa, TaylorSeer and FoCa is as follows. Now, in the comparison with a speedup ratio less than **3x**, cache-dit achieved the best accuracy. Please check [📚How to Reproduce?](https://github.com/vipshop/cache-dit/raw/main/bench/) for more details.

| Method | TFLOPs(↓) | SpeedUp(↑) | ImageReward(↑) | Clip Score(↑) |
| --- | --- | --- | --- | --- |
| [**FLUX.1**-dev]: 50 steps | 3726.87 | 1.00× | 0.9898 | 32.404 |
| [**FLUX.1**-dev]: 60% steps | 2231.70 | 1.67× | 0.9663 | 32.312 |
| Δ-DiT(N=2) | 2480.01 | 1.50× | 0.9444 | 32.273 |
| Δ-DiT(N=3) | 1686.76 | 2.21× | 0.8721 | 32.102 |
| [**FLUX.1**-dev]: 34% steps | 1264.63 | 3.13× | 0.9453 | 32.114 |
| Chipmunk | 1505.87 | 2.47× | 0.9936 | 32.776 |
| FORA(N=3) | 1320.07 | 2.82× | 0.9776 | 32.266 |
| **[DBCache(F=4,B=0,W=4,MC=4)](https://github.com/vipshop/cache-dit)** | 1400.08 | **2.66×** | **1.0065** | 32.838 |
| **[DBCache+TaylorSeer(F=1,B=0,O=1)](https://github.com/vipshop/cache-dit)** | 1153.05 | **3.23×** | **1.0221** | 32.819 |
| DuCa(N=5) | 978.76 | 3.80× | 0.9955 | 32.241 |
| TaylorSeer(N=4,O=2) | 1042.27 | 3.57× | 0.9857 | 32.413 |
| **[DBCache(F=1,B=0,W=4,MC=6)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | 0.9997 | 32.849 |
| **[DBCache+TaylorSeer(F=1,B=0,O=1)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | **1.0107** | 32.865 |
| **[FoCa(N=5): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 893.54 | **4.16×** | **1.0029** | **32.948** |

<details>
<summary> Show all comparison </summary>  

| Method | TFLOPs(↓) | SpeedUp(↑) | ImageReward(↑) | Clip Score(↑) |
| --- | --- | --- | --- | --- |
| [**FLUX.1**-dev]: 50 steps | 3726.87 | 1.00× | 0.9898 | 32.404 |
| [**FLUX.1**-dev]: 60% steps | 2231.70 | 1.67× | 0.9663 | 32.312 |
| Δ-DiT(N=2) | 2480.01 | 1.50× | 0.9444 | 32.273 |
| Δ-DiT(N=3) | 1686.76 | 2.21× | 0.8721 | 32.102 |
| [**FLUX.1**-dev]: 34% steps | 1264.63 | 3.13× | 0.9453 | 32.114 |
| Chipmunk | 1505.87 | 2.47× | 0.9936 | 32.776 |
| FORA(N=3) | 1320.07 | 2.82× | 0.9776 | 32.266 |
| **[DBCache(F=4,B=0,W=4,MC=4)](https://github.com/vipshop/cache-dit)** | 1400.08 | **2.66×** | **1.0065** | 32.838 |
| DuCa(N=5) | 978.76 | 3.80× | 0.9955 | 32.241 |
| TaylorSeer(N=4,O=2) | 1042.27 | 3.57× | 0.9857 | 32.413 |
| **[DBCache+TaylorSeer(F=1,B=0,O=1)](https://github.com/vipshop/cache-dit)** | 1153.05 | **3.23×** | **1.0221** | 32.819 |
| **[DBCache(F=1,B=0,W=4,MC=6)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | 0.9997 | 32.849 |
| **[DBCache+TaylorSeer(F=1,B=0,O=1)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | **1.0107** | 32.865 |
| **[FoCa(N=5): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 893.54 | **4.16×** | **1.0029** | **32.948** |
| [**FLUX.1**-dev]: 22% steps | 818.29 | 4.55× | 0.8183 | 31.772 |
| FORA(N=4) | 967.91 | 3.84× | 0.9730 | 32.142 |
| ToCa(N=8) | 784.54 | 4.74× | 0.9451 | 31.993 |
| DuCa(N=7) | 760.14 | 4.89× | 0.9757 | 32.066 |
| TeaCache(l=0.8) | 892.35 | 4.17× | 0.8683 | 31.704 |
| **[DBCache(F=4,B=0,W=4,MC=10)](https://github.com/vipshop/cache-dit)** | 816.65 | 4.56x | 0.8245 | 32.191 |
| TaylorSeer(N=5,O=2) | 893.54 | 4.16× | 0.9768 | 32.467 |
| **[FoCa(N=7): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 670.44 | **5.54×** | **0.9891** | **32.920** |
| FORA(N=7) | 670.14 | 5.55× | 0.7418 | 31.519 |
| ToCa(N=12) | 644.70 | 5.77× | 0.7155 | 31.808 |
| DuCa(N=10) | 606.91 | 6.13× | 0.8382 | 31.759 |
| TeaCache(l=1.2) | 669.27 | 5.56× | 0.7394 | 31.704 |
| **[DBCache(F=1,B=0,W=4,MC=10)](https://github.com/vipshop/cache-dit)** | 651.90 | **5.72x** | 0.8796 | **32.318** |
| TaylorSeer(N=7,O=2) | 670.44 | 5.54× | 0.9128 | 32.128 |
| **[FoCa(N=8): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 596.07 | **6.24×** | **0.9502** | **32.706** |

NOTE: Except for DBCache, other performance data are referenced from the paper [FoCa, arxiv.2508.16211](https://arxiv.org/pdf/2508.16211).

</details>

### 📚Text2Image Distillation DrawBench: Qwen-Image-Lightning

Surprisingly, cache-dit: DBCache still works in the extremely few-step distill model. For example,  **Qwen-Image-Lightning w/ 4 steps**, with the F16B16 configuration, the PSNR is 34.8163, the Clip Score is 35.6109, and the ImageReward is 1.2614. It maintained a relatively high precision.

| Config                     |  PSNR(↑)      | Clip Score(↑) | ImageReward(↑) | TFLOPs(↓)   | SpeedUp(↑) |
|----------------------------|-----------|------------|--------------|----------|------------|
| [**Lightning**]: 4 steps   | INF       | 35.5797    | 1.2630       | 274.33   | 1.00x       |
| F24B24_W2MC1_R0.8          | 36.3242   | 35.6224    | 1.2630       | 264.74   | 1.04x       |
| F16B16_W2MC1_R0.8          | 34.8163   | 35.6109    | 1.2614       | 244.25   | 1.12x       |
| F12B12_W2MC1_R0.8          | 33.8953   | 35.6535    | 1.2549       | 234.63   | 1.17x       |
| F8B8_W2MC1_R0.8            | 33.1374   | 35.7284    | 1.2517       | 224.29   | 1.22x       |
| F1B0_W2MC1_R0.8            | 31.8317   | 35.6651    | 1.2397       | 206.90   | 1.33x       |

## 🎉Unified Cache APIs

<div id="unified"></div>  

### 📚Forward Pattern Matching 

Currently, for any **Diffusion** models with **Transformer Blocks** that match the specific **Input/Output patterns**, we can use the **Unified Cache APIs** from **cache-dit**, namely, the `cache_dit.enable_cache(...)` API. The **Unified Cache APIs** are currently in the experimental phase; please stay tuned for updates. The supported patterns are listed as follows:

![](https://github.com/vipshop/cache-dit/raw/main/assets/patterns-v1.png)

### ♥️Cache Acceleration with One-line Code

In most cases, you only need to call **one-line** of code, that is `cache_dit.enable_cache(...)`. After this API is called, you just need to call the pipe as normal. The `pipe` param can be **any** Diffusion Pipeline. Please refer to [Qwen-Image](https://github.com/vipshop/cache-dit/raw/main/examples/pipeline/run_qwen_image.py) as an example. 

```python
import cache_dit
from diffusers import DiffusionPipeline 

# Can be any diffusion pipeline
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")

# One-line code with default cache options.
cache_dit.enable_cache(pipe) 

# Just call the pipe as normal.
output = pipe(...)

# Disable cache and run original pipe.
cache_dit.disable_cache(pipe)
```

### 🔥Automatic Block Adapter

But in some cases, you may have a **modified** Diffusion Pipeline or Transformer that is not located in the diffusers library or not officially supported by **cache-dit** at this time. The **BlockAdapter** can help you solve this problems. Please refer to [🔥Qwen-Image w/ BlockAdapter](https://github.com/vipshop/cache-dit/raw/main/examples/adapter/run_qwen_image_adapter.py) as an example.

```python
from cache_dit import ForwardPattern, BlockAdapter

# Use 🔥BlockAdapter with `auto` mode.
cache_dit.enable_cache(
    BlockAdapter(
        # Any DiffusionPipeline, Qwen-Image, etc.  
        pipe=pipe, auto=True,
        # Check `📚Forward Pattern Matching` documentation and hack the code of
        # of Qwen-Image, you will find that it has satisfied `FORWARD_PATTERN_1`.
        forward_pattern=ForwardPattern.Pattern_1,
    ),   
)

# Or, manually setup transformer configurations.
cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe, # Qwen-Image, etc.
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_1,
    ), 
)
```
For such situations, **BlockAdapter** can help you quickly apply various cache acceleration features to your own Diffusion Pipelines and Transformers. Please check the [📚BlockAdapter.md](https://github.com/vipshop/cache-dit/raw/main/docs/BlockAdapter.md) for more details.

### 📚Hybird Forward Pattern

Sometimes, a Transformer class will contain more than one transformer `blocks`. For example, **FLUX.1** (HiDream, Chroma, etc) contains transformer_blocks and single_transformer_blocks (with different forward patterns). The **BlockAdapter** can also help you solve this problem. Please refer to [📚FLUX.1](https://github.com/vipshop/cache-dit/raw/main/examples/adapter/run_flux_adapter.py) as an example.

```python
# For diffusers <= 0.34.0, FLUX.1 transformer_blocks and 
# single_transformer_blocks have different forward patterns.
cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe, # FLUX.1, etc.
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_3,
        ],
    ),
)
```

Even sometimes you have more complex cases, such as **Wan 2.2 MoE**, which has more than one Transformer (namely `transformer` and `transformer_2`) in its structure. Fortunately, **cache-dit** can also handle this situation very well. Please refer to [📚Wan 2.2 MoE](https://github.com/vipshop/cache-dit/raw/main/examples/pipeline/run_wan_2.2.py) as an example.

```python
from cache_dit import ForwardPattern, BlockAdapter, ParamsModifier, BasicCacheConfig

cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe,
        transformer=[
            pipe.transformer,
            pipe.transformer_2,
        ],
        blocks=[
            pipe.transformer.blocks,
            pipe.transformer_2.blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_2,
            ForwardPattern.Pattern_2,
        ],
        # Setup different cache params for each 'blocks'. You can 
        # pass any specific cache params to ParamModifier, the old 
        # value will be overwrite by the new one.
        params_modifiers=[
            ParamsModifier(
                cache_config=BasicCacheConfig(
                    max_warmup_steps=4,
                    max_cached_steps=8,
                ),
            ),
            ParamsModifier(
                cache_config=BasicCacheConfig(
                    max_warmup_steps=2,
                    max_cached_steps=20,
                ),
            ),
        ],
        has_separate_cfg=True,
    ),
)
```
### 📚Implement Patch Functor

For any PATTERN not in {0...5}, we introduced the simple abstract concept of **Patch Functor**. Users can implement a subclass of Patch Functor to convert an unknown Pattern into a known PATTERN, and for some models, users may also need to fuse the operations within the blocks for loop into block forward. 

![](https://github.com/vipshop/cache-dit/raw/main/assets/patch-functor.png)

Some Patch functors have already been provided in cache-dit: [📚HiDreamPatchFunctor](https://github.com/vipshop/cache-dit/raw/main/src/cache_dit/cache_factory/patch_functors/functor_hidream.py), [📚ChromaPatchFunctor](https://github.com/vipshop/cache-dit/raw/main/src/cache_dit/cache_factory/patch_functors/functor_chroma.py), etc. After implementing Patch Functor, users need to set the `patch_functor` property of **BlockAdapter**.

```python
@BlockAdapterRegistry.register("HiDream")
def hidream_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import HiDreamImageTransformer2DModel
    from cache_dit.cache_factory.patch_functors import HiDreamPatchFunctor

    assert isinstance(pipe.transformer, HiDreamImageTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.double_stream_blocks,
            pipe.transformer.single_stream_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_0,
            ForwardPattern.Pattern_3,
        ],
        # NOTE: Setup your custom patch functor here.
        patch_functor=HiDreamPatchFunctor(),
        **kwargs,
    )
```

### 🤖Cache Acceleration Stats Summary

After finishing each inference of `pipe(...)`, you can call the `cache_dit.summary()` API on pipe to get the details of the **Cache Acceleration Stats** for the current inference. 
```python
stats = cache_dit.summary(pipe)
```

You can set `details` param as `True` to show more details of cache stats. (markdown table format) Sometimes, this may help you analyze what values of the residual diff threshold would be better.

```python
⚡️Cache Steps and Residual Diffs Statistics: QwenImagePipeline

| Cache Steps | Diffs Min | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Max |
|-------------|-----------|-----------|-----------|-----------|-----------|-----------|
| 23          | 0.045     | 0.084     | 0.114     | 0.147     | 0.241     | 0.297     |
```

## ⚡️DBCache: Dual Block Cache  

<div id="dbcache"></div>

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-v1.png)

**DBCache**: **Dual Block Caching** for Diffusion Transformers. Different configurations of compute blocks (**F8B12**, etc.) can be customized in DBCache, enabling a balanced trade-off between performance and precision. Moreover, it can be entirely **training**-**free**. Please check [DBCache.md](https://github.com/vipshop/cache-dit/raw/main/docs/DBCache.md) docs for more design details.

- **Fn**: Specifies that DBCache uses the **first n** Transformer blocks to fit the information at time step t, enabling the calculation of a more stable L1 diff and delivering more accurate information to subsequent blocks.
- **Bn**: Further fuses approximate information in the **last n** Transformer blocks to enhance prediction accuracy. These blocks act as an auto-scaler for approximate hidden states that use residual cache.

```python
import cache_dit
from diffusers import FluxPipeline

pipe_or_adapter = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Default options, F8B0, 8 warmup steps, and unlimited cached 
# steps for good balance between performance and precision
cache_dit.enable_cache(pipe_or_adapter)

# Custom options, F8B8, higher precision
from cache_dit import BasicCacheConfig

cache_dit.enable_cache(
    pipe_or_adapter,
    cache_config=BasicCacheConfig(
        max_warmup_steps=8,  # steps do not cache
        max_cached_steps=-1, # -1 means no limit
        Fn_compute_blocks=8, # Fn, F8, etc.
        Bn_compute_blocks=8, # Bn, B8, etc.
        residual_diff_threshold=0.12,
    ),
)
```  

<div align="center">
  <p align="center">
    DBCache, <b> L20x1 </b>, Steps: 28, "A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|F1B0 (0.08)|F1B0 (0.20)|F8B8 (0.15)|F12B12 (0.20)|F16B16 (0.20)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|15.59s|8.58s|15.41s|15.11s|17.74s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.08_S11.png width=105px> | <img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.2_S19.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F8B8S1_R0.15_S15.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F12B12S4_R0.2_S16.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F16B16S4_R0.2_S13.png width=105px>|

## 🔥TaylorSeer Calibrator

<div id="taylorseer"></div>

We have supported the [TaylorSeers: From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/pdf/2503.06923) algorithm to further improve the precision of DBCache in cases where the cached steps are large, namely, **Hybrid TaylorSeer + DBCache**. At timesteps with significant intervals, the feature similarity in diffusion models decreases substantially, significantly harming the generation quality. 

**TaylorSeer** employs a differential method to approximate the higher-order derivatives of features and predict features in future timesteps with Taylor series expansion. The TaylorSeer implemented in cache-dit supports both hidden states and residual cache types. That is F_pred can be a residual cache or a hidden-state cache.

```python
from cache_dit import BasicCacheConfig, TaylorSeerCalibratorConfig

cache_dit.enable_cache(
    pipe_or_adapter,
    # Basic DBCache w/ FnBn configurations
    cache_config=BasicCacheConfig(
        max_warmup_steps=8,  # steps do not cache
        max_cached_steps=-1, # -1 means no limit
        Fn_compute_blocks=8, # Fn, F8, etc.
        Bn_compute_blocks=8, # Bn, B8, etc.
        residual_diff_threshold=0.12,
    ),
    # Then, you can use the TaylorSeer Calibrator to approximate 
    # the values in cached steps, taylorseer_order default is 1.
    calibrator_config=TaylorSeerCalibratorConfig(
        taylorseer_order=1,
    ),
)
``` 

> [!Important]
> Please note that if you have used TaylorSeer as the calibrator for approximate hidden states, the **Bn** param of DBCache can be set to **0**. In essence, DBCache's Bn is also act as a calibrator, so you can choose either Bn > 0 or TaylorSeer. We recommend using the configuration scheme of **TaylorSeer** + **DBCache FnB0**.

<div align="center">
  <p align="center">
    <b>DBCache F1B0 + TaylorSeer</b>, L20x1, Steps: 28, <br>"A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|F1B0 (0.12)|+TaylorSeer|F1B0 (0.15)|+TaylorSeer|+compile| 
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|12.85s|12.86s|10.27s|10.28s|8.48s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T0ET0_R0.12_S14_T12.85s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T1ET1_R0.12_S14_T12.86s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T0ET0_R0.15_S17_T10.27s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T1ET1_R0.15_S17_T10.28s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C1_DBCACHE_F1B0S1W0T1ET1_R0.15_S17_T8.48s.png width=105px>|

## ⚡️Hybrid Cache CFG

<div id="cfg"></div>

cache-dit supports caching for **CFG (classifier-free guidance)**. For models that fuse CFG and non-CFG into a single forward step, or models that do not include CFG (classifier-free guidance) in the forward step, please set `enable_separate_cfg` param to **False (default, None)**. Otherwise, set it to True. For examples:

```python
from cache_dit import BasicCacheConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=BasicCacheConfig(
        ...,
        # CFG: classifier free guidance or not
        # For model that fused CFG and non-CFG into single forward step,
        # should set enable_separate_cfg as False. For example, set it as True 
        # for Wan 2.1/Qwen-Image and set it as False for FLUX.1, HunyuanVideo, 
        # CogVideoX, Mochi, LTXVideo, Allegro, CogView3Plus, EasyAnimate, SD3, etc.
        enable_separate_cfg=True, # Wan 2.1, Qwen-Image, CogView4, Cosmos, SkyReelsV2, etc.
        # Compute cfg forward first or not, default False, namely, 
        # 0, 2, 4, ..., -> non-CFG step; 1, 3, 5, ... -> CFG step.
        cfg_compute_first=False,
        # Compute separate diff values for CFG and non-CFG step, 
        # default True. If False, we will use the computed diff from 
        # current non-CFG transformer step for current CFG step.
        cfg_diff_compute_separate=True,
    ),
)
```

## ⚙️Torch Compile

<div id="compile"></div>  

By the way, **cache-dit** is designed to work compatibly with **torch.compile.** You can easily use cache-dit with torch.compile to further achieve a better performance. For example:

```python
cache_dit.enable_cache(pipe)

# Compile the Transformer module
pipe.transformer = torch.compile(pipe.transformer)
```
However, users intending to use **cache-dit** for DiT with **dynamic input shapes** should consider increasing the **recompile** **limit** of `torch._dynamo`. Otherwise, the recompile_limit error may be triggered, causing the module to fall back to eager mode. 
```python
torch._dynamo.config.recompile_limit = 96  # default is 8
torch._dynamo.config.accumulated_recompile_limit = 2048  # default is 256
```

Please check [perf.py](https://github.com/vipshop/cache-dit/raw/main/bench/perf.py) for more details.
