import os
import sys

import torch

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    ControlNetModel,
    LCMScheduler,
    StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLAdapterPipeline,
    StableDiffusionXLControlNetPipeline,
    T2IAdapter,
    WuerstchenCombinedPipeline,
    AutoencoderKL,
)
from diffusers.utils import load_image


sys.path.append(".")

from utils import (  # noqa: E402
    BASE_PATH,
    PROMPT,
    BenchmarkInfo,
    benchmark_fn,
    bytes_to_giga_bytes,
    flush,
    generate_csv_dict,
    generate_csv_dict_model,
    write_to_csv,
    write_list_to_csv,
)


RESOLUTION_MAPPING = {
    "Lykon/DreamShaper": (512, 512),
    "lllyasviel/sd-controlnet-canny": (512, 512),
    "diffusers/controlnet-canny-sdxl-1.0": (1024, 1024),
    "TencentARC/t2iadapter_canny_sd14v1": (512, 512),
    "TencentARC/t2i-adapter-canny-sdxl-1.0": (1024, 1024),
    "stabilityai/stable-diffusion-2-1": (768, 768),
    "stabilityai/stable-diffusion-xl-base-1.0": (1024, 1024),
    "stabilityai/stable-diffusion-xl-refiner-1.0": (1024, 1024),
    "stabilityai/sdxl-turbo": (512, 512),
}


class BaseBenchmak:
    pipeline_class = None

    def __init__(self, args):
        super().__init__()

    def run_inference(self, args):
        raise NotImplementedError

    def benchmark(self, args):
        raise NotImplementedError

    def get_result_filepath(self, args):
        pipeline_class_name = str(self.pipe.__class__.__name__)
        name = (
            args.ckpt.replace("/", "_")
            + "_"
            + pipeline_class_name
            + f"-bs@{args.batch_size}-steps@{args.num_inference_steps}-mco@{args.model_cpu_offload}-compile@{args.run_compile}.csv"
        )
        filepath = os.path.join(BASE_PATH, name)
        return filepath


class TextToImageBenchmark(BaseBenchmak):
    pipeline_class = AutoPipelineForText2Image

    def __init__(self, args):
        pipe = self.pipeline_class.from_pretrained(args.ckpt, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        if args.run_compile:
            if not isinstance(pipe, WuerstchenCombinedPipeline):
                pipe.unet.to(memory_format=torch.channels_last)
                print("Run torch compile")
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

                if hasattr(pipe, "movq") and getattr(pipe, "movq", None) is not None:
                    pipe.movq.to(memory_format=torch.channels_last)
                    pipe.movq = torch.compile(pipe.movq, mode="reduce-overhead", fullgraph=True)
            else:
                print("Run torch compile")
                pipe.decoder = torch.compile(pipe.decoder, mode="reduce-overhead", fullgraph=True)
                pipe.vqgan = torch.compile(pipe.vqgan, mode="reduce-overhead", fullgraph=True)

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
        )

    def benchmark(self, args):
        flush()

        print(f"[INFO] {self.pipe.__class__.__name__}: Running benchmark with: {vars(args)}\n")

        time = benchmark_fn(self.run_inference, self.pipe, args)  # in seconds.
        memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.
        benchmark_info = BenchmarkInfo(time=time, memory=memory)

        pipeline_class_name = str(self.pipe.__class__.__name__)
        flush()
        csv_dict = generate_csv_dict(
            pipeline_cls=pipeline_class_name, ckpt=args.ckpt, args=args, benchmark_info=benchmark_info
        )
        filepath = self.get_result_filepath(args)
        write_to_csv(filepath, csv_dict)
        print(f"Logs written to: {filepath}")
        flush()


class TurboTextToImageBenchmark(TextToImageBenchmark):
    def __init__(self, args):
        super().__init__(args)

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
            guidance_scale=0.0,
        )


class LCMLoRATextToImageBenchmark(TextToImageBenchmark):
    lora_id = "latent-consistency/lcm-lora-sdxl"

    def __init__(self, args):
        super().__init__(args)
        self.pipe.load_lora_weights(self.lora_id)
        self.pipe.fuse_lora()
        self.pipe.unload_lora_weights()
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

    def get_result_filepath(self, args):
        pipeline_class_name = str(self.pipe.__class__.__name__)
        name = (
            self.lora_id.replace("/", "_")
            + "_"
            + pipeline_class_name
            + f"-bs@{args.batch_size}-steps@{args.num_inference_steps}-mco@{args.model_cpu_offload}-compile@{args.run_compile}.csv"
        )
        filepath = os.path.join(BASE_PATH, name)
        return filepath

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
            guidance_scale=1.0,
        )

    def benchmark(self, args):
        flush()

        print(f"[INFO] {self.pipe.__class__.__name__}: Running benchmark with: {vars(args)}\n")

        time = benchmark_fn(self.run_inference, self.pipe, args)  # in seconds.
        memory = bytes_to_giga_bytes(torch.cuda.reset_peak_memory_stats())  # in GBs.
        benchmark_info = BenchmarkInfo(time=time, memory=memory)

        pipeline_class_name = str(self.pipe.__class__.__name__)
        flush()
        csv_dict = generate_csv_dict(
            pipeline_cls=pipeline_class_name, ckpt=self.lora_id, args=args, benchmark_info=benchmark_info
        )
        filepath = self.get_result_filepath(args)
        write_to_csv(filepath, csv_dict)
        print(f"Logs written to: {filepath}")
        flush()


class ImageToImageBenchmark(TextToImageBenchmark):
    pipeline_class = AutoPipelineForImage2Image
    url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/1665_Girl_with_a_Pearl_Earring.jpg"
    image = load_image(url).convert("RGB")

    def __init__(self, args):
        super().__init__(args)
        self.image = self.image.resize(RESOLUTION_MAPPING[args.ckpt])

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            image=self.image,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
        )


class TurboImageToImageBenchmark(ImageToImageBenchmark):
    def __init__(self, args):
        super().__init__(args)

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            image=self.image,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
            guidance_scale=0.0,
            strength=0.5,
        )


class InpaintingBenchmark(ImageToImageBenchmark):
    pipeline_class = AutoPipelineForInpainting
    mask_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/overture-creations-5sI6fQgYIuo_mask.png"
    mask = load_image(mask_url).convert("RGB")

    def __init__(self, args):
        super().__init__(args)
        self.image = self.image.resize(RESOLUTION_MAPPING[args.ckpt])
        self.mask = self.mask.resize(RESOLUTION_MAPPING[args.ckpt])

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            image=self.image,
            mask_image=self.mask,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
        )


class IPAdapterTextToImageBenchmark(TextToImageBenchmark):
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png"
    image = load_image(url)

    def __init__(self, args):
        pipe = self.pipeline_class.from_pretrained(args.ckpt, torch_dtype=torch.float16).to("cuda")
        pipe.load_ip_adapter(
            args.ip_adapter_id[0],
            subfolder="models" if "sdxl" not in args.ip_adapter_id[1] else "sdxl_models",
            weight_name=args.ip_adapter_id[1],
        )

        if args.run_compile:
            pipe.unet.to(memory_format=torch.channels_last)
            print("Run torch compile")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            ip_adapter_image=self.image,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
        )


class ControlNetBenchmark(TextToImageBenchmark):
    pipeline_class = StableDiffusionControlNetPipeline
    aux_network_class = ControlNetModel
    root_ckpt = "Lykon/DreamShaper"

    url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/canny_image_condition.png"
    image = load_image(url).convert("RGB")

    def __init__(self, args):
        aux_network = self.aux_network_class.from_pretrained(args.ckpt, torch_dtype=torch.float16)
        pipe = self.pipeline_class.from_pretrained(self.root_ckpt, controlnet=aux_network, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

        if args.run_compile:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.controlnet.to(memory_format=torch.channels_last)

            print("Run torch compile")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead", fullgraph=True)

        self.image = self.image.resize(RESOLUTION_MAPPING[args.ckpt])

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            image=self.image,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
        )


class ControlNetSDXLBenchmark(ControlNetBenchmark):
    pipeline_class = StableDiffusionXLControlNetPipeline
    root_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, args):
        super().__init__(args)


class T2IAdapterBenchmark(ControlNetBenchmark):
    pipeline_class = StableDiffusionAdapterPipeline
    aux_network_class = T2IAdapter
    root_ckpt = "Lykon/DreamShaper"

    url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/canny_for_adapter.png"
    image = load_image(url).convert("L")

    def __init__(self, args):
        aux_network = self.aux_network_class.from_pretrained(args.ckpt, torch_dtype=torch.float16)
        pipe = self.pipeline_class.from_pretrained(self.root_ckpt, adapter=aux_network, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

        if args.run_compile:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.adapter.to(memory_format=torch.channels_last)

            print("Run torch compile")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            pipe.adapter = torch.compile(pipe.adapter, mode="reduce-overhead", fullgraph=True)

        self.image = self.image.resize(RESOLUTION_MAPPING[args.ckpt])


class T2IAdapterSDXLBenchmark(T2IAdapterBenchmark):
    pipeline_class = StableDiffusionXLAdapterPipeline
    root_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"

    url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/canny_for_adapter_sdxl.png"
    image = load_image(url)

    def __init__(self, args):
        super().__init__(args)


class BaseBenchmarkTestCase:
    model_class = None
    pretrained_model_name_or_path = None
    model_class_name = None

    def __init__(self):
        super().__init__()

    def get_result_filepath(self, suffix):
        name = (
            self.model_class_name
            + "_"
            + self.pretrained_model_name_or_path.replace("/", "_")
            + "_"
            + f"{suffix}.csv"
        )
        filepath = os.path.join(BASE_PATH, name)
        return filepath


class AutoencoderKLBenchmark(BaseBenchmarkTestCase):
    model_class = AutoencoderKL

    def __init__(self, pretrained_model_name_or_path, dtype, tiling, **kwargs):
        super().__init__()
        self.dtype = getattr(torch, dtype)
        model = self.model_class.from_pretrained(pretrained_model_name_or_path, torch_dtype=self.dtype, **kwargs).eval()
        model = model.to("cuda")
        self.tiling = False
        if tiling:
            model.enable_tiling()
            self.tiling = True
        self.model = model
        self.model_class_name = str(self.model.__class__.__name__)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    @torch.no_grad()
    def run_decode(self, model, tensor):
        _ = model.decode(tensor)

    @torch.no_grad
    def _test_decode(self, **kwargs):
        batch = kwargs.get("batch")
        height = kwargs.get("height")
        width = kwargs.get("width")

        tensor = torch.randn((batch, self.model.config.latent_channels, height, width), dtype=self.dtype, device="cuda")

        try:
            time = benchmark_fn(self.run_decode, self.model, tensor)
            memory = bytes_to_giga_bytes(torch.cuda.max_memory_reserved())
        except torch.OutOfMemoryError:
            time = "OOM"
            memory = "OOM"

        benchmark_info = BenchmarkInfo(time=time, memory=memory)
        csv_dict = generate_csv_dict_model(
            model_cls=self.model_class_name, ckpt=self.pretrained_model_name_or_path, benchmark_info=benchmark_info, **kwargs,
        )
        print(f"{self.model_class_name} decode - shape: {list(tensor.shape)}, time: {time}, memory: {memory}")
        return csv_dict

    def test_decode(self):
        benchmark_infos = []

        batches = (1,)
        # heights = (32, 64, 128, 256,)
        widths = (32, 64, 128, 256,)
        for batch in batches:
            # for height in heights:
                for width in widths:
                    benchmark_info = self._test_decode(batch=batch, height=width, width=width)
                    benchmark_infos.append(benchmark_info)

        suffix = "decode"
        if self.tiling:
            suffix = "tiled_decode"
        filepath = self.get_result_filepath(suffix)
        write_list_to_csv(filepath, benchmark_infos)
