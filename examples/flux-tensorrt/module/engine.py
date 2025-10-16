import gc
import subprocess
from collections import OrderedDict, defaultdict

import tensorrt as trt
import torch
from cuda import cudart
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes


trt_to_torch_dtype_dict = {
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.HALF: torch.float16,
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.BF16: torch.bfloat16,
}


class Engine:
    def __init__(self, engine_path: str, stream=None):
        self.engine_path = engine_path
        self._binding_indices = {}
        self.stream = stream
        self.tensors = OrderedDict()

    def load_engine(self, stream=None):
        self.engine_bytes_cpu = bytes_from_path(self.engine_path)
        self.engine = engine_from_bytes(self.engine_bytes_cpu)
        self.context = self.engine.create_execution_context()

        if stream is None:
            self.stream = cudart.cudaStreamCreate()[1]

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
                print(
                    f"[W]: {self.engine_path}: Could not find '{name}' in shape dict {shape_dict}.  Using shape {shape} inferred from the engine."
                )
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            dtype = trt_to_torch_dtype_dict[self.engine.get_tensor_dtype(name)]
            tensor = torch.empty(tuple(shape), dtype=dtype).to(device=device)
            self.tensors[name] = tensor

    def infer(self, feed_dict, stream):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError(f"ERROR: inference of {self.engine_path} failed.")

        return self.tensors

    def unload_engine(self):
        del self.engine
        self.engine = None
        gc.collect()

    def get_shape_dict(self):
        pass

    def check_dims(
        self,
        batch_size,
        image_height,
        image_width,
        compression_factor=8,
        min_batch=1,
        max_batch=16,
        min_latent_shape=16,
        max_latent_shape=1024,
    ):
        assert batch_size >= min_batch and batch_size <= max_batch
        latent_height = image_height // compression_factor
        latent_width = image_width // compression_factor
        assert latent_height >= min_latent_shape and latent_height <= max_latent_shape
        assert latent_width >= min_latent_shape and latent_width <= max_latent_shape
        return (latent_height, latent_width)

    def build(
        self,
        onnx_path,
        strongly_typed=False,
        fp16=False,
        bf16=True,
        tf32=True,
        int8=False,
        fp8=False,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
        native_instancenorm=True,
        verbose=False,
        weight_streaming=False,
        builder_optimization_level=3,
        precision_constraints="none",
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")

        # Handle weight streaming case: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#streaming-weights.
        if weight_streaming:
            strongly_typed, fp16, bf16, int8, fp8 = True, False, False, False, False

        # Base command
        build_command = [f"polygraphy convert {onnx_path} --convert-to trt --output {self.engine_path}"]

        # Precision flags
        build_args = [
            "--fp16" if fp16 else "",
            "--bf16" if bf16 else "",
            "--tf32" if tf32 else "",
            "--fp8" if fp8 else "",
            "--int8" if int8 else "",
            "--strongly-typed" if strongly_typed else "",
        ]

        # Additional arguments
        build_args.extend(
            [
                "--weight-streaming" if weight_streaming else "",
                "--refittable" if enable_refit else "",
                "--tactic-sources" if not enable_all_tactics else "",
                "--onnx-flags native_instancenorm" if native_instancenorm else "",
                f"--builder-optimization-level {builder_optimization_level}",
                f"--precision-constraints {precision_constraints}",
            ]
        )

        # Timing cache
        if timing_cache:
            build_args.extend([f"--load-timing-cache {timing_cache}", f"--save-timing-cache {timing_cache}"])

        # Verbosity setting
        verbosity = "extra_verbose" if verbose else "error"
        build_args.append(f"--verbosity {verbosity}")

        # Output names
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            # build_args.append(f"--trt-outputs {' '.join(update_output_names)}")
            build_args.append(f"--trt-outputs {update_output_names}")

        # Input profiles
        if input_profile:
            profile_args = defaultdict(str)
            for name, dims in input_profile.items():
                assert len(dims) == 3
                profile_args["--trt-min-shapes"] += f"{name}:{str(list(dims[0])).replace(' ', '')} "
                profile_args["--trt-opt-shapes"] += f"{name}:{str(list(dims[1])).replace(' ', '')} "
                profile_args["--trt-max-shapes"] += f"{name}:{str(list(dims[2])).replace(' ', '')} "

            build_args.extend(f"{k} {v}" for k, v in profile_args.items())

        # Filter out empty strings and join command
        build_args = [arg for arg in build_args if arg]
        final_command = " ".join(build_command + build_args)

        # Execute command with improved error handling
        try:
            print(f"Engine build command: {final_command}")
            subprocess.run(final_command, check=True, shell=True)
        except subprocess.CalledProcessError as exc:
            error_msg = f"Failed to build TensorRT engine. Error details:\nCommand: {exc.cmd}\n"
            raise RuntimeError(error_msg) from exc

    def get_minmax_dims(
        self,
        batch_size,
        image_height,
        image_width,
        static_batch,
        static_shape,
        compression_factor=8,
        min_batch=1,
        max_batch=8,
        min_image_shape=256,
        max_image_shape=1344,
        min_latent_shape=16,
        max_latent_shape=1024,
    ):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // compression_factor
        latent_width = image_width // compression_factor
        min_image_height = image_height if static_shape else min_image_shape
        max_image_height = image_height if static_shape else max_image_shape
        min_image_width = image_width if static_shape else min_image_shape
        max_image_width = image_width if static_shape else max_image_shape
        min_latent_height = latent_height if static_shape else min_latent_shape
        max_latent_height = latent_height if static_shape else max_latent_shape
        min_latent_width = latent_width if static_shape else min_latent_shape
        max_latent_width = latent_width if static_shape else max_latent_shape
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )
