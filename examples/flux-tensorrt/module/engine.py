import gc
from cuda import cudart
import torch
import tensorrt as trt

from collections import OrderedDict
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
    def __init__(self, engine_path: str, stream = None):
        self.engine_path = engine_path
        self._binding_indices = {}
        self.stream = stream
        self.tensors = OrderedDict()

    def load_engine(self,stream = None):
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

    def check_dims(self, batch_size, image_height, image_width, compression_factor = 8, min_batch = 1, max_batch = 16, min_latent_shape = 16, max_latent_shape = 1024):
        assert batch_size >= min_batch and batch_size <= max_batch
        latent_height = image_height // compression_factor
        latent_width = image_width // compression_factor
        assert latent_height >= min_latent_shape and latent_height <= max_latent_shape
        assert latent_width >= min_latent_shape and latent_width <= max_latent_shape
        return (latent_height, latent_width)


    

