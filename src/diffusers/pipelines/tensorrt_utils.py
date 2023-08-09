import time

import numpy as np

# from polygraphy import cuda
import pycuda.driver as cuda
import tensorrt as trt


def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


class TensorRTModel:
    def __init__(
        self,
        trt_engine_path,
        **kwargs,
    ):
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        trt_runtime = trt.Runtime(TRT_LOGGER)
        engine = load_engine(trt_runtime, trt_engine_path)
        context = engine.create_execution_context()

        # allocates memory for network inputs/outputs on both CPU and GPU
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        input_names = []
        output_names = []

        for binding in engine:
            datatype = engine.get_binding_dtype(binding)
            if datatype == trt.DataType.HALF:
                dtype = np.float16
            else:
                dtype = np.float32

            shape = tuple(engine.get_binding_shape(binding))
            host_mem = cuda.pagelocked_empty(shape, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))

            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                input_names.append(binding)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
                output_names.append(binding)

        self.stream = stream
        self.context = context
        self.engine = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

        self.input_names = input_names
        self.output_names = output_names

        self.dtype = dtype

    def __call__(self, **kwargs):
        context = self.context
        stream = self.stream
        bindings = self.bindings

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs

        time.time()
        for idx, input_name in enumerate(self.input_names):
            _input = np.array(kwargs[input_name], dtype=self.dtype)
            np.copyto(host_inputs[idx], _input)
            # transfer input data to the GPU
            cuda.memcpy_htod_async(cuda_inputs[idx], host_inputs[idx], stream)

        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        result = {}
        for idx, output_name in enumerate(self.output_names):
            # transfer predictions back from the GPU
            cuda.memcpy_dtoh_async(host_outputs[idx], cuda_outputs[idx], stream)
            result[output_name] = host_outputs[idx]

        stream.synchronize()

        return result
