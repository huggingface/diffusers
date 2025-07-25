# Diffusers Benchmarks

Welcome to Diffusers Benchmarks. These benchmarks are use to obtain latency and memory information of the most popular models across different scenarios such as:

* Base case i.e., when using `torch.bfloat16` and `torch.nn.functional.scaled_dot_product_attention`.
* Base + `torch.compile()`
* NF4 quantization
* Layerwise upcasting

Instead of full diffusion pipelines, only the forward pass of the respective model classes (such as `FluxTransformer2DModel`) is tested with the real checkpoints (such as `"black-forest-labs/FLUX.1-dev"`). 

The entrypoint to running all the currently available benchmarks is in `run_all.py`. However, one can run the individual benchmarks, too, e.g., `python benchmarking_flux.py`. It should produce a CSV file containing various information about the benchmarks run.

The benchmarks are run on a weekly basis and the CI is defined in [benchmark.yml](../.github/workflows/benchmark.yml).

## Running the benchmarks manually

First set up `torch` and install `diffusers` from the root of the directory:

```py
pip install -e ".[quality,test]"
```

Then make sure the other dependencies are installed:

```sh
cd benchmarks/
pip install -r requirements.txt
```

We need to be authenticated to access some of the checkpoints used during benchmarking:

```sh
hf auth login
```

We use an L40 GPU with 128GB RAM to run the benchmark CI. As such, the benchmarks are configured to run on NVIDIA GPUs. So, make sure you have access to a similar machine (or modify the benchmarking scripts accordingly).

Then you can either launch the entire benchmarking suite by running:

```sh
python run_all.py
```

Or, you can run the individual benchmarks.

## Customizing the benchmarks

We define "scenarios" to cover the most common ways in which these models are used. You can
define a new scenario, modifying an existing benchmark file:

```py
BenchmarkScenario(
    name=f"{CKPT_ID}-bnb-8bit",
    model_cls=FluxTransformer2DModel,
    model_init_kwargs={
        "pretrained_model_name_or_path": CKPT_ID,
        "torch_dtype": torch.bfloat16,
        "subfolder": "transformer",
        "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
    },
    get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
    model_init_fn=model_init_fn,
)
```

You can also configure a new model-level benchmark and add it to the existing suite. To do so, just defining a valid benchmarking file like `benchmarking_flux.py` should be enough.

Happy benchmarking 🧨