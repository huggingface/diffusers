from module.transformers import FluxTransformerModel
from module.clip import CLIPModel
from module.t5xxl import T5XXLModel
from module.vae import VAEModel

models_config = {
    "transformer": (FluxTransformerModel, "onnx/transformer.opt/bf16/model.onnx"),
    "clip":        (CLIPModel,          "onnx/clip.opt/model.onnx"),
    "t5":          (T5XXLModel,         "onnx/t5.opt/model.onnx"),
    "vae":         (VAEModel,           "onnx/vae.opt/model.onnx"),
}

engines = {}

for name, (ModelClass, onnx_path) in models_config.items():
    engine_path = f"checkpoints_trt/{name}/engine.plan"
    engine = ModelClass(engine_path=engine_path, build=True)

    input_profile = engine.get_input_profile(
        opt_batch_size=1,
        opt_image_height=1024,
        opt_image_width=1024,
        static_batch=True,
        dynamic_shape=True,
    )

    engine.build(onnx_path=onnx_path, input_profile=input_profile)
    engines[name] = engine
