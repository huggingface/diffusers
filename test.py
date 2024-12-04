import os
os.environ['HF_HUB_CACHE'] = '/share/shitao/downloaded_models2'

# from huggingface_hub import snapshot_download

# from diffusers.models import OmniGenTransformer2DModel
# from transformers import Phi3Model, Phi3Config


# from safetensors.torch import load_file

# model_name = "Shitao/OmniGen-v1"
# config = Phi3Config.from_pretrained("Shitao/OmniGen-v1")
# model = OmniGenTransformer2DModel(transformer_config=config)
# cache_folder = os.getenv('HF_HUB_CACHE')
# model_name = snapshot_download(repo_id=model_name,
#                                 cache_dir=cache_folder,
#                                 ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])
# print(model_name)
# model_path = os.path.join(model_name, 'model.safetensors')
# ckpt = load_file(model_path, 'cpu')


# mapping_dict = {
#     "pos_embed": "patch_embedding.pos_embed",
#     "x_embedder.proj.weight": "patch_embedding.output_image_proj.weight",
#     "x_embedder.proj.bias": "patch_embedding.output_image_proj.bias",
#     "input_x_embedder.proj.weight": "patch_embedding.input_image_proj.weight",
#     "input_x_embedder.proj.bias": "patch_embedding.input_image_proj.bias",
#     "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
#     "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
#     "final_layer.linear.weight": "proj_out.weight",
#     "final_layer.linear.bias": "proj_out.bias",

# }

# new_ckpt = {}
# for k, v in ckpt.items():
#     # new_ckpt[k] = v
#     if k in mapping_dict:
#         new_ckpt[mapping_dict[k]] = v
#     else:
#         new_ckpt[k] = v
    


# model.load_state_dict(new_ckpt)


from tests.pipelines.omnigen.test_pipeline_omnigen import OmniGenPipelineFastTests, OmniGenPipelineSlowTests

test1 = OmniGenPipelineFastTests()
test1.test_inference()

test2 = OmniGenPipelineSlowTests()
test2.test_omnigen_inference()



