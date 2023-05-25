# from transformers import PretrainedConfig

# from diffusers.pipelines.kandinsky import MultilingualCLIP

# out = "/home/yiyi_huggingface_co/model_repo/Kandinsky"
# out_prior = "/home/yiyi_huggingface_co/model_repo/Kandinsky-prior"
# # # prior_tokenizer/tokenizer2
# # tokenizer2 = CLIPTokenizer.from_pretrained("kakaobrain/karlo-v1-alpha", subfolder="tokenizer")
# # tokenizer2.save_pretrained(f"{out}/prior_tokenizer")
# # print(f"tokenizer saved at {out}/prior_tokenizer")
# # # prior_text_encoder/text_encoder
# # prior_text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
# # prior_text_encoder.save_pretrained(f"{out}/prior_text_encoder")
# # print(f"text_encoder saved at {out}/text_encoder")
# # image_encoder
# clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
# clip_image_encoder.save_pretrained(f"{out_prior}/image_encoder")
# # text_encoder & tokenizer
# model_name = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
# mclip_text_encoder = MultilingualCLIP.from_pretrained(model_name)
# mclip_tokenizer = AutoTokenizer.from_pretrained(model_name)
# mclip_text_encoder.save_pretrained(f"{out}/text_encoder")
# mclip_tokenizer.save_pretrained(f"{out}/tokenizer")
# tiny-random-mclip-base
# from transformers import XLMRobertaConfig, XLMRobertaForMaskedLM
# config= XLMRobertaConfig(
#     bos_token_id=0,
#     eos_token_id=2,
#     hidden_size=32,
#     intermediate_size=37,
#     layer_norm_eps=1e-05,
#     num_attention_heads=4,
#     num_hidden_layers=5,
#     pad_token_id=1,
#     vocab_size=1000)
# base_model = XLMRobertaForMaskedLM(config)
# base_model.save_pretrained("/home/yiyi_huggingface_co/model_repo/tiny-random-mclip-base")

# tiny-random-mclip
# from diffusers.pipelines.kandinsky.text_encoder import MultilingualCLIP


# config = PretrainedConfig(modelBase="YiYiXu/tiny-random-mclip-base", numDims=100, transformerDimensions=32)

# mclip_testing = MultilingualCLIP(config)
# print(mclip_testing)
