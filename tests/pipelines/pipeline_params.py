# These are canonical sets of parameters for different types of pipelines.
# They are set on subclasses of `PipelineTesterMixin` as `params` and
# `batch_params`.
#
# If a pipeline's set of arguments has minor changes from one of the common sets
# of arguments, do not make modifications to the existing common sets of arguments.
# I.e. a text to image pipeline with non-configurable height and width arguments
# should set its attribute as `params = TEXT_TO_IMAGE_PARAMS - {'height', 'width'}`.

TEXT_TO_IMAGE_PARAMS = frozenset(
    [
        "prompt",
        "height",
        "width",
        "guidance_scale",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
        "cross_attention_kwargs",
    ]
)

TEXT_TO_IMAGE_BATCH_PARAMS = frozenset(["prompt", "negative_prompt"])

TEXT_TO_IMAGE_IMAGE_PARAMS = frozenset([])

IMAGE_TO_IMAGE_IMAGE_PARAMS = frozenset(["image"])

IMAGE_VARIATION_PARAMS = frozenset(
    [
        "image",
        "height",
        "width",
        "guidance_scale",
    ]
)

IMAGE_VARIATION_BATCH_PARAMS = frozenset(["image"])

TEXT_GUIDED_IMAGE_VARIATION_PARAMS = frozenset(
    [
        "prompt",
        "image",
        "height",
        "width",
        "guidance_scale",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]
)

TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS = frozenset(["prompt", "image", "negative_prompt"])

TEXT_GUIDED_IMAGE_INPAINTING_PARAMS = frozenset(
    [
        # Text guided image variation with an image mask
        "prompt",
        "image",
        "mask_image",
        "height",
        "width",
        "guidance_scale",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]
)

TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS = frozenset(["prompt", "image", "mask_image", "negative_prompt"])

IMAGE_INPAINTING_PARAMS = frozenset(
    [
        # image variation with an image mask
        "image",
        "mask_image",
        "height",
        "width",
        "guidance_scale",
    ]
)

IMAGE_INPAINTING_BATCH_PARAMS = frozenset(["image", "mask_image"])

IMAGE_GUIDED_IMAGE_INPAINTING_PARAMS = frozenset(
    [
        "example_image",
        "image",
        "mask_image",
        "height",
        "width",
        "guidance_scale",
    ]
)

IMAGE_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS = frozenset(["example_image", "image", "mask_image"])

CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS = frozenset(["class_labels"])

CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS = frozenset(["class_labels"])

UNCONDITIONAL_IMAGE_GENERATION_PARAMS = frozenset(["batch_size"])

UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS = frozenset([])

UNCONDITIONAL_AUDIO_GENERATION_PARAMS = frozenset(["batch_size"])

UNCONDITIONAL_AUDIO_GENERATION_BATCH_PARAMS = frozenset([])

TEXT_TO_AUDIO_PARAMS = frozenset(
    [
        "prompt",
        "audio_length_in_s",
        "guidance_scale",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
        "cross_attention_kwargs",
    ]
)

TEXT_TO_AUDIO_BATCH_PARAMS = frozenset(["prompt", "negative_prompt"])
TOKENS_TO_AUDIO_GENERATION_PARAMS = frozenset(["input_tokens"])

TOKENS_TO_AUDIO_GENERATION_BATCH_PARAMS = frozenset(["input_tokens"])

TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS = frozenset(["prompt_embeds"])
