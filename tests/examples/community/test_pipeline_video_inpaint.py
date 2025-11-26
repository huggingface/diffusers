import numpy as np
import PIL.Image
import torch

from examples.community.pipeline_video_inpaint import VideoInpaintPipeline


def make_pipeline_stub():
    pipe = object.__new__(VideoInpaintPipeline)
    pipe.vae_scale_factor = 8
    pipe.vae = None  # only needed for attributes in helper tests
    return pipe


def test_resize_flow_scales_to_target_shape():
    flow = torch.ones(1, 2, 4, 4)
    resized = VideoInpaintPipeline._resize_flow(flow, (2, 2))
    assert resized.shape[-2:] == (2, 2)
    # Values should be scaled by target/source ratio (0.5 here)
    assert torch.allclose(resized, torch.full_like(resized, 0.5))


def test_warp_tensor_identity_for_zero_flow():
    tensor = torch.arange(9.0).view(1, 1, 3, 3)
    flow = torch.zeros(1, 2, 3, 3)
    warped = VideoInpaintPipeline._warp_tensor(tensor, flow)
    assert torch.allclose(warped, tensor)


def test_prepare_noise_reuses_previous_when_blend_one():
    pipe = make_pipeline_stub()
    prev_noise = torch.randn(1, 4, 2, 2)
    result = pipe._prepare_noise(
        latent_shape=prev_noise.shape,
        generator=None,
        dtype=prev_noise.dtype,
        device=prev_noise.device,
        prev_noise=prev_noise,
        flow=None,
        noise_blend=1.0,
    )
    assert torch.allclose(result, prev_noise)


def test_prepare_latent_hint_resizes_and_warps():
    pipe = make_pipeline_stub()
    latents = torch.randn(1, 4, 4, 4)
    flow = torch.zeros(1, 2, 4, 4)
    flow[:, 0] = 1.0  # shift right by 1px
    hint = pipe._prepare_latent_hint(latents, target_shape=(2, 2), flow=flow, strength=1.0)
    assert hint is not None
    assert hint.shape[-2:] == (2, 2)


def test_ensure_mask_frames_repeats_single_mask():
    pipe = make_pipeline_stub()
    frame_size = (64, 32)
    mask = PIL.Image.fromarray(np.zeros(frame_size, dtype=np.uint8))
    masks = pipe._ensure_mask_frames([mask], num_frames=3, frame_size=frame_size)
    assert len(masks) == 3
    for m in masks:
        assert m.size == frame_size
