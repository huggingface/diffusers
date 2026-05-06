import unittest

import torch

from diffusers import BlockRefinementScheduler, UniLLaDaPipeline
from diffusers.pipelines.unillada.pipeline_output import UniLLaDaPipelineOutput


class _DummyConfig:
    image_token_offset = 100


class _DummyGenerateResult(dict):
    """Mimic the dict returned by transformer.generate_image()."""

    pass


class _DummyTransformer:
    """Mock transformer that mimics the UniLLaDA backbone interface."""

    config = _DummyConfig()
    _hf_hook = None  # needed for DiffusionPipeline

    def generate_image(self, prompt, image_h=1024, image_w=1024, steps=8, cfg_scale=2.0, block_length=32):
        return {"token_ids": list(range(16)), "h": 4, "w": 4}

    def understand_image(self, image_tokens, h, w, question, steps=8):
        return "This is a test response."

    def edit_image(
        self, image_tokens, h, w, instruction, steps=8, block_length=32, cfg_text_scale=2.0, cfg_image_scale=0.0
    ):
        return {"token_ids": list(range(16)), "h": h, "w": w}

    # Required for register_modules
    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32


class _DummyTokenizer:
    """Mock tokenizer."""

    eos_token_id = 2
    mask_token_id = 31
    _hf_hook = None


class _DummyImageTokenizer:
    """Mock image tokenizer."""

    _hf_hook = None

    def encode_with_info(self, image):
        return {"token_ids": list(range(16)), "grid_thw": (1, 4, 4)}


def _make_pipeline(with_image_tokenizer=False):
    transformer = _DummyTransformer()
    tokenizer = _DummyTokenizer()
    scheduler = BlockRefinementScheduler()
    image_tokenizer = _DummyImageTokenizer() if with_image_tokenizer else None
    return UniLLaDaPipeline(
        transformer=transformer,
        tokenizer=tokenizer,
        scheduler=scheduler,
        image_tokenizer=image_tokenizer,
    )


def _dummy_decode_fn(token_ids, h, w, **kwargs):
    """Return a small dummy PIL image."""
    import PIL.Image

    return PIL.Image.new("RGB", (64, 64), color=(128, 128, 128))


class UniLLaDaPipelineTest(unittest.TestCase):
    def test_text_to_image(self):
        pipe = _make_pipeline()
        out = pipe(
            prompt="A test prompt",
            decode_fn=_dummy_decode_fn,
            output_type="pil",
        )
        self.assertIsInstance(out, UniLLaDaPipelineOutput)
        self.assertIsNotNone(out.images)
        self.assertEqual(len(out.images), 1)

    def test_text_to_image_tokens(self):
        pipe = _make_pipeline()
        out = pipe(
            prompt="A test prompt",
            output_type="tokens",
        )
        self.assertIsNone(out.images)
        self.assertIsNotNone(out.text)

    def test_image_understanding(self):
        import PIL.Image

        pipe = _make_pipeline(with_image_tokenizer=True)
        img = PIL.Image.new("RGB", (256, 256))
        out = pipe(image=img, question="What is this?")
        self.assertIsInstance(out, UniLLaDaPipelineOutput)
        self.assertIsNotNone(out.text)
        self.assertEqual(out.text, "This is a test response.")

    def test_image_editing(self):
        import PIL.Image

        pipe = _make_pipeline(with_image_tokenizer=True)
        img = PIL.Image.new("RGB", (256, 256))
        out = pipe(
            image=img,
            instruction="Make it red",
            decode_fn=_dummy_decode_fn,
            output_type="pil",
        )
        self.assertIsNotNone(out.images)
        self.assertEqual(len(out.images), 1)

    def test_invalid_input_raises(self):
        pipe = _make_pipeline()
        with self.assertRaises(ValueError):
            pipe()  # No inputs

    def test_invalid_output_type_raises(self):
        pipe = _make_pipeline()
        with self.assertRaises(ValueError):
            pipe(prompt="test", output_type="invalid")

    def test_understanding_without_image_tokenizer_raises(self):
        import PIL.Image

        pipe = _make_pipeline(with_image_tokenizer=False)
        img = PIL.Image.new("RGB", (256, 256))
        with self.assertRaises(ValueError):
            pipe(image=img, question="What is this?")

    def test_return_dict_false(self):
        pipe = _make_pipeline()
        out = pipe(
            prompt="A test prompt",
            decode_fn=_dummy_decode_fn,
            output_type="pil",
            return_dict=False,
        )
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
