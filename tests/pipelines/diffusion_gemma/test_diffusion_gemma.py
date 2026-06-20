import unittest

import torch

from diffusers import BlockRefinementScheduler, DiffusionGemmaPipeline


# --- Lightweight stand-in for input-validation tests that never reach the model ---


class _DummyTextConfig:
    def __init__(self, vocab_size: int):
        self.vocab_size = int(vocab_size)
        self.eos_token_id = None


class _DummyConfig:
    def __init__(self, canvas_length: int, vocab_size: int):
        self.canvas_length = int(canvas_length)
        self._text_config = _DummyTextConfig(vocab_size)

    def get_text_config(self, decoder: bool = False):
        return self._text_config


class _DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, canvas_length: int = 8):
        super().__init__()
        self.config = _DummyConfig(canvas_length, vocab_size)


def _make_dummy_pipeline(processor=None, canvas_length: int = 8):
    model = _DummyModel(vocab_size=32, canvas_length=canvas_length)
    return DiffusionGemmaPipeline(model=model, scheduler=BlockRefinementScheduler(), processor=processor)


class DiffusionGemmaPipelineInputTest(unittest.TestCase):
    """Input validation and prompt encoding, which short-circuit before the model is called."""

    def test_no_inputs_raises(self):
        pipe = _make_dummy_pipeline()
        with self.assertRaises(ValueError):
            pipe(gen_length=8, num_inference_steps=2, output_type="seq")

    def test_output_type_invalid_raises(self):
        pipe = _make_dummy_pipeline()
        with self.assertRaises(ValueError):
            pipe(prompt="hi", gen_length=8, output_type="invalid")

    def test_prompt_and_messages_together_raises(self):
        pipe = _make_dummy_pipeline()
        with self.assertRaises(ValueError):
            pipe(prompt="hi", messages=[{"role": "user", "content": "hi"}], gen_length=8, output_type="seq")


# --- End-to-end generation: the prefill-once path drives the real encoder/decoder, so it needs the tiny model ---

_MODEL_ID = "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion"


def _load_pipeline(test):
    try:
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    except ImportError as e:
        test.skipTest(f"transformers without DiffusionGemma: {e}")
    try:
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(_MODEL_ID, dtype=torch.float32).eval()
        processor = AutoProcessor.from_pretrained(_MODEL_ID)
    except Exception as e:  # noqa: BLE001 - offline / hub errors should skip, not fail
        test.skipTest(f"tiny DiffusionGemma checkpoint unavailable: {e}")
    pipe = DiffusionGemmaPipeline(model=model, scheduler=BlockRefinementScheduler(), processor=processor)
    pipe.set_progress_bar_config(disable=True)
    return pipe, model.config.canvas_length


class DiffusionGemmaPipelineTest(unittest.TestCase):
    def setUp(self):
        self.pipe, self.canvas_length = _load_pipeline(self)
        self.prompt = "Name a color."

    def test_generate_seq_shape(self):
        out = self.pipe(
            prompt=self.prompt,
            gen_length=self.canvas_length * 2,
            num_inference_steps=4,
            temperature=0.0,
            eos_early_stop=False,
            output_type="seq",
        )
        self.assertEqual(out.sequences.shape, (1, self.canvas_length * 2))
        self.assertIsNone(out.texts)

    def test_generate_text_and_return_tuple(self):
        sequences, texts = self.pipe(
            prompt=self.prompt,
            gen_length=self.canvas_length,
            num_inference_steps=4,
            temperature=0.0,
            eos_early_stop=False,
            output_type="text",
            return_dict=False,
        )
        self.assertEqual(sequences.shape, (1, self.canvas_length))
        self.assertEqual(len(texts), 1)

    def test_callback_receives_advertised_keys(self):
        observed: list[str] = []

        def callback(pipe, step, timestep, callback_kwargs):
            observed.extend(sorted(callback_kwargs.keys()))
            return {}

        keys = list(self.pipe._callback_tensor_inputs)
        self.pipe(
            prompt=self.prompt,
            gen_length=self.canvas_length,
            num_inference_steps=2,
            temperature=0.0,
            eos_early_stop=False,
            output_type="seq",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=keys,
        )
        self.assertEqual(set(observed), set(keys))

    def test_generate_with_image(self):
        import numpy as np
        from PIL import Image

        image = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
        out = self.pipe(
            prompt="What?",
            image=image,
            gen_length=self.canvas_length,
            num_inference_steps=2,
            temperature=0.0,
            eos_early_stop=False,
            output_type="seq",
        )
        self.assertEqual(out.sequences.shape, (1, self.canvas_length))

    def test_schedulers_are_interchangeable(self):
        from diffusers import DiscreteDDIMScheduler, EntropyBoundScheduler

        for scheduler in (DiscreteDDIMScheduler(), EntropyBoundScheduler(entropy_bound=0.1)):
            self.pipe.scheduler = scheduler
            out = self.pipe(
                prompt=self.prompt,
                gen_length=self.canvas_length,
                num_inference_steps=4,
                temperature=0.0,
                eos_early_stop=False,
                output_type="seq",
            )
            self.assertEqual(out.sequences.shape, (1, self.canvas_length))

    def test_predictor_corrector_sampling(self):
        from diffusers import DiscreteDDIMScheduler

        self.pipe.scheduler = DiscreteDDIMScheduler(corrector_steps=2, corrector_k=2)
        out = self.pipe(
            prompt=self.prompt,
            gen_length=self.canvas_length,
            num_inference_steps=4,
            temperature=0.0,
            eos_early_stop=False,
            output_type="seq",
        )
        self.assertEqual(out.sequences.shape, (1, self.canvas_length))

    def test_static_cache_matches_dynamic(self):
        kwargs = {
            "prompt": self.prompt,
            "gen_length": self.canvas_length * 2,  # two canvases -> exercises the cache extension between blocks
            "num_inference_steps": 4,
            "temperature": 0.0,
            "eos_early_stop": False,
            "output_type": "seq",
        }
        dynamic = self.pipe(generator=torch.Generator().manual_seed(0), **kwargs).sequences
        static = self.pipe(
            generator=torch.Generator().manual_seed(0), cache_implementation="static", **kwargs
        ).sequences
        self.assertTrue(torch.equal(dynamic, static))


if __name__ == "__main__":
    unittest.main()
