import unittest
from types import SimpleNamespace

import torch

from diffusers import BlockRefinementScheduler, DiffusionGemmaPipeline, EntropyBoundScheduler
from diffusers.utils.testing_utils import require_peft_backend, require_peft_version_greater


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
    adaptive_stopping_vocab_size = 8

    def setUp(self):
        self.pipe, self.canvas_length = _load_pipeline(self)
        self.prompt = "Name a color."

    def _run_adaptive_stopping(self, prompt):
        self.pipe.model.config.get_text_config(decoder=True).vocab_size = self.adaptive_stopping_vocab_size
        return self.pipe(
            prompt=prompt,
            gen_length=self.canvas_length,
            num_inference_steps=5,
            confidence_threshold=0.005,
            eos_early_stop=False,
            output_type="seq",
        )

    def test_generate(self):
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

    def test_adaptive_stopping_freezes_finished_rows(self):
        forward_calls = 0

        def forward(decoder_input_ids, **kwargs):
            nonlocal forward_calls
            batch_size, canvas_length = decoder_input_ids.shape
            token_ids = ([1, 3], [1, 4], [2, 5], [2, 5], [2, 6])[forward_calls]
            tokens = torch.tensor(token_ids, device=decoder_input_ids.device)[:, None].expand_as(decoder_input_ids)
            logits = torch.full(
                (batch_size, canvas_length, self.adaptive_stopping_vocab_size),
                -100.0,
                device=decoder_input_ids.device,
            )
            logits.scatter_(-1, tokens[..., None], 100.0)
            forward_calls += 1
            return SimpleNamespace(logits=logits)

        self.pipe.model.forward = forward
        self.pipe.scheduler = BlockRefinementScheduler()
        output = self._run_adaptive_stopping(["Short prompt.", "A somewhat longer prompt for the second batch row."])

        self.assertEqual(forward_calls, 4)
        self.assertTrue((output.sequences[0] == 1).all())
        self.assertTrue((output.sequences[1] == 5).all())

    def test_adaptive_stopping_uses_scheduler_logits(self):
        forward_calls = 0

        def forward(decoder_input_ids, **kwargs):
            nonlocal forward_calls
            forward_calls += 1
            batch_size, canvas_length = decoder_input_ids.shape
            logits = torch.zeros(
                batch_size, canvas_length, self.adaptive_stopping_vocab_size, device=decoder_input_ids.device
            )
            logits[..., 0] = 2.0
            return SimpleNamespace(logits=logits)

        self.pipe.model.forward = forward
        self.pipe.scheduler = EntropyBoundScheduler(t_max=0.1, t_min=0.1)
        output = self._run_adaptive_stopping(self.prompt)

        self.assertEqual(forward_calls, 2)
        self.assertTrue((output.sequences == 0).all())

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

    @require_peft_backend
    @require_peft_version_greater("0.18.9")
    def test_peft_adapter_api(self):
        from peft import LoraConfig

        # Adapters are managed on the model component directly (the adapter API is adapter-type-agnostic; LoRA stands
        # in for any PEFT adapter: DoRA, IA3, ...).
        self.pipe.model.add_adapter(
            LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear"),
            adapter_name="test",
        )
        self.pipe.model.set_adapter("test")
        self.assertIn("test", self.pipe.model.active_adapters())

        out = self.pipe(
            prompt=self.prompt,
            gen_length=self.canvas_length,
            num_inference_steps=2,
            temperature=0.0,
            eos_early_stop=False,
            output_type="seq",
        )
        self.assertEqual(out.sequences.shape, (1, self.canvas_length))

        self.pipe.model.disable_adapters()
        self.pipe.model.enable_adapters()
        self.pipe.model.delete_adapter("test")

    def test_static_cache_matches_dynamic(self):
        # Greedy and no adaptive stopping, so the only difference between the two runs is the cache path itself.
        kwargs = {
            "prompt": self.prompt,
            "gen_length": self.canvas_length * 2,  # two canvases -> exercises the cache extension between blocks
            "num_inference_steps": 4,
            "temperature": 0.0,
            "confidence_threshold": None,
            "eos_early_stop": False,
            "output_type": "seq",
        }
        dynamic = self.pipe(generator=torch.Generator().manual_seed(0), **kwargs).sequences
        static = self.pipe(
            generator=torch.Generator().manual_seed(0), cache_implementation="static", **kwargs
        ).sequences
        ndiff = (dynamic != static).sum().item()
        self.assertEqual(ndiff, 0, f"static/dynamic agree on only ndiff={ndiff}/{dynamic.numel()} tokens")


if __name__ == "__main__":
    unittest.main()
