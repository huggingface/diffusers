import unittest

import torch

from diffusers import BlockRefinementScheduler, DiffusionGemmaPipeline


class _DummyModelOutput:
    def __init__(self, logits):
        self.logits = logits


class _DummyTextConfig:
    def __init__(self, vocab_size: int):
        self.vocab_size = int(vocab_size)
        self.eos_token_id = None


class _DummyConfig:
    def __init__(self, canvas_length: int, vocab_size: int):
        self.canvas_length = int(canvas_length)
        self._text_config = _DummyTextConfig(vocab_size)

    def get_text_config(self):
        return self._text_config


class _DummyBlockDiffusionModel(torch.nn.Module):
    """Stand-in for `DiffusionGemmaForBlockDiffusion`: returns logits over the decoder canvas."""

    def __init__(self, vocab_size: int = 32, canvas_length: int = 8):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.config = _DummyConfig(canvas_length, vocab_size)
        self.register_buffer("_device_anchor", torch.empty(0))

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return self._device_anchor.device

    def forward(self, input_ids=None, decoder_input_ids=None, **kwargs):
        batch_size, canvas_len = decoder_input_ids.shape
        device = decoder_input_ids.device
        logits = torch.zeros((batch_size, canvas_len, self.vocab_size), device=device, dtype=torch.float32)
        # Make confidence vary with canvas position so the commit quota is deterministic.
        positions = torch.arange(canvas_len, device=device, dtype=torch.float32).view(1, canvas_len, 1)
        token_ids = (torch.arange(canvas_len, device=device) % (self.vocab_size - 2)).view(1, canvas_len, 1)
        logits.scatter_(2, token_ids.expand(batch_size, -1, -1), 1.0 + positions.expand(batch_size, -1, -1) * 0.1)
        return _DummyModelOutput(logits=logits)


def _make_pipeline(processor=None, canvas_length: int = 8):
    model = _DummyBlockDiffusionModel(vocab_size=32, canvas_length=canvas_length)
    scheduler = BlockRefinementScheduler()
    return DiffusionGemmaPipeline(model=model, scheduler=scheduler, processor=processor)


class DiffusionGemmaPipelineTest(unittest.TestCase):
    def test_pipeline_runs(self):
        pipe = _make_pipeline().to("cpu")
        input_ids = torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.long)
        out = pipe(
            input_ids=input_ids,
            gen_length=24,  # 3 canvases of length 8
            num_inference_steps=8,
            temperature=0.0,
            eos_early_stop=False,
            output_type="seq",
        )
        self.assertEqual(out.sequences.shape, (2, 24))
        self.assertIsNone(out.texts)

    def test_pipeline_return_tuple(self):
        pipe = _make_pipeline().to("cpu")
        sequences, texts = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            gen_length=16,
            num_inference_steps=4,
            eos_early_stop=False,
            output_type="seq",
            return_dict=False,
        )
        self.assertEqual(sequences.shape, (1, 16))
        self.assertIsNone(texts)

    def test_output_type_text_with_processor(self):
        processor = type(
            "Proc",
            (),
            {
                "tokenizer": type("Tok", (), {"eos_token_id": None})(),
                "batch_decode": lambda self, seqs, **kw: [f"decoded_{len(s)}" for s in seqs],
            },
        )()
        pipe = _make_pipeline(processor=processor).to("cpu")
        out = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            gen_length=16,
            num_inference_steps=4,
            eos_early_stop=False,
            output_type="text",
        )
        self.assertIsNotNone(out.texts)
        self.assertEqual(len(out.texts), 1)
        self.assertTrue(out.texts[0].startswith("decoded_"))

    def test_output_type_invalid_raises(self):
        pipe = _make_pipeline().to("cpu")
        with self.assertRaises(ValueError):
            pipe(
                input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
                gen_length=8,
                num_inference_steps=2,
                output_type="invalid",
            )

    def test_no_inputs_raises(self):
        pipe = _make_pipeline().to("cpu")
        with self.assertRaises(ValueError):
            pipe(gen_length=8, num_inference_steps=2, output_type="seq")

    def test_prepare_input_ids_from_1d_tensor(self):
        pipe = _make_pipeline()
        ids = torch.tensor([1, 2, 3], dtype=torch.long)
        result_ids, result_mask = pipe._prepare_input_ids(
            prompt=None, messages=None, input_ids=ids, attention_mask=None, add_generation_prompt=False
        )
        self.assertEqual(result_ids.shape, (1, 3))
        self.assertEqual(result_mask.shape, (1, 3))
        self.assertTrue((result_mask == 1).all().item())

    def test_callback_receives_advertised_keys(self):
        observed: list[str] = []

        def cb(pipe, step, timestep, kwargs):
            observed.extend(sorted(kwargs.keys()))
            return {}

        pipe = _make_pipeline().to("cpu")
        keys = list(pipe._callback_tensor_inputs)
        pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            gen_length=8,
            num_inference_steps=4,
            eos_early_stop=False,
            output_type="seq",
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=keys,
        )
        self.assertEqual(set(observed), set(keys))

    def test_progress_bar_disable_is_preserved_after_call(self):
        pipe = _make_pipeline().to("cpu")
        pipe.set_progress_bar_config(disable=True)
        before = dict(pipe._progress_bar_config)
        pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            gen_length=8,
            num_inference_steps=2,
            eos_early_stop=False,
            output_type="seq",
        )
        self.assertEqual(pipe._progress_bar_config, before)


class DiffusionGemmaStaticCacheTest(unittest.TestCase):
    """The static-cache path uses the real model internals (encoder prefill + `StaticCache`), so it needs the tiny
    checkpoint rather than a stand-in. Skips when the model can't be fetched (e.g. offline CI)."""

    def _load_pipeline(self):
        try:
            from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
        except ImportError as e:
            self.skipTest(f"transformers without DiffusionGemma: {e}")
        model_id = "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion"
        try:
            model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_id, dtype=torch.float32).eval()
            processor = AutoProcessor.from_pretrained(model_id)
        except Exception as e:  # noqa: BLE001 - offline / hub errors should skip, not fail
            self.skipTest(f"tiny DiffusionGemma checkpoint unavailable: {e}")
        pipe = DiffusionGemmaPipeline(model=model, scheduler=BlockRefinementScheduler(), processor=processor)
        pipe.set_progress_bar_config(disable=True)
        return pipe, model.config.canvas_length

    def test_static_cache_matches_dynamic(self):
        pipe, canvas_length = self._load_pipeline()
        kwargs = {
            "messages": [{"role": "user", "content": "Name a color."}],
            "gen_length": canvas_length * 2,  # two canvases -> exercises the cache extension between blocks
            "num_inference_steps": 4,
            "temperature": 0.0,
            "eos_early_stop": False,
            "output_type": "seq",
        }
        dynamic = pipe(generator=torch.Generator().manual_seed(0), **kwargs).sequences
        static = pipe(generator=torch.Generator().manual_seed(0), cache_implementation="static", **kwargs).sequences
        self.assertEqual(dynamic.shape, (1, canvas_length * 2))
        self.assertTrue(torch.equal(dynamic, static))


if __name__ == "__main__":
    unittest.main()
