# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import MagicMock

import torch

from diffusers.pipelines.pipeline_utils import DiscreteDiffusionPipelineMixin


class TestTopPFiltering(unittest.TestCase):
    def test_top_p_filtering(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        filtered = DiscreteDiffusionPipelineMixin._top_p_filtering(logits, top_p=0.5)
        # Only the top token(s) summing to <= 0.5 probability should remain;
        # the rest should be -inf (or dtype min).
        # Verify that at least one token survived
        self.assertTrue((filtered > torch.finfo(filtered.dtype).min).any())
        # Verify that some tokens were filtered
        self.assertTrue((filtered == torch.finfo(filtered.dtype).min).any())

    def test_top_p_filtering_none(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = DiscreteDiffusionPipelineMixin._top_p_filtering(logits, top_p=None)
        self.assertTrue(torch.equal(result, logits))

    def test_top_p_filtering_one(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = DiscreteDiffusionPipelineMixin._top_p_filtering(logits, top_p=1.0)
        self.assertTrue(torch.equal(result, logits))


class TestTopKFiltering(unittest.TestCase):
    def test_top_k_filtering(self):
        logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
        filtered = DiscreteDiffusionPipelineMixin._top_k_filtering(logits, top_k=2)
        # Only the top-2 values (4.0 and 3.0) should survive
        self.assertAlmostEqual(filtered[0, 1].item(), 4.0)
        self.assertAlmostEqual(filtered[0, 3].item(), 3.0)
        self.assertEqual(filtered[0, 0].item(), torch.finfo(filtered.dtype).min)
        self.assertEqual(filtered[0, 2].item(), torch.finfo(filtered.dtype).min)

    def test_top_k_filtering_none(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = DiscreteDiffusionPipelineMixin._top_k_filtering(logits, top_k=None)
        self.assertTrue(torch.equal(result, logits))

    def test_top_k_filtering_zero(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = DiscreteDiffusionPipelineMixin._top_k_filtering(logits, top_k=0)
        self.assertTrue(torch.equal(result, logits))

    def test_top_k_filtering_large_k(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = DiscreteDiffusionPipelineMixin._top_k_filtering(logits, top_k=100)
        self.assertTrue(torch.equal(result, logits))


class TestSampleWithTemperature(unittest.TestCase):
    def test_greedy_sampling(self):
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        tokens, probs = DiscreteDiffusionPipelineMixin._sample_with_temperature_topk_topp(
            logits,
            temperature=0.0,
            top_k=None,
            top_p=None,
            generator=None,
            use_multinomial=False,
        )
        self.assertEqual(tokens.item(), 1)  # index of max logit (5.0)
        self.assertEqual(tokens.shape, (1,))
        self.assertEqual(probs.shape, (1,))

    def test_multinomial_sampling(self):
        logits = torch.tensor([[0.0, 100.0, -100.0]])
        gen = torch.Generator().manual_seed(42)
        tokens, probs = DiscreteDiffusionPipelineMixin._sample_with_temperature_topk_topp(
            logits,
            temperature=1.0,
            top_k=None,
            top_p=None,
            generator=gen,
            use_multinomial=True,
        )
        # With such extreme logits, token should always be 1
        self.assertEqual(tokens.item(), 1)

    def test_temperature_scaling(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        # With very low temperature, should pick the max
        tokens, _ = DiscreteDiffusionPipelineMixin._sample_with_temperature_topk_topp(
            logits,
            temperature=0.01,
            top_k=None,
            top_p=None,
            generator=None,
            use_multinomial=False,
        )
        self.assertEqual(tokens.item(), 2)  # index of max logit (3.0)

    def test_negative_temperature_raises(self):
        logits = torch.tensor([[1.0, 2.0]])
        with self.assertRaises(ValueError, msg="`temperature` must be >= 0"):
            DiscreteDiffusionPipelineMixin._sample_with_temperature_topk_topp(
                logits,
                temperature=-1.0,
                top_k=None,
                top_p=None,
                generator=None,
                use_multinomial=False,
            )


class TestResolveStartTokenId(unittest.TestCase):
    def _make_mixin(self, tokenizer=None):
        obj = DiscreteDiffusionPipelineMixin()
        obj.tokenizer = tokenizer
        return obj

    def test_no_tokenizer(self):
        mixin = self._make_mixin(tokenizer=None)
        self.assertIsNone(mixin._resolve_start_token_id())

    def test_bos_token_id(self):
        tok = MagicMock()
        tok.bos_token_id = 1
        tok.cls_token_id = None
        mixin = self._make_mixin(tokenizer=tok)
        self.assertEqual(mixin._resolve_start_token_id(), 1)

    def test_cls_token_id_fallback(self):
        tok = MagicMock()
        tok.bos_token_id = None
        tok.cls_token_id = 101
        mixin = self._make_mixin(tokenizer=tok)
        self.assertEqual(mixin._resolve_start_token_id(), 101)

    def test_no_token_ids(self):
        tok = MagicMock(spec=[])
        mixin = self._make_mixin(tokenizer=tok)
        self.assertIsNone(mixin._resolve_start_token_id())


class TestNormalizePrefixIds(unittest.TestCase):
    def _make_mixin(self):
        return DiscreteDiffusionPipelineMixin()

    def test_1d_input(self):
        mixin = self._make_mixin()
        prefix = torch.tensor([10, 20, 30], dtype=torch.long)
        result = mixin._normalize_prefix_ids(prefix, batch_size=1, device=torch.device("cpu"))
        self.assertEqual(result.shape, (1, 3))

    def test_broadcast(self):
        mixin = self._make_mixin()
        prefix = torch.tensor([[10, 20]], dtype=torch.long)
        result = mixin._normalize_prefix_ids(prefix, batch_size=4, device=torch.device("cpu"))
        self.assertEqual(result.shape, (4, 2))
        self.assertTrue(torch.equal(result[0], result[3]))

    def test_wrong_dtype_raises(self):
        mixin = self._make_mixin()
        prefix = torch.tensor([1.0, 2.0])
        with self.assertRaises(ValueError, msg="int64"):
            mixin._normalize_prefix_ids(prefix, batch_size=1, device=torch.device("cpu"))

    def test_wrong_batch_dim_raises(self):
        mixin = self._make_mixin()
        prefix = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long)
        with self.assertRaises(ValueError, msg="batch dim"):
            mixin._normalize_prefix_ids(prefix, batch_size=2, device=torch.device("cpu"))


class TestPrepareInputIds(unittest.TestCase):
    def _make_mixin(self, tokenizer=None):
        obj = DiscreteDiffusionPipelineMixin()
        obj.tokenizer = tokenizer
        return obj

    def test_from_tensor(self):
        mixin = self._make_mixin()
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        result = mixin._prepare_input_ids(
            prompt=None,
            messages=None,
            input_ids=ids,
            use_chat_template=False,
            add_generation_prompt=False,
            chat_template_kwargs=None,
        )
        self.assertTrue(torch.equal(result, ids))

    def test_from_tensor_1d(self):
        mixin = self._make_mixin()
        ids = torch.tensor([1, 2, 3], dtype=torch.long)
        result = mixin._prepare_input_ids(
            prompt=None,
            messages=None,
            input_ids=ids,
            use_chat_template=False,
            add_generation_prompt=False,
            chat_template_kwargs=None,
        )
        self.assertEqual(result.shape, (1, 3))

    def test_from_prompt(self):
        tok = MagicMock()
        tok.chat_template = None
        tok.return_value = {"input_ids": torch.tensor([[10, 20, 30]])}
        mixin = self._make_mixin(tokenizer=tok)
        result = mixin._prepare_input_ids(
            prompt="hello",
            messages=None,
            input_ids=None,
            use_chat_template=False,
            add_generation_prompt=False,
            chat_template_kwargs=None,
        )
        self.assertEqual(result.shape, (1, 3))
        tok.assert_called_once()

    def test_no_tokenizer_raises(self):
        mixin = self._make_mixin(tokenizer=None)
        with self.assertRaises(ValueError, msg="Tokenizer is required"):
            mixin._prepare_input_ids(
                prompt="hello",
                messages=None,
                input_ids=None,
                use_chat_template=False,
                add_generation_prompt=False,
                chat_template_kwargs=None,
            )

    def test_both_prompt_and_messages_raises(self):
        tok = MagicMock()
        mixin = self._make_mixin(tokenizer=tok)
        with self.assertRaises(ValueError, msg="not both"):
            mixin._prepare_input_ids(
                prompt="hello",
                messages=[{"role": "user", "content": "hi"}],
                input_ids=None,
                use_chat_template=False,
                add_generation_prompt=False,
                chat_template_kwargs=None,
            )

    def test_neither_prompt_nor_messages_raises(self):
        tok = MagicMock()
        mixin = self._make_mixin(tokenizer=tok)
        with self.assertRaises(ValueError, msg="Provide one of"):
            mixin._prepare_input_ids(
                prompt=None,
                messages=None,
                input_ids=None,
                use_chat_template=False,
                add_generation_prompt=False,
                chat_template_kwargs=None,
            )


if __name__ == "__main__":
    unittest.main()
