# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

from diffusers.modular_pipelines import (
    HunyuanVideo15Blocks,
    HunyuanVideo15ModularPipeline,
)


class TestHunyuanVideo15ModularPipelineStructure(unittest.TestCase):
    def test_import(self):
        blocks = HunyuanVideo15Blocks()
        self.assertIsNotNone(blocks)

    def test_pipeline_class(self):
        blocks = HunyuanVideo15Blocks()
        pipe = blocks.init_pipeline()
        self.assertIsInstance(pipe, HunyuanVideo15ModularPipeline)

    def test_block_names(self):
        blocks = HunyuanVideo15Blocks()
        self.assertEqual(blocks.block_names, ["text_encoder", "denoise", "decode"])

    def test_denoise_sub_blocks(self):
        blocks = HunyuanVideo15Blocks()
        denoise = blocks.sub_blocks["denoise"]
        self.assertEqual(
            list(denoise.sub_blocks.keys()),
            ["input", "set_timesteps", "prepare_latents", "denoise"],
        )

    def test_denoise_loop_sub_blocks(self):
        blocks = HunyuanVideo15Blocks()
        denoise_loop = blocks.sub_blocks["denoise"].sub_blocks["denoise"]
        self.assertEqual(
            list(denoise_loop.sub_blocks.keys()),
            ["before_denoiser", "denoiser", "after_denoiser"],
        )

    def test_expected_components(self):
        blocks = HunyuanVideo15Blocks()
        comp_names = {c.name for c in blocks.expected_components}
        self.assertIn("transformer", comp_names)
        self.assertIn("vae", comp_names)
        self.assertIn("text_encoder", comp_names)
        self.assertIn("text_encoder_2", comp_names)
        self.assertIn("tokenizer", comp_names)
        self.assertIn("tokenizer_2", comp_names)
        self.assertIn("scheduler", comp_names)
        self.assertIn("guider", comp_names)

    def test_model_name(self):
        blocks = HunyuanVideo15Blocks()
        self.assertEqual(blocks.model_name, "hunyuan-video-1.5")

    def test_top_level_export(self):
        from diffusers import HunyuanVideo15Blocks as Top, HunyuanVideo15ModularPipeline as TopPipe

        self.assertIs(Top, HunyuanVideo15Blocks)
        self.assertIs(TopPipe, HunyuanVideo15ModularPipeline)


if __name__ == "__main__":
    unittest.main()
