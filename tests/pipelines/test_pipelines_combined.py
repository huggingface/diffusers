# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

import torch
from huggingface_hub import ModelCard

from diffusers import (
    DDPMScheduler,
    DiffusionPipeline,
    KandinskyV22CombinedPipeline,
    KandinskyV22Pipeline,
    KandinskyV22PriorPipeline,
)
from diffusers.pipelines.pipeline_utils import CONNECTED_PIPES_KEYS


def state_dicts_almost_equal(sd1, sd2):
    sd1 = dict(sorted(sd1.items()))
    sd2 = dict(sorted(sd2.items()))

    models_are_equal = True
    for ten1, ten2 in zip(sd1.values(), sd2.values()):
        if (ten1 - ten2).abs().sum() > 1e-3:
            models_are_equal = False

    return models_are_equal


class CombinedPipelineFastTest(unittest.TestCase):
    def modelcard_has_connected_pipeline(self, model_id):
        modelcard = ModelCard.load(model_id)
        connected_pipes = {prefix: getattr(modelcard.data, prefix, [None])[0] for prefix in CONNECTED_PIPES_KEYS}
        connected_pipes = {k: v for k, v in connected_pipes.items() if v is not None}

        return len(connected_pipes) > 0

    def test_correct_modelcard_format(self):
        # hf-internal-testing/tiny-random-kandinsky-v22-prior has no metadata
        assert not self.modelcard_has_connected_pipeline("hf-internal-testing/tiny-random-kandinsky-v22-prior")

        # see https://huggingface.co/hf-internal-testing/tiny-random-kandinsky-v22-decoder/blob/8baff9897c6be017013e21b5c562e5a381646c7e/README.md?code=true#L2
        assert self.modelcard_has_connected_pipeline("hf-internal-testing/tiny-random-kandinsky-v22-decoder")

    def test_load_connected_checkpoint_when_specified(self):
        pipeline_prior = DiffusionPipeline.from_pretrained("hf-internal-testing/tiny-random-kandinsky-v22-prior")
        pipeline_prior_connected = DiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-random-kandinsky-v22-prior", load_connected_pipeline=True
        )

        # Passing `load_connected_pipeline` to prior is a no-op as the pipeline has no connected pipeline
        assert pipeline_prior.__class__ == pipeline_prior_connected.__class__

        pipeline = DiffusionPipeline.from_pretrained("hf-internal-testing/tiny-random-kandinsky-v22-decoder")
        pipeline_connected = DiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-random-kandinsky-v22-decoder", load_connected_pipeline=True
        )

        # Passing `load_connected_pipeline` to decoder loads the combined pipeline
        assert pipeline.__class__ != pipeline_connected.__class__
        assert pipeline.__class__ == KandinskyV22Pipeline
        assert pipeline_connected.__class__ == KandinskyV22CombinedPipeline

        # check that loaded components match prior and decoder components
        assert set(pipeline_connected.components.keys()) == set(
            ["prior_" + k for k in pipeline_prior.components.keys()] + list(pipeline.components.keys())
        )

    def test_load_connected_checkpoint_default(self):
        prior = KandinskyV22PriorPipeline.from_pretrained("hf-internal-testing/tiny-random-kandinsky-v22-prior")
        decoder = KandinskyV22Pipeline.from_pretrained("hf-internal-testing/tiny-random-kandinsky-v22-decoder")

        # check that combined pipeline loads both prior & decoder because of
        # https://huggingface.co/hf-internal-testing/tiny-random-kandinsky-v22-decoder/blob/8baff9897c6be017013e21b5c562e5a381646c7e/README.md?code=true#L3
        assert (
            KandinskyV22CombinedPipeline._load_connected_pipes
        )  # combined pipelines will download more checkpoints that just the one specified
        pipeline = KandinskyV22CombinedPipeline.from_pretrained(
            "hf-internal-testing/tiny-random-kandinsky-v22-decoder"
        )

        prior_comps = prior.components
        decoder_comps = decoder.components
        for k, component in pipeline.components.items():
            if k.startswith("prior_"):
                k = k[6:]
                comp = prior_comps[k]
            else:
                comp = decoder_comps[k]

            if isinstance(component, torch.nn.Module):
                assert state_dicts_almost_equal(component.state_dict(), comp.state_dict())
            elif hasattr(component, "config"):
                assert dict(component.config) == dict(comp.config)
            else:
                assert component.__class__ == comp.__class__

    def test_load_connected_checkpoint_with_passed_obj(self):
        pipeline = KandinskyV22CombinedPipeline.from_pretrained(
            "hf-internal-testing/tiny-random-kandinsky-v22-decoder"
        )
        prior_scheduler = DDPMScheduler.from_config(pipeline.prior_scheduler.config)
        scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

        # make sure we pass a different scheduler and prior_scheduler
        assert pipeline.prior_scheduler.__class__ != prior_scheduler.__class__
        assert pipeline.scheduler.__class__ != scheduler.__class__

        pipeline_new = KandinskyV22CombinedPipeline.from_pretrained(
            "hf-internal-testing/tiny-random-kandinsky-v22-decoder",
            prior_scheduler=prior_scheduler,
            scheduler=scheduler,
        )
        assert dict(pipeline_new.prior_scheduler.config) == dict(prior_scheduler.config)
        assert dict(pipeline_new.scheduler.config) == dict(scheduler.config)
