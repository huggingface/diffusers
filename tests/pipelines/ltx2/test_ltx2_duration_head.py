# Copyright 2026 The HuggingFace Team.
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

import math
import unittest

import pytest
import torch

from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import AttentionBackendName
from diffusers.pipelines.ltx2.duration_head import LTX2AutoDuration, LTX2DurationHead

from ...testing_utils import enable_full_determinism


enable_full_determinism()


class LTX2DurationHeadTests(unittest.TestCase):
    video_dim = 32
    audio_dim = 16
    batch_size = 2

    def get_head(self):
        torch.manual_seed(0)
        return LTX2DurationHead(
            video_cross_attention_dim=self.video_dim,
            audio_cross_attention_dim=self.audio_dim,
            pooler_hidden_dim=8,
            num_queries=1,
            num_pooler_heads=2,
            mlp_hidden_dim=8,
        )

    def test_predicts_positive_seconds_from_video_only(self):
        head = self.get_head()
        video_tokens = torch.randn(self.batch_size, 6, self.video_dim)

        seconds = head(video_tokens=video_tokens)

        assert seconds.shape == (self.batch_size,)
        assert torch.all(seconds > 0)

    def test_predicts_positive_seconds_from_audio_only(self):
        head = self.get_head()
        audio_tokens = torch.randn(self.batch_size, 4, self.audio_dim)

        seconds = head(audio_tokens=audio_tokens)

        assert seconds.shape == (self.batch_size,)
        assert torch.all(seconds > 0)

    def test_predicts_positive_seconds_from_both_modalities(self):
        head = self.get_head()

        seconds = head(
            video_tokens=torch.randn(self.batch_size, 6, self.video_dim),
            audio_tokens=torch.randn(self.batch_size, 4, self.audio_dim),
        )

        assert seconds.shape == (self.batch_size,)
        assert torch.all(seconds > 0)

    def test_output_is_invariant_to_sequence_length(self):
        # The attention pooler produces a fixed-shape output regardless of input length.
        head = self.get_head()

        short = head(video_tokens=torch.randn(1, 3, self.video_dim))
        long = head(video_tokens=torch.randn(1, 128, self.video_dim))

        assert short.shape == long.shape == (1,)

    def test_both_modalities_change_the_prediction(self):
        # Guards against the audio stream being silently dropped: adding audio tokens must move
        # the pooled result, since the pooler attends over the concatenated sequence.
        head = self.get_head()
        video_tokens = torch.randn(1, 6, self.video_dim)
        audio_tokens = torch.randn(1, 4, self.audio_dim)

        video_only = head(video_tokens=video_tokens)
        both = head(video_tokens=video_tokens, audio_tokens=audio_tokens)

        assert not torch.allclose(video_only, both)

    def test_accepts_input_in_a_different_dtype_than_the_weights(self):
        # A bf16 head fed fp32 connector output must not raise: forward casts the inputs.
        head = self.get_head().to(torch.bfloat16)

        seconds = head(video_tokens=torch.randn(1, 6, self.video_dim, dtype=torch.float32))

        assert seconds.dtype == torch.bfloat16
        assert torch.all(seconds > 0)

    def test_supports_more_than_one_pooling_query(self):
        # num_queries > 1 is not exercised by any published checkpoint (rc2 ships query_tokens [1, 256]),
        # so this pins the `pooler_hidden_dim * num_queries` flatten path that would otherwise be untested.
        torch.manual_seed(0)
        head = LTX2DurationHead(
            video_cross_attention_dim=self.video_dim,
            audio_cross_attention_dim=self.audio_dim,
            pooler_hidden_dim=8,
            num_queries=3,
            num_pooler_heads=2,
            mlp_hidden_dim=8,
        )
        assert head.mlp_hidden.in_features == 8 * 3

        seconds = head(video_tokens=torch.randn(2, 6, self.video_dim))

        assert seconds.shape == (2,)
        assert torch.all(seconds > 0)

    def test_attention_backend_is_selectable_on_the_pooler(self):
        # The pooler follows the AttentionModuleMixin + processor pattern specifically so that
        # `set_attention_backend` reaches it -- it walks `self.modules()` and skips anything that is
        # not an AttentionModuleMixin, so a plain nn.Module pooler would be silently ignored.
        head = self.get_head()
        assert isinstance(head.attention_pooler, AttentionModuleMixin)

        head.set_attention_backend("native")

        assert head.attention_pooler.processor._attention_backend == AttentionBackendName.NATIVE

    def test_raises_when_both_modalities_are_none(self):
        head = self.get_head()

        with pytest.raises(ValueError, match="at least one of"):
            head()


class LTX2DurationHeadFrameSnappingTests(unittest.TestCase):
    """
    Exercises the clamp/snap arithmetic with the model's prediction stubbed to a fixed value.

    Every expected frame count here was generated by running the reference implementation,
    `ltx_pipelines.utils.helpers.seconds_to_clamped_num_frames`, rather than derived by hand.
    """

    tokens = torch.randn(1, 4, 8)

    def get_head_predicting(self, seconds):
        """A real head whose forward returns `seconds` for any input.

        `forward` ends in `mlp_out` followed by `exp()`, so zeroing that layer's weight and setting
        its bias to `log(seconds)` pins the output without stubbing any method -- the clamp/snap
        arithmetic is still exercised through the real forward pass.
        """
        torch.manual_seed(0)
        head = LTX2DurationHead(
            video_cross_attention_dim=8,
            audio_cross_attention_dim=8,
            pooler_hidden_dim=8,
            num_queries=1,
            num_pooler_heads=2,
            mlp_hidden_dim=8,
        )
        with torch.no_grad():
            head.mlp_out.weight.zero_()
            head.mlp_out.bias.fill_(math.log(seconds))
        return head

    def predict(self, seconds, **kwargs):
        kwargs.setdefault("frame_rate", 24.0)
        kwargs.setdefault("temporal_compression_ratio", 8)
        return self.get_head_predicting(seconds).predict_num_frames(video_tokens=self.tokens, **kwargs)

    def test_the_fixture_head_predicts_exactly_what_was_asked(self):
        # Guards the helper the rest of this class relies on: if this drifts, every expected frame
        # count below would be testing the wrong prediction.
        for seconds in [0.5, 3.0, 12.75]:
            got = self.get_head_predicting(seconds)(video_tokens=self.tokens).item()
            assert got == pytest.approx(seconds, rel=1e-6), f"asked {seconds}, head returned {got}"

    def test_result_is_always_on_the_grid(self):
        for seconds, expected in [(2.3, 49), (4.0, 89), (5.7, 137), (11.9, 281), (19.99, 473)]:
            num_frames = self.predict(seconds)
            assert num_frames == expected, f"seconds={seconds}"
            assert (num_frames - 1) % 8 == 0, f"{num_frames} is off-grid for seconds={seconds}"

    def test_floors_to_the_grid(self):
        # 5.0s * 24fps = 120 frames, which floors to 113 (8 * 14 + 1).
        assert self.predict(5.0) == 113

    def test_clamps_above_max_seconds(self):
        # 60s is far past the 20s default: 20 * 24 = 480, which floors to 473.
        assert self.predict(60.0) == 473

    def test_clamps_below_min_seconds(self):
        # 0.1s is below the 1s default, so it clamps up to 24 frames. Flooring 24 gives 17, which
        # is under min_frames, so the undershoot correction snaps up to the next grid point (25).
        assert self.predict(0.1) == 25

    def test_snaps_up_when_flooring_would_undershoot_min_frames(self):
        # min_seconds=1.0 at 8fps gives min_frames=8; flooring 8 gives 1, below the minimum, so it
        # must snap *up* to 9 to honour the [min_frames, max_frames] contract.
        assert self.predict(1.0, frame_rate=8.0) == 9

    def test_clamps_before_snapping_not_after(self):
        # max_seconds=5.0 at 24fps gives max_frames=120. A 10s prediction clamps to 120 and then
        # floors to 113. Snapping first would give 233, and clamping that to 120 would leave an
        # off-grid result.
        num_frames = self.predict(10.0, max_seconds=5.0)
        assert num_frames == 113
        assert (num_frames - 1) % 8 == 0

    def test_honours_a_non_default_compression_ratio(self):
        # A ratio=4 grid is 4k+1, so 120 frames floors to 117 rather than 113.
        assert self.predict(5.0, temporal_compression_ratio=4) == 117

    def test_uses_the_nearest_grid_point_when_bounds_contain_none(self):
        # min<max in seconds can still round to a frame window with no valid count in it: at 24 fps
        # [1.0s, 1.02s] gives [24, 24], and 24 is not on the 8k+1 grid. The neighbours are 17 and 25;
        # 25 is nearer to the requested 24, so it wins even though it is above max_frames.
        num_frames = self.predict(1.0, min_seconds=1.0, max_seconds=1.02)

        assert num_frames == 25
        assert (num_frames - 1) % 8 == 0

    def test_prefers_the_lower_grid_point_when_it_is_nearer(self):
        # [0.75s, 0.79s] at 24 fps gives [18, 19]; the neighbours are 17 and 25, and 17 is nearer to
        # the requested 19. Guards against always rounding up.
        num_frames = self.predict(0.79, min_seconds=0.75, max_seconds=0.79)

        assert num_frames == 17
        assert (num_frames - 1) % 8 == 0

    def test_warns_when_bounds_contain_no_grid_point(self):
        with self.assertLogs("diffusers.pipelines.ltx2.duration_head", level="WARNING") as logs:
            self.predict(1.0, min_seconds=1.0, max_seconds=1.02)

        assert any("admit no frame count" in line for line in logs.output)

    def test_raises_on_a_multi_item_batch(self):
        head = self.get_head_predicting(3.0)

        with pytest.raises(ValueError, match="single prediction"):
            head.predict_num_frames(video_tokens=torch.randn(2, 4, 8), frame_rate=24.0, temporal_compression_ratio=8)


class LTX2AutoDurationTests(unittest.TestCase):
    def test_defaults_match_the_reference(self):
        bounds = LTX2AutoDuration()

        assert bounds.min_seconds == 1.0
        assert bounds.max_seconds == 20.0

    def test_raises_when_min_exceeds_max(self):
        with pytest.raises(ValueError, match="must be less than"):
            LTX2AutoDuration(min_seconds=10.0, max_seconds=2.0)

    def test_raises_when_min_equals_max(self):
        # A collapsed range is unsatisfiable: at 24 fps it pins the frame count to 24, which is not
        # on the 8k+1 grid, so it would produce a count the VAE rejects.
        with pytest.raises(ValueError, match="must be less than"):
            LTX2AutoDuration(min_seconds=1.0, max_seconds=1.0)
