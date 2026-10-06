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

"""Colour transforms and HDR export utilities of the LTX-2.5 SDR-To-HDR IC-LoRA.

Expected values were computed independently in float64 with colour-science 0.4.4 (`log_encoding_ACEScct`,
`log_decoding_ACEScct`, `cctf_decoding(..., function="sRGB")`, `matrix_RGB_to_RGB(..., "Bradford")`,
`oetf_BT2100_HLG` / `oetf_inverse_BT2100_HLG`), not with the code under test.
"""

import numpy as np
import PIL.Image
import pytest
import torch

from diffusers.pipelines.ltx2.image_processor import (
    ACESCCT_A,
    ACESCCT_B,
    ACESCG_TO_REC709,
    ACESCG_TO_REC2020,
    REC709_TO_ACESCG,
    REC709_TO_REC2020,
    LTX2VideoHDRProcessor,
    acescct_decode,
    acescct_encode,
    acescct_to_linear,
    apply_primaries_matrix,
    srgb_eotf_to_linear,
    to_acescct,
)
from diffusers.utils import is_av_available, is_openexr_available


# colour-science 0.4.4, float64: `colour.matrix_RGB_to_RGB(src, dst, chromatic_adaptation_transform="Bradford")`.
COLOUR_ACESCG_TO_REC709 = [
    [1.7050509926579835, -0.6217921206570048, -0.08325887200097871],
    [-0.13025641750704345, 1.1408047365754013, -0.010548319068358038],
    [-0.024003356804618025, -0.12896897606497054, 1.1529723328695887],
]
COLOUR_REC709_TO_ACESCG = [
    [0.6130974024, 0.3395231462, 0.0473794514],
    [0.0701937225, 0.9163538791, 0.0134523985],
    [0.0206155929, 0.1095697729, 0.8698146342],
]
COLOUR_REC709_TO_REC2020 = [
    [0.6274038959, 0.3292830384, 0.0433130657],
    [0.0690972894, 0.9195403951, 0.0113623156],
    [0.0163914389, 0.0880133079, 0.8955952532],
]
COLOUR_ACESCG_TO_REC2020 = [
    [1.0258247477, -0.0200531908, -0.0057715568],
    [-0.0022343695, 1.0045865019, -0.0023521324],
    [-0.0050133515, -0.0252900718, 1.0303034233],
]


def _logc3_decompress_reference(t: float) -> float:
    # ARRI LogC3 EI 800 decoding, written out independently of the processor.
    a, b, c, d, e, f, cut = 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809, 0.010591
    if t >= e * cut + f:
        return (10.0 ** ((t - d) / c) - b) / a
    return (t - f) / e


class TestACEScct:
    def test_encode_known_values(self):
        linear = torch.tensor([0.0, 0.0078125, 0.18, 1.0, 16.0, 222.0], dtype=torch.float64)
        expected = torch.tensor(
            [
                0.0729055341958355,
                0.1552511415525113,
                0.4135884024924423,
                0.5547945205479452,
                0.7831050228310503,
                0.9996812709103942,
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(acescct_encode(linear), expected, rtol=0, atol=1e-12)

    def test_encode_clamps_like_the_reference(self):
        # Negative inputs are clamped to 0 and the output is clamped to [0, 1] (the ACEScct spec does neither).
        out = acescct_encode(torch.tensor([-1.0, 1000.0], dtype=torch.float64))
        torch.testing.assert_close(out, torch.tensor([ACESCCT_B, 1.0], dtype=torch.float64), rtol=0, atol=1e-12)

    def test_decode_known_values(self):
        codes = torch.tensor(
            [0.0729055341958355, 0.155251141552511, 0.4135884024924423, 1.0, 0.0, 0.5], dtype=torch.float64
        )
        expected = torch.tensor(
            [0.0, 0.0078125, 0.18, 222.8609442038076, -0.006916877586898862, 0.5140569133280329],
            dtype=torch.float64,
        )
        torch.testing.assert_close(acescct_decode(codes), expected, rtol=1e-12, atol=1e-12)

    def test_decode_clamps_input(self):
        out = acescct_decode(torch.tensor([-0.5, 1.5], dtype=torch.float64))
        expected = torch.tensor([-ACESCCT_B / ACESCCT_A, 222.8609442038076], dtype=torch.float64)
        torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)

    def test_round_trip_linear(self):
        linear = torch.cat([torch.linspace(0.0, 0.01, 101), torch.logspace(-2, np.log10(222.0), 200)]).double()
        torch.testing.assert_close(acescct_decode(acescct_encode(linear)), linear, rtol=1e-12, atol=1e-15)

    def test_round_trip_codes(self):
        # The valid code domain starts at the code of linear 0; codes below decode to negative linear values,
        # which the encoder clamps.
        codes = torch.linspace(ACESCCT_B, 1.0, 1001, dtype=torch.float64)
        torch.testing.assert_close(acescct_encode(acescct_decode(codes)), codes, rtol=0, atol=1e-12)


class TestPrimaries:
    @pytest.mark.parametrize(
        "matrix, expected",
        [
            (ACESCG_TO_REC709, COLOUR_ACESCG_TO_REC709),
            (REC709_TO_ACESCG, COLOUR_REC709_TO_ACESCG),
            (REC709_TO_REC2020, COLOUR_REC709_TO_REC2020),
            (ACESCG_TO_REC2020, COLOUR_ACESCG_TO_REC2020),
        ],
    )
    def test_matrix_values(self, matrix, expected):
        # The stored matrices are float32, as in the reference.
        torch.testing.assert_close(
            torch.tensor(matrix, dtype=torch.float64), torch.tensor(expected, dtype=torch.float64), rtol=0, atol=1e-7
        )

    @pytest.mark.parametrize("matrix", [ACESCG_TO_REC709, REC709_TO_ACESCG, REC709_TO_REC2020, ACESCG_TO_REC2020])
    def test_white_is_preserved(self, matrix):
        row_sums = torch.tensor(matrix, dtype=torch.float64).sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(3, dtype=torch.float64), rtol=0, atol=1e-6)

    def test_rec709_acescg_are_inverse(self):
        product = torch.tensor(REC709_TO_ACESCG, dtype=torch.float64) @ torch.tensor(
            ACESCG_TO_REC709, dtype=torch.float64
        )
        torch.testing.assert_close(product, torch.eye(3, dtype=torch.float64), rtol=0, atol=1e-6)

    def test_apply_primaries_matrix_layouts(self):
        generator = torch.Generator().manual_seed(0)
        video = torch.rand(2, 3, 4, 5, 6, generator=generator)
        expected = torch.einsum("dc,bcfhw->bdfhw", torch.tensor(REC709_TO_ACESCG), video)
        torch.testing.assert_close(apply_primaries_matrix(video, REC709_TO_ACESCG), expected)
        # Non-5D inputs use `dim=-3` as the channel axis.
        frames = video[0].permute(1, 0, 2, 3)  # (F, C, H, W)
        torch.testing.assert_close(apply_primaries_matrix(frames, REC709_TO_ACESCG), expected[0].permute(1, 0, 2, 3))


class TestInputTransform:
    # Neutral greys, a saturated red and an arbitrary colour, as (N, 3, 1, 1) images.
    pixels = torch.tensor([[0, 0, 0], [0.04045] * 3, [0.5] * 3, [1, 1, 1], [1.0, 0.0, 0.0], [0.2, 0.6, 0.9]])

    def _as_images(self, values):
        return torch.as_tensor(values, dtype=torch.float32)[:, :, None, None]

    def test_srgb_eotf(self):
        out = srgb_eotf_to_linear(torch.tensor([0.0, 0.04045, 0.5, 1.0, -0.2, 1.3]))
        expected = torch.tensor([0.0, 0.0031308072830676845, 0.21404114048223255, 1.0, 0.0, 1.0])
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, expected, rtol=0, atol=1e-7)

    def test_srgb_gamma(self):
        expected = [
            [0.0729055341958355] * 3,
            [0.10590498888079533, 0.10590498734413856, 0.1059049870368072],
            [0.4278516036650092, 0.42785159983049315, 0.42785159906358994],
            [0.5547945245358419, 0.5547945207013258, 0.5547945199344226],
            [0.5145084623593209, 0.3360437118624257, 0.23515295334335573],
            [0.4068006341802645, 0.45696458686011593, 0.5277994813544838],
        ]
        out = to_acescct(self._as_images(self.pixels), "srgb_gamma")
        torch.testing.assert_close(out, self._as_images(expected), rtol=0, atol=1e-6)

    def test_srgb_linear(self):
        expected = [
            [0.0729055341958355] * 3,
            [0.2906554556363287, 0.2906554518018126, 0.2906554510349094],
            [0.49771689896506566, 0.4977168951305496, 0.49771689436364636],
            [0.5547945245358419, 0.5547945207013258, 0.5547945199344226],
            [0.5145084623593209, 0.3360437118624257, 0.23515295334335573],
            [0.47269375324230106, 0.509362790853311, 0.5416727756641857],
        ]
        out = to_acescct(self._as_images(self.pixels), "srgb")
        torch.testing.assert_close(out, self._as_images(expected), rtol=0, atol=1e-6)

    def test_acescg_and_acescct(self):
        linear = torch.tensor([0.0, 0.18, 1.0, 16.0, -1.0, 1000.0]).view(2, 3, 1, 1)
        expected = torch.tensor(
            [0.0729055341958355, 0.4135884024924423, 0.5547945205479452, 0.7831050228310503, ACESCCT_B, 1.0]
        ).view(2, 3, 1, 1)
        torch.testing.assert_close(to_acescct(linear, "acescg"), expected, rtol=0, atol=1e-6)
        codes = torch.tensor([-0.5, 0.25, 0.5, 0.75, 1.0, 2.0]).view(2, 3, 1, 1)
        torch.testing.assert_close(to_acescct(codes, "acescct"), codes.clamp(0.0, 1.0), rtol=0, atol=0)

    def test_invalid_colorspace(self):
        with pytest.raises(ValueError, match="Unsupported input colorspace"):
            to_acescct(torch.zeros(1, 3, 1, 1), "rec2020")


class TestOutputTransform:
    codes = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.7, 0.4, 0.2], [1.0, 1.0, 1.0]])[:, :, None, None]

    def test_acescg(self):
        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # the negative toe (-0.0069169) is clamped to 0
                [0.5140569133280329] * 3,
                [5.83203751526631, 0.1526183140836417, 0.013452331067795364],
                [222.8609442038076] * 3,
            ]
        )[:, :, None, None]
        torch.testing.assert_close(acescct_to_linear(self.codes, "acescg"), expected, rtol=1e-6, atol=1e-7)

    def test_rec709(self):
        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5140568788578308, 0.5140569310418868, 0.5140569209880779],
                [9.847904184623355, 0.0, 0.0],  # out-of-gamut values are clipped after the matrix
                [222.8609292598168, 222.86095188335844, 222.86094752469444],
            ]
        )[:, :, None, None]
        torch.testing.assert_close(acescct_to_linear(self.codes, "rec709"), expected, rtol=1e-6, atol=1e-7)

    def test_round_trip_through_working_space(self):
        # sRGB code -> ACEScct -> Rec.709 linear equals the sRGB EOTF on in-gamut colours.
        generator = torch.Generator().manual_seed(0)
        srgb = torch.rand(1, 3, 2, 8, 8, generator=generator)
        linear = acescct_to_linear(to_acescct(srgb, "srgb_gamma"), "rec709")
        torch.testing.assert_close(linear, srgb_eotf_to_linear(srgb), rtol=1e-4, atol=1e-6)

    def test_invalid_colorspace(self):
        with pytest.raises(ValueError, match="Unsupported output colorspace"):
            acescct_to_linear(self.codes, "acescct")


class TestLTX2VideoHDRProcessor:
    def test_logc3_is_the_default_and_unchanged(self):
        processor = LTX2VideoHDRProcessor()
        assert processor.config.hdr_transform == "logc3"

        # Preprocessing: plain [-1, 1] normalization of the sRGB codes, then reflect-padding.
        video = (np.arange(2 * 32 * 64 * 3) % 251).astype(np.uint8).reshape(2, 32, 64, 3)
        out = processor.preprocess_reference_video_hdr([PIL.Image.fromarray(frame) for frame in video], 32, 80)
        expected = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0  # (F, C, H, W)
        expected = torch.nn.functional.pad(expected, (0, 16, 0, 0), mode="reflect")
        torch.testing.assert_close(out, expected.permute(1, 0, 2, 3)[None])

        # Postprocessing: LogC3 decoding of the [0, 1] codes, no primaries conversion, channels-last.
        codes = torch.tensor([0.0, 0.05, 0.391007, 0.6, 0.9, 1.0])
        decoded = processor.postprocess_hdr_video((codes * 2.0 - 1.0).view(1, 3, 2, 1, 1), output_type="pt")
        expected = torch.tensor([_logc3_decompress_reference(float(t)) for t in codes]).view(1, 3, 2, 1, 1)
        assert decoded.shape == (1, 2, 1, 1, 3)
        torch.testing.assert_close(decoded, expected.permute(0, 2, 3, 4, 1), rtol=1e-5, atol=1e-6)

    def test_logc3_rejects_colorspace_options(self):
        processor = LTX2VideoHDRProcessor()
        with pytest.raises(ValueError, match="input_colorspace"):
            processor.preprocess_reference_video_hdr(np.zeros((1, 4, 4, 3), dtype=np.uint8), 4, 4, "srgb")
        with pytest.raises(ValueError, match="output_colorspace"):
            processor.postprocess_hdr_video(torch.zeros(1, 3, 1, 4, 4), "pt", "rec709")

    def test_invalid_transform(self):
        with pytest.raises(ValueError, match="Unsupported HDR transform"):
            LTX2VideoHDRProcessor(hdr_transform="pq")

    def test_acescct_preprocess_srgb_gamma(self):
        processor = LTX2VideoHDRProcessor(hdr_transform="acescct")
        # Black and white 8-bit frames: sRGB 0 -> ACEScct 0.0729055 -> -0.8541889, sRGB 1 -> 0.5547945 -> 0.1095890.
        video = np.zeros((2, 20, 24, 3), dtype=np.uint8)
        video[1] = 255
        out = processor.preprocess_reference_video_hdr(video, 32, 32)
        assert out.shape == (1, 3, 2, 32, 32)
        torch.testing.assert_close(out[:, :, 0], torch.full((1, 3, 32, 32), -0.854188931608329), rtol=0, atol=1e-6)
        torch.testing.assert_close(out[:, :, 1], torch.full((1, 3, 32, 32), 0.10958904109589), rtol=0, atol=1e-6)

    def test_acescct_preprocess_matches_functional_transform(self):
        processor = LTX2VideoHDRProcessor(hdr_transform="acescct")
        generator = torch.Generator().manual_seed(0)
        # Integer input: transform, then reflect-pad.
        frames = (torch.rand(3, 3, 20, 24, generator=generator) * 255).to(torch.uint8)  # (F, C, H, W)
        out = processor.preprocess_reference_video_hdr(frames, 32, 32, input_colorspace="srgb_gamma")
        working = to_acescct(frames.permute(1, 0, 2, 3)[None].float() / 255.0, "srgb_gamma")
        expected = processor._resize_and_reflect_pad_video(working, 32, 32) * 2.0 - 1.0
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        # Float input (e.g. an EXR plate, values above 1): reflect-pad, then transform.
        linear = torch.rand(1, 3, 20, 24, 3, generator=generator).numpy() * 50.0  # (B, F, H, W, C)
        out = processor.preprocess_reference_video_hdr(linear, 32, 32, input_colorspace="acescg")
        padded = processor._resize_and_reflect_pad_video(torch.from_numpy(linear).permute(0, 4, 1, 2, 3), 32, 32)
        torch.testing.assert_close(out, to_acescct(padded, "acescg") * 2.0 - 1.0, rtol=0, atol=0)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_acescct_postprocess(self):
        processor = LTX2VideoHDRProcessor(hdr_transform="acescct")
        codes = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.7, 0.4, 0.2], [1.0, 1.0, 1.0]])
        decoded = (codes * 2.0 - 1.0).T.reshape(1, 3, 4, 1, 1)  # (B, C, F, H, W) in [-1, 1]

        out = processor.postprocess_hdr_video(decoded, output_type="pt")
        assert out.shape == (1, 4, 1, 1, 3)
        expected = acescct_to_linear(codes.T.reshape(1, 3, 4, 1, 1), "rec709").permute(0, 2, 3, 4, 1)
        torch.testing.assert_close(out, expected)

        out = processor.postprocess_hdr_video(decoded, output_type="np", output_colorspace="acescg")
        assert isinstance(out, np.ndarray) and out.dtype == np.float32
        np.testing.assert_allclose(out[0, 2, 0, 0], [5.83203751526631, 0.1526183140836417, 0.013452331067795364], 1e-6)

        out = processor.postprocess_hdr_video(decoded, output_type="pt", output_colorspace="acescct")
        torch.testing.assert_close(out[0, :, 0, 0], codes, rtol=0, atol=1e-7)


class TestHLG:
    @pytest.fixture(autouse=True)
    def _export_utils(self):
        if not is_av_available():
            pytest.skip("PyAV is not installed.")
        from diffusers.pipelines.ltx2 import export_utils

        self.export_utils = export_utils

    def test_inverse_oetf_and_rolloff(self):
        white_x = self.export_utils._hlg_inverse_oetf(0.75)
        assert white_x == pytest.approx(0.26496256042100724, abs=1e-15)
        assert white_x / (1.0 - white_x) == pytest.approx(0.36047491753994165, abs=1e-15)
        assert self.export_utils._hlg_inverse_oetf(0.5) == pytest.approx(1.0 / 12.0, abs=1e-15)
        assert self.export_utils._hlg_inverse_oetf(0.3) == pytest.approx(0.03, abs=1e-15)

    def test_signal_known_values(self):
        white_x = self.export_utils._hlg_inverse_oetf(0.75)
        identity = torch.eye(3)
        # Neutral Rec.709 linear 0.18 / 1.0 / 4.0 / 100.0 (Rec.709 -> Rec.2020 keeps neutrals neutral).
        linear = torch.tensor([0.18, 1.0, 4.0, 100.0]).view(4, 1, 1, 1).expand(4, 3, 1, 1)
        signal = self.export_utils._linear_to_hlg_signal(
            linear, torch.tensor(REC709_TO_REC2020), white_x, white_x / (1.0 - white_x)
        )
        expected = torch.tensor([0.3782588830779046, 0.75, 0.947280748538359, 0.9999999950661305])
        torch.testing.assert_close(signal[:, 0, 0, 0], expected, rtol=0, atol=2e-6)
        # A coloured pixel goes through the Rec.709 -> Rec.2020 matrix first.
        signal = self.export_utils._linear_to_hlg_signal(
            torch.tensor([2.0, 0.5, 0.05]).view(1, 3, 1, 1),
            torch.tensor(REC709_TO_REC2020),
            white_x,
            white_x / (1.0 - white_x),
        )
        expected = torch.tensor([0.8139141545892317, 0.6460072658555785, 0.3108599919641505])
        torch.testing.assert_close(signal[0, :, 0, 0], expected, rtol=0, atol=2e-6)
        # Negative and NaN values are mapped to 0.
        bad = torch.tensor([-1.0, float("nan"), float("-inf")]).view(1, 3, 1, 1)
        signal = self.export_utils._linear_to_hlg_signal(bad, identity, white_x, 1.0)
        assert torch.equal(signal, torch.zeros_like(signal))

    def test_yuv_code_levels(self):
        rgb = torch.zeros(2, 3, 2, 2)
        rgb[1] = 1.0
        y, u, v = self.export_utils._rgb_to_yuv420p10_bt2020_limited(rgb)
        assert y.shape == (2, 2, 2) and u.shape == (2, 1, 1) and v.shape == (2, 1, 1)
        assert y[0].unique().tolist() == [64] and y[1].unique().tolist() == [940]
        assert u.unique().tolist() == [512] and v.unique().tolist() == [512]
        # Pure Rec.2020 red: Y = 0.2627, Cb = -0.5 * 0.2627 / (1 - 0.0593), Cr = 0.5.
        red = torch.zeros(1, 3, 2, 2)
        red[:, 0] = 1.0
        y, u, v = self.export_utils._rgb_to_yuv420p10_bt2020_limited(red)
        assert y.unique().tolist() == [round((219 * 0.2627 + 16) * 4)]
        assert u.unique().tolist() == [round((224 * -0.5 * 0.2627 / (1 - 0.0593) + 128) * 4)]
        assert v.unique().tolist() == [round((224 * 0.5 + 128) * 4)]

    def test_encode_hlg_mp4(self, tmp_path):
        import av

        if "libx265" not in av.codecs_available:
            pytest.skip("The FFmpeg build used by PyAV does not include libx265.")

        # Diffuse white (linear 1.0) maps to the HLG signal 0.75 -> Y = round((219 * 0.75 + 16) * 4) = 721.
        frames = torch.ones(9, 64, 96, 3)
        output = tmp_path / "hlg.mp4"
        self.export_utils.encode_hdr_tensor_to_hlg_mp4(frames, output, frame_rate=24.0, thread_count=1)

        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            context = stream.codec_context
            assert context.name == "hevc"
            assert context.pix_fmt == "yuv420p10le"
            assert stream.codec_tag == "hvc1"
            assert (stream.width, stream.height) == (96, 64)
            assert stream.average_rate == 24
            assert context.color_primaries == 9  # BT.2020
            assert context.color_trc == 18  # ARIB STD-B67 (HLG)
            assert context.colorspace == 9  # BT.2020 NCL
            assert context.color_range == 1  # limited
            decoded = list(container.decode(video=0))
        assert len(decoded) == 9
        y_plane = decoded[0].planes[0]
        y = np.frombuffer(y_plane, dtype=np.uint16).reshape(y_plane.height, y_plane.line_size // 2)[:, :96]
        assert abs(int(np.median(y)) - 721) <= 2

    def test_encode_hlg_mp4_rejects_odd_sizes(self, tmp_path):
        import av

        if "libx265" not in av.codecs_available:
            pytest.skip("The FFmpeg build used by PyAV does not include libx265.")
        with pytest.raises(ValueError, match="even"):
            self.export_utils.encode_hdr_tensor_to_hlg_mp4(torch.ones(1, 63, 64, 3), tmp_path / "x.mp4", 24.0)
        assert not (tmp_path / "x.mp4").exists()


class TestEXR:
    @pytest.fixture(autouse=True)
    def _export_utils(self):
        if not is_av_available():
            pytest.skip("PyAV is not installed.")
        if not is_openexr_available():
            pytest.skip("OpenEXR is not installed.")
        from diffusers.pipelines.ltx2 import export_utils

        self.export_utils = export_utils

    @staticmethod
    def _read(path):
        import OpenEXR

        with OpenEXR.File(str(path), separate_channels=True) as exr_file:
            header = dict(exr_file.header())
            channels = {name: channel.pixels.copy() for name, channel in exr_file.channels().items()}
        return header, channels

    @pytest.mark.parametrize(
        "exr_colorspace, chromaticities, color_space",
        [
            ("acescg", (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767), "ACEScg"),
            ("acescct", (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767), "ACEScct"),
            ("srgb_linear", (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290), "sRGB"),
        ],
    )
    def test_export_sequence_round_trip(self, tmp_path, exr_colorspace, chromaticities, color_space):
        import OpenEXR

        generator = torch.Generator().manual_seed(0)
        frames = torch.rand(2, 6, 10, 3, generator=generator) * 100.0
        paths = self.export_utils.export_to_exr_sequence(frames, tmp_path / "exr", exr_colorspace=exr_colorspace)
        assert [p.split("/")[-1] for p in paths] == ["frame_00000.exr", "frame_00001.exr"]

        for path, frame in zip(paths, frames):
            header, channels = self._read(path)
            assert set(channels) == {"R", "G", "B"}
            assert header["compression"] == OpenEXR.ZIP_COMPRESSION
            assert header["colorSpace"] == color_space
            np.testing.assert_allclose(header["chromaticities"], chromaticities, rtol=0, atol=1e-7)
            for index, name in enumerate("RGB"):
                assert channels[name].dtype == np.float16
                assert channels[name].shape == (6, 10)
                np.testing.assert_array_equal(channels[name], frame[..., index].numpy().astype(np.float16))

    def test_save_frame_channels_first_and_full_float(self, tmp_path):
        frame = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2) * 1e3  # (C, H, W)
        self.export_utils.save_exr_frame(frame, tmp_path / "frame.exr", primaries="acescg", half=False)
        header, channels = self._read(tmp_path / "frame.exr")
        assert header["colorSpace"] == "sRGB"
        for index, name in enumerate("RGB"):
            assert channels[name].dtype == np.float32
            np.testing.assert_array_equal(channels[name], frame[index].numpy())
