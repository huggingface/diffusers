# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


imageio = pytest.importorskip("imageio.v2")
pytest.importorskip("imageio_ffmpeg")

from examples.community.hy_nomas_keyboard_overlay import (  # noqa: E402
    KEY_LAYOUT,
    KeyboardTimeline,
    draw_keyboard_overlay,
    export_keyboard_video,
)


def make_turn(events=None, num_frames=60, fps=60, duration_ms=1000):
    return {
        "num_frames": num_frames,
        "video_fps": fps,
        "camera_trajectory": {"durationMs": duration_ms, "keyEvents": events or []},
    }


def test_subframe_events_simultaneous_keys_and_release():
    timeline = KeyboardTimeline(
        [
            make_turn(
                [
                    {"t": 50, "code": "KeyW", "pressed": False},
                    {"t": 16.3, "code": "KeyW", "pressed": True},
                    {"t": 16.3, "code": "ArrowUp", "pressed": True},
                    {"t": 66.7, "code": "ArrowUp", "pressed": False},
                ]
            )
        ]
    )
    assert timeline.state_at_frame(0) == (set(), "RECORDED KEY EVENTS")
    assert timeline.state_at_frame(1)[0] == {"KeyW", "ArrowUp"}
    assert timeline.state_at_frame(2)[0] == {"KeyW", "ArrowUp"}
    assert timeline.state_at_frame(3)[0] == {"ArrowUp"}
    assert timeline.state_at_frame(4)[0] == {"ArrowUp"}
    assert timeline.state_at_frame(5)[0] == set()


def test_recording_duration_maps_to_video_clock_and_turn_boundary_resets():
    first = make_turn([{"t": 250, "code": "ArrowDown", "pressed": True}], num_frames=48, fps=24)
    first["camera_trajectory"] = json.dumps(first["camera_trajectory"])
    second = make_turn(num_frames=24, fps=24)
    second["camera_trajectory"]["frames"] = [{"position": {"x": 100, "y": 0, "z": 50}}]
    timeline = KeyboardTimeline([first, second])
    assert timeline.state_at_frame(11)[0] == set()
    assert timeline.state_at_frame(12)[0] == {"ArrowDown"}
    assert timeline.state_at_frame(47)[0] == {"ArrowDown"}
    assert timeline.state_at_frame(48) == (set(), "NO KEY EVENTS")
    with pytest.raises(IndexError):
        timeline.state_at_frame(72)


def test_missing_recording_duration_uses_video_fps():
    turn = make_turn([{"t": 100, "code": "KeyD", "pressed": True}], num_frames=24, fps=24)
    del turn["camera_trajectory"]["durationMs"]
    timeline = KeyboardTimeline([turn])
    assert timeline.state_at_frame(2)[0] == set()
    assert timeline.state_at_frame(3)[0] == {"KeyD"}


@pytest.mark.parametrize("code", KEY_LAYOUT)
def test_each_key_highlights_only_its_own_box_without_modifying_input(code):
    frame = np.full((352, 640, 3), 100, dtype=np.uint8)
    original = frame.copy()
    events = [{"t": 0, "code": code, "pressed": True}, {"t": 500, "code": code, "pressed": False}]
    timeline = KeyboardTimeline([make_turn(events)])
    active = draw_keyboard_overlay(frame, *timeline.state_at_frame(0))
    idle = draw_keyboard_overlay(frame, *timeline.state_at_frame(30))
    difference = np.any(active != idle, axis=-1)
    column, row = KEY_LAYOUT[code]
    x, y = 8 + 10 + column * 28, 352 - 112 - 8 + 28 + row * 28
    assert difference[y : y + 25, x : x + 25].sum() > 300
    difference[y : y + 25, x : x + 25] = False
    assert not difference.any()
    np.testing.assert_array_equal(frame, original)
    np.testing.assert_array_equal(active[:200], frame[:200])


def test_four_arrows_are_distinct_vector_glyphs():
    frame = draw_keyboard_overlay(np.zeros((352, 640, 3), dtype=np.uint8), set(), "NO KEY EVENTS")
    glyphs = []
    for code in ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]:
        column, row = KEY_LAYOUT[code]
        x, y = 8 + 10 + column * 28, 352 - 112 - 8 + 28 + row * 28
        glyph = frame[y + 5 : y + 20, x + 5 : x + 20, 0] > 150
        assert glyph.sum() > 40
        glyphs.append(glyph)
    assert len({glyph.tobytes() for glyph in glyphs}) == 4


@pytest.mark.parametrize(
    "event",
    [
        {"t": -1, "code": "KeyW", "pressed": True},
        {"t": float("nan"), "code": "KeyW", "pressed": True},
        {"t": 0, "code": "KeyW", "pressed": "false"},
        {"t": 0, "code": "Space", "pressed": True},
    ],
)
def test_invalid_recorded_events_raise(event):
    with pytest.raises(ValueError):
        KeyboardTimeline([make_turn([event])])


@pytest.fixture
def original_video(tmp_path):
    video = tmp_path / "original.mp4"
    with imageio.get_writer(str(video), fps=4, macro_block_size=1, codec="libx264") as writer:
        for _ in range(4):
            writer.append_data(np.full((352, 640, 3), 100, dtype=np.uint8))
    return video


def test_cli_exports_real_events_to_separate_video(original_video, tmp_path):
    metadata = tmp_path / "replay.json"
    turn = make_turn(
        [{"t": 250, "code": "ArrowDown", "pressed": True}, {"t": 750, "code": "ArrowDown", "pressed": False}],
        num_frames=4,
        fps=4,
    )
    metadata.write_text(json.dumps({"sessions": [{"turns": [{"params": turn}]}]}))
    output = tmp_path / "keyboard.mp4"
    original_sha256 = hashlib.sha256(original_video.read_bytes()).hexdigest()
    script = Path(__file__).resolve().parents[2] / "examples/community/hy_nomas_keyboard_overlay.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--video",
            str(original_video),
            "--manifest",
            str(metadata),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["num_frames"] == 4
    assert report["fps"] == 4
    assert report["recorded_key_events"] == 2
    assert hashlib.sha256(original_video.read_bytes()).hexdigest() == original_sha256
    with imageio.get_reader(str(output)) as reader:
        assert reader.count_frames() == 4
        assert reader.get_meta_data()["fps"] == 4
        assert reader.get_meta_data()["size"] == (640, 352)
        green_pixel_counts = []
        for frame in reader:
            frame = np.asarray(frame)
            green_pixel_counts.append(((frame[..., 1] > 180) & (frame[..., 0] < 120)).sum())
    assert green_pixel_counts[0] < 10
    assert green_pixel_counts[1] > 200
    assert green_pixel_counts[2] > 200
    assert green_pixel_counts[3] < 10


def test_camera_json_without_events_stays_idle(original_video, tmp_path):
    metadata = tmp_path / "camera.json"
    metadata.write_text(json.dumps({"durationMs": 1000, "frames": [{"position": [1, 2, 3]}]}))
    report = export_keyboard_video(original_video, metadata, tmp_path / "idle.mp4")
    assert report["recorded_key_events"] == 0
    assert report["segments"][0]["key_source"] == "none"


def test_export_rejects_mismatched_manifest_before_creating_output(original_video, tmp_path):
    metadata = tmp_path / "manifest.json"
    metadata.write_text(json.dumps({"sessions": [{"turns": [{"params": make_turn()}]}]}))
    output = tmp_path / "mismatched.mp4"
    with pytest.raises(ValueError, match="does not match the manifest"):
        export_keyboard_video(original_video, metadata, output)
    assert not output.exists()


def test_export_never_overwrites_original_or_existing_output(tmp_path):
    original = tmp_path / "original.mp4"
    original.write_bytes(b"original")
    alias = tmp_path / "alias.mp4"
    alias.symlink_to(original)
    with pytest.raises(ValueError, match="different file"):
        export_keyboard_video(original, tmp_path / "unused.json", alias)
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        export_keyboard_video(original, tmp_path / "unused.json", output)
    assert original.read_bytes() == b"original"
    assert output.read_bytes() == b"existing"
