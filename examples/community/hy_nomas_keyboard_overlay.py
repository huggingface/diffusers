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

"""Display recorded Hy-Nomas keyboard events on an exported, silent video.

Install the example dependencies with `pip install numpy pillow imageio imageio-ffmpeg`.
The input JSON can be a replay manifest or a camera trajectory containing `keyEvents`.
No model weights or reference repository are needed.
"""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


KEY_LAYOUT = {
    "KeyW": (1, 0),
    "KeyA": (0, 1),
    "KeyS": (1, 1),
    "KeyD": (2, 1),
    "ArrowUp": (5, 0),
    "ArrowLeft": (4, 1),
    "ArrowDown": (5, 1),
    "ArrowRight": (6, 1),
}


class KeyboardTimeline:
    """Map concatenated turn frames to recorded key states.

    Each turn is a replay manifest's `params` dictionary with `num_frames`, `video_fps`, and optional
    `camera_trajectory` (a dictionary or JSON string). Event `t` values are milliseconds. Like camera playback,
    the recording's `durationMs` is mapped to the turn duration; without it, timestamps use the turn's video clock.
    Key states reset at every turn boundary. Missing events never imply keys from camera motion.
    """

    def __init__(self, turns: list[dict]):
        if not turns:
            raise ValueError("At least one turn is required.")
        self.fps = float(turns[0]["video_fps"])
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("video_fps must be finite and positive.")
        self.segments = []
        self.num_frames = 0
        for turn in turns:
            num_frames = turn["num_frames"]
            if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
                raise ValueError("num_frames must be a positive integer.")
            if not math.isclose(float(turn["video_fps"]), self.fps, rel_tol=1e-6):
                raise ValueError("All turns must use the same video_fps.")
            if turn.get("stable_prefix_carry_frames", 0):
                raise ValueError("Use turns without duplicated stable-prefix carry frames.")
            trajectory = turn.get("camera_trajectory") or {}
            if isinstance(trajectory, str):
                trajectory = json.loads(trajectory)
            duration_ms = float(trajectory.get("durationMs", num_frames * 1000 / self.fps))
            if not math.isfinite(duration_ms) or duration_ms <= 0:
                raise ValueError("durationMs must be finite and positive.")
            events = trajectory.get("keyEvents", [])
            for event in events:
                if event["code"] not in KEY_LAYOUT:
                    raise ValueError(f"Unsupported key code: {event['code']}")
                if not isinstance(event["pressed"], bool):
                    raise ValueError("Each key event's pressed value must be a boolean.")
                if not math.isfinite(float(event["t"])) or float(event["t"]) < 0:
                    raise ValueError("Each key event's t must be finite, nonnegative milliseconds.")
            self.segments.append(
                {
                    "start_frame": self.num_frames,
                    "end_frame": self.num_frames + num_frames,
                    "duration_ms": duration_ms,
                    "events": sorted(events, key=lambda event: float(event["t"])),
                }
            )
            self.num_frames += num_frames

    def state_at_frame(self, frame_index: int) -> tuple[set[str], str]:
        """Return pressed key codes and a provenance label for a zero-based video frame."""
        if not 0 <= frame_index < self.num_frames:
            raise IndexError("Frame index is outside the keyboard timeline.")
        for segment in self.segments:
            if frame_index < segment["end_frame"]:
                local_frame = frame_index - segment["start_frame"]
                time_ms = local_frame * segment["duration_ms"] / (segment["end_frame"] - segment["start_frame"])
                pressed = set()
                for event in segment["events"]:
                    if float(event["t"]) > time_ms:
                        break
                    if event["pressed"]:
                        pressed.add(event["code"])
                    else:
                        pressed.discard(event["code"])
                label = "RECORDED KEY EVENTS" if segment["events"] else "NO KEY EVENTS"
                return pressed, label


def draw_keyboard_overlay(frame: np.ndarray, pressed_keys: set[str], label: str) -> np.ndarray:
    """Return an RGB uint8 copy with WASD and vector arrows; the input frame is preserved."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("frame must be an RGB uint8 array with shape (height, width, 3).")
    panel = Image.new("RGBA", (216, 112))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.rounded_rectangle((0, 0, 215, 111), radius=7, fill=(12, 18, 30, 225))
    draw.text((10, 8), "MOVE", font=font, fill=(221, 230, 242))
    draw.text((122, 8), "LOOK", font=font, fill=(221, 230, 242))
    for code, (column, row) in KEY_LAYOUT.items():
        x, y = 10 + column * 28, 28 + row * 28
        active = code in pressed_keys
        fill = (63, 226, 185, 255) if active else (47, 59, 77, 255)
        foreground = (9, 28, 28, 255) if active else (221, 230, 242, 255)
        draw.rounded_rectangle((x, y, x + 24, y + 24), radius=4, fill=fill)
        if code.startswith("Key"):
            draw.text((x + 12, y + 12), code[-1], font=font, anchor="mm", fill=foreground)
        else:
            points = np.array([[-5, -1], [0, -6], [5, -1], [2, -1], [2, 6], [-2, 6], [-2, -1]])
            if code == "ArrowDown":
                points = -points
            elif code == "ArrowLeft":
                points = points[:, ::-1]
            elif code == "ArrowRight":
                points = -points[:, ::-1]
            draw.polygon([tuple(point + [x + 12, y + 12]) for point in points], fill=foreground)
    draw.text((10, 94), label, font=font, fill=(185, 202, 220))
    height, width = frame.shape[:2]
    scale = min(width / 640, height / 352)
    panel = panel.resize((max(1, round(216 * scale)), max(1, round(112 * scale))), Image.Resampling.LANCZOS)
    margin = round(8 * scale)
    output = Image.fromarray(frame.copy())
    output.paste(panel, (margin, height - panel.height - margin), panel)
    return np.asarray(output)


def export_keyboard_video(video_path, metadata_path, output_path, session_index=0) -> dict:
    """Stream a silent video to a new MP4 using a replay manifest or camera-trajectory JSON."""
    video_path, metadata_path, output_path = Path(video_path), Path(metadata_path), Path(output_path)
    if output_path.resolve() == video_path.resolve():
        raise ValueError("The output must be a different file from the original video.")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("The output filename must end in .mp4.")
    metadata = json.loads(metadata_path.read_text())
    with imageio.get_reader(str(video_path)) as reader:
        fps = reader.get_meta_data()["fps"]
        num_frames = reader.count_frames()
        if "sessions" in metadata:
            if not 0 <= session_index < len(metadata["sessions"]):
                raise ValueError("session_index is outside the manifest's sessions.")
            turns = [turn["params"] for turn in metadata["sessions"][session_index]["turns"]]
        else:
            if session_index != 0:
                raise ValueError("session_index only applies to replay manifests.")
            turns = [{"num_frames": num_frames, "video_fps": fps, "camera_trajectory": metadata}]
        timeline = KeyboardTimeline(turns)
        if timeline.num_frames != num_frames or not math.isclose(timeline.fps, fps, rel_tol=1e-6):
            raise ValueError(
                f"Video ({num_frames} frames, {fps} FPS) does not match the manifest "
                f"({timeline.num_frames} frames, {timeline.fps} FPS)."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(suffix=".mp4", prefix=".keyboard-", dir=output_path.parent)
        os.close(fd)
        try:
            with imageio.get_writer(
                temporary_path,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                ffmpeg_log_level="error",
                output_params=["-threads", "2"],
            ) as writer:
                written_frames = 0
                for frame_index, frame in enumerate(reader):
                    pressed, label = timeline.state_at_frame(frame_index)
                    writer.append_data(draw_keyboard_overlay(frame, pressed, label))
                    written_frames += 1
            if written_frames != num_frames:
                raise ValueError(f"Decoded {written_frames} frames; expected {num_frames}.")
            os.replace(temporary_path, output_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
    return {
        "input_video": str(video_path),
        "output_video": str(output_path),
        "fps": fps,
        "num_frames": num_frames,
        "recorded_key_events": sum(len(segment["events"]) for segment in timeline.segments),
        "segments": [
            {
                "start_frame": segment["start_frame"],
                "end_frame": segment["end_frame"],
                "key_source": "recorded" if segment["events"] else "none",
            }
            for segment in timeline.segments
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Original silent MP4, preserved without changes.")
    metadata_group = parser.add_mutually_exclusive_group(required=True)
    metadata_group.add_argument("--manifest", dest="metadata", help="Replay manifest JSON with sessions and turns.")
    metadata_group.add_argument("--camera-trajectory", dest="metadata", help="Camera JSON containing keyEvents.")
    parser.add_argument("--output", required=True, help="New MP4 filename; existing files are never overwritten.")
    parser.add_argument("--session-index", type=int, default=0, help="Zero-based session index in a replay manifest.")
    args = parser.parse_args()
    report = export_keyboard_video(args.video, args.metadata, args.output, args.session_index)
    print(json.dumps(report, indent=2))
