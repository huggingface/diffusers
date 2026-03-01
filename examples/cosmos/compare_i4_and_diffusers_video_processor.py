import json
import os
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from typing import Any, Callable, Optional
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo
from diffusers.utils import load_video
from diffusers.video_processor import VideoProcessor

H, W = 704, 1280
NUM_FRAMES = 93


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two video preprocessing pipelines.")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resample", type=str, default="bilinear", choices=["bilinear", "lanczos", "bicubic", "nearest"],
                        help="Resampling filter for VideoProcessor resize. Defaults to VideoProcessor's own default.")
    return parser.parse_args()


class CosmosVideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        prompt_type: str | None = None,  # "long", "short", "medium", or None for auto
        caption_format: str = "auto",  # "text", "json", or "auto"
        video_paths: Optional[list[str]] = None,
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (tuple[int, int]): Target size (H,W) for video frames
            prompt_type (str | None): Which prompt to use from JSON ("long", "short", "medium").
                                     If None, uses the first available prompt type.
                                     Only applicable when using JSON format.
            caption_format (str): Caption format - "text", "json", or "auto" to detect automatically

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.prompt_type = prompt_type
        self.caption_format = caption_format

        # Determine caption format and directory
        self._setup_caption_format()

        video_dir = os.path.join(self.dataset_dir, "videos")

        if video_paths is None:
            self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
            self.video_paths = sorted(self.video_paths)
        else:
            self.video_paths = video_paths
        print(f"{len(self.video_paths)} videos in total")

        self.num_failed_loads = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video(self, video_path: str) -> tuple[np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        # randomly sample a sequence of frames
        max_start_idx = total_frames - self.sequence_length
        np.random.seed(0)
        start_frame = np.random.randint(0, max_start_idx)
        end_frame = start_frame + self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache

        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps

    def _setup_caption_format(self) -> None:
        """Determine the caption format and set up the caption directory."""
        metas_dir = os.path.join(self.dataset_dir, "metas")
        captions_dir = os.path.join(self.dataset_dir, "captions")

        if self.caption_format == "auto":
            # Auto-detect based on directory existence
            if os.path.exists(captions_dir) and any(f.endswith(".json") for f in os.listdir(captions_dir)):
                self.caption_format = "json"
                self.caption_dir = captions_dir
            elif os.path.exists(metas_dir) and any(f.endswith(".txt") for f in os.listdir(metas_dir)):
                self.caption_format = "text"
                self.caption_dir = metas_dir
            else:
                raise ValueError(
                    f"Could not auto-detect caption format. Neither 'metas/*.txt' nor 'captions/*.json' found in {self.dataset_dir}"
                )
        elif self.caption_format == "json":
            if not os.path.exists(captions_dir):
                raise ValueError(f"JSON format specified but 'captions' directory not found in {self.dataset_dir}")
            self.caption_dir = captions_dir
        elif self.caption_format == "text":
            if not os.path.exists(metas_dir):
                raise ValueError(f"Text format specified but 'metas' directory not found in {self.dataset_dir}")
            self.caption_dir = metas_dir
        else:
            raise ValueError(f"Invalid caption_format: {self.caption_format}. Must be 'text', 'json', or 'auto'")

    def _load_text(self, text_source: Path) -> str:
        """Load text caption from file."""
        try:
            return text_source.read_text().strip()
        except Exception as e:
            print(f"Failed to read caption file {text_source}: {e}")
            return ""

    def _load_json_caption(self, json_path: Path) -> str:
        """Load caption from JSON file with prompt type selection."""
        try:
            with open(json_path, "r") as f:
                content = f.read()
                # Handle JSON that might not have top-level object
                if not content.strip().startswith("{"):
                    # Wrap in object if needed
                    data = json.loads("{" + content + "}")
                else:
                    data = json.loads(content)

            # Get the first model's captions (e.g., "qwen3_vl_30b_a3b")
            model_key = next(iter(data.keys()))
            captions = data[model_key]

            if self.prompt_type:
                # Use specified prompt type
                if self.prompt_type in captions:
                    return captions[self.prompt_type]
                else:
                    print(
                        f"Prompt type '{self.prompt_type}' not found in {json_path}. "
                        f"Available: {list(captions.keys())}. Using first available."
                    )

            # Use first available prompt type
            first_prompt = next(iter(captions.values()))
            return first_prompt

        except Exception as e:
            print(f"Failed to read JSON caption file {json_path}: {e}")
            return ""

    def _get_frames(self, video_path: str) -> tuple[torch.Tensor, float]:
        frames, fps = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps

    def __getitem__(self, index: int) -> dict | Any:
        try:
            data = dict()
            video, fps = self._get_frames(self.video_paths[index])
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]

            # Load caption based on format
            video_path = self.video_paths[index]
            video_basename = os.path.basename(video_path).replace(".mp4", "")

            if self.caption_format == "json":
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.json")
                caption = self._load_json_caption(Path(caption_path))
            else:  # text format
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.txt")
                caption = self._load_text(Path(caption_path))

            data["video"] = video
            data["caption"] = caption

            _, _, h, w = video.shape

            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        except Exception as e:
            self.num_failed_loads += 1
            print(
                f"Failed to load video {self.video_paths[index]} (total failures: {self.num_failed_loads}): {e}\n"
            )
            # Randomly sample another video
            return self[np.random.randint(len(self.video_paths))]




class DiffusersCosmosVideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        prompt_type: str | None = None,  # "long", "short", "medium", or None for auto
        caption_format: str = "auto",  # "text", "json", or "auto"
        video_paths: Optional[list[str]] = None,
        resample: str = None,
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (tuple[int, int]): Target size (H,W) for video frames
            prompt_type (str | None): Which prompt to use from JSON ("long", "short", "medium").
                                     If None, uses the first available prompt type.
                                     Only applicable when using JSON format.
            caption_format (str): Caption format - "text", "json", or "auto" to detect automatically
            resample (str): Resampling filter for VideoProcessor resize (e.g. "bilinear", "lanczos").

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.num_frames = num_frames
        self.prompt_type = prompt_type
        self.caption_format = caption_format

        # Determine caption format and directory
        self._setup_caption_format()

        video_dir = os.path.join(self.dataset_dir, "videos")

        if video_paths is None:
            self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
            self.video_paths = sorted(self.video_paths)
        else:
            self.video_paths = video_paths
        print(f"{len(self.video_paths)} videos in total")

        self.video_size = video_size
        print("VideoProcessor resample mode:", resample)
        vp_kwargs = {"resample": resample} if resample is not None else {}
        self.video_processor = VideoProcessor(vae_scale_factor=8, **vp_kwargs)
        self.num_failed_loads = 0

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video(self, video_path: str) -> list:
        frames = load_video(video_path)
        total_frames = len(frames)
        if total_frames < self.num_frames:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.num_frames} frames are required."
            )

        # randomly sample a consecutive window of frames
        max_start_idx = total_frames - self.num_frames
        np.random.seed(0)
        start_frame = np.random.randint(0, max_start_idx)
        return frames[start_frame : start_frame + self.num_frames]

    def _setup_caption_format(self) -> None:
        """Determine the caption format and set up the caption directory."""
        metas_dir = os.path.join(self.dataset_dir, "metas")
        captions_dir = os.path.join(self.dataset_dir, "captions")

        if self.caption_format == "auto":
            # Auto-detect based on directory existence
            if os.path.exists(captions_dir) and any(f.endswith(".json") for f in os.listdir(captions_dir)):
                self.caption_format = "json"
                self.caption_dir = captions_dir
            elif os.path.exists(metas_dir) and any(f.endswith(".txt") for f in os.listdir(metas_dir)):
                self.caption_format = "text"
                self.caption_dir = metas_dir
            else:
                raise ValueError(
                    f"Could not auto-detect caption format. Neither 'metas/*.txt' nor 'captions/*.json' found in {self.dataset_dir}"
                )
        elif self.caption_format == "json":
            if not os.path.exists(captions_dir):
                raise ValueError(f"JSON format specified but 'captions' directory not found in {self.dataset_dir}")
            self.caption_dir = captions_dir
        elif self.caption_format == "text":
            if not os.path.exists(metas_dir):
                raise ValueError(f"Text format specified but 'metas' directory not found in {self.dataset_dir}")
            self.caption_dir = metas_dir
        else:
            raise ValueError(f"Invalid caption_format: {self.caption_format}. Must be 'text', 'json', or 'auto'")

    def _load_text(self, text_source: Path) -> str:
        """Load text caption from file."""
        try:
            return text_source.read_text().strip()
        except Exception as e:
            print(f"Failed to read caption file {text_source}: {e}")
            return ""

    def _load_json_caption(self, json_path: Path) -> str:
        """Load caption from JSON file with prompt type selection."""
        try:
            with open(json_path, "r") as f:
                content = f.read()
                # Handle JSON that might not have top-level object
                if not content.strip().startswith("{"):
                    # Wrap in object if needed
                    data = json.loads("{" + content + "}")
                else:
                    data = json.loads(content)

            # Get the first model's captions (e.g., "qwen3_vl_30b_a3b")
            model_key = next(iter(data.keys()))
            captions = data[model_key]

            if self.prompt_type:
                # Use specified prompt type
                if self.prompt_type in captions:
                    return captions[self.prompt_type]
                else:
                    print(
                        f"Prompt type '{self.prompt_type}' not found in {json_path}. "
                        f"Available: {list(captions.keys())}. Using first available."
                    )

            # Use first available prompt type
            first_prompt = next(iter(captions.values()))
            return first_prompt

        except Exception as e:
            print(f"Failed to read JSON caption file {json_path}: {e}")
            return ""

    def _get_frames(self, video_path: str) -> torch.Tensor:
        frames = self._load_video(video_path)  # list of PIL images
        video = self.video_processor.preprocess_video(frames, height=self.video_size[0], width=self.video_size[1])
        # video: [1, C, T, H, W] in [-1, 1]
        return video.squeeze(0)  # [C, T, H, W]

    def __getitem__(self, index: int) -> dict | Any:
        try:
            data = dict()
            video = self._get_frames(self.video_paths[index])  # [C, T, H, W]

            # Load caption based on format
            video_path = self.video_paths[index]
            video_basename = os.path.basename(video_path).replace(".mp4", "")

            if self.caption_format == "json":
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.json")
                caption = self._load_json_caption(Path(caption_path))
            else:  # text format
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.txt")
                caption = self._load_text(Path(caption_path))

            data["video"] = video
            data["caption"] = caption

            return data
        except Exception as e:
            self.num_failed_loads += 1
            print(
                f"Failed to load video {self.video_paths[index]} (total failures: {self.num_failed_loads}): {e}\n"
            )
            # Randomly sample another video
            return self[np.random.randint(len(self.video_paths))]


def build_dataloader(args, dataset_cls, **resample_kwargs):
    dataset = dataset_cls(
            video_paths=None,
            num_frames=93,
            video_size=[H, W],
            dataset_dir=args.train_data_dir,
            **resample_kwargs,
        )
    
    dataloader = DataLoader(
            dataset=dataset,
            sampler=None,
            batch_size=args.train_batch_size,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )
    return dataloader


def main():
    args = parse_args()

    data_loader1 = build_dataloader(args, CosmosVideoDataset)
    resample_kwargs = {"resample": args.resample} if args.resample is not None else {}
    data_loader2 = build_dataloader(args, DiffusersCosmosVideoDataset, **resample_kwargs)

    for (batch1, batch2) in zip(data_loader1, data_loader2):
        x1 = batch1["video"].float() / 127.5 - 1.0  # cosmos: uint8 [0, 255] → float [-1, 1]
        x2 = batch2["video"]                        # diffusers: already float [-1, 1]

        diff = (x1 - x2).abs()
        print(f"diff  max={diff.amax():.4f}  mean={diff.mean():.4f}")


if __name__ == "__main__":
    main()
