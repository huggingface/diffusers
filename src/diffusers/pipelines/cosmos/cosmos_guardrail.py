# Copyright 2024 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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

# The following code has been copied and modified from https://github.com/NVIDIA/Cosmos

import json
import os
import pathlib
import re
import string
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, Tuple, Union

import numpy as np
import PIL.Image
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipModel, SiglipProcessor

from ...utils import (
    get_logger,
    is_better_profanity_available,
    is_nltk_available,
    is_peft_available,
    is_pytorch_retinaface_available,
    load_video,
)
from .cosmos_utils import (
    CLASS_IDX_TO_NAME,
    KEEP_TOP_K,
    NMS_THRESHOLD,
    TOP_K,
    UNSAFE_CATEGORIES,
    decode_batch,
    filter_detected_boxes,
    load_model,
    pixelate_face,
    read_keyword_list_from_dir,
    to_ascii,
)


if is_better_profanity_available():
    from better_profanity import profanity

if is_nltk_available():
    import nltk

if is_peft_available():
    from peft import PeftModel

if is_pytorch_retinaface_available():
    from pytorch_retinaface.data import cfg_re50
    from pytorch_retinaface.layers.functions.prior_box import PriorBox
    from pytorch_retinaface.models.retinaface import RetinaFace


logger = get_logger(__name__)  # pylint: disable=invalid-name

CENSOR = "*"
COSMOS_GUARDRAIL_CHECKPOINT = "nvidia/Cosmos-1.0-Guardrail"


class ContentSafetyGuardrail:
    def is_safe(self, **kwargs) -> Tuple[bool, str]:
        raise NotImplementedError("ContentSafetyGuardrail::is_safe method must be implemented by child classes")


class PostprocessingGuardrail:
    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        raise NotImplementedError("PostprocessingGuardrail::postprocess method must be implemented by child classes")


class GuardrailRunner:
    def __init__(
        self,
        safety_models: list[ContentSafetyGuardrail] | None = None,
        generic_block_msg: str = "",
        generic_safe_msg: str = "",
        postprocessors: list[PostprocessingGuardrail] | None = None,
    ):
        self.safety_models = safety_models
        self.generic_block_msg = generic_block_msg
        self.generic_safe_msg = generic_safe_msg if generic_safe_msg else "Prompt is safe"
        self.postprocessors = postprocessors

    def run_safety_check(self, input: Any) -> Tuple[bool, str]:
        """Run the safety check on the input."""
        if not self.safety_models:
            logger.warning("No safety models found, returning safe")
            return True, self.generic_safe_msg

        for guardrail in self.safety_models:
            guardrail_name = str(guardrail.__class__.__name__).upper()
            logger.debug(f"Running guardrail: {guardrail_name}")
            safe, message = guardrail.is_safe(input)
            if not safe:
                reasoning = self.generic_block_msg if self.generic_block_msg else f"{guardrail_name}: {message}"
                return False, reasoning

        return True, self.generic_safe_msg

    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        """Run the postprocessing on the video frames."""
        if not self.postprocessors:
            logger.warning("No postprocessors found, returning original frames")
            return frames

        for guardrail in self.postprocessors:
            guardrail_name = str(guardrail.__class__.__name__).upper()
            logger.debug(f"Running guardrail: {guardrail_name}")
            frames = guardrail.postprocess(frames)

        return frames


@dataclass
class ModelConfig:
    input_size: int = 1152
    num_classes: int = 7


class SafetyClassifier(torch.nn.Module):
    def __init__(self, input_size: int = 1024, num_classes: int = 2):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.num_classes),
            # Note: No activation function here; CrossEntropyLoss expects raw logits
        )

    def forward(self, x):
        return self.layers(x)


class VideoSafetyModel(torch.nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.network = SafetyClassifier(input_size=config.input_size, num_classes=self.num_classes)

    @torch.inference_mode()
    def forward(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self.network(data_batch["data"].cuda())
        return {"logits": logits}


class SigLIPEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
    ) -> None:
        super().__init__()

        checkpoint_dir = snapshot_download(checkpoint_id)
        checkpoint_dir = (pathlib.Path(checkpoint_dir) / "video_content_safety_filter").as_posix()

        self.checkpoint_dir = checkpoint_dir
        self.model = SiglipModel.from_pretrained(model_name, cache_dir=self.checkpoint_dir)
        self.processor = SiglipProcessor.from_pretrained(model_name, cache_dir=self.checkpoint_dir)

    @torch.inference_mode()
    def encode_image(self, input_img: PIL.Image.Image) -> torch.Tensor:
        """Encode an image into a feature vector."""
        with torch.no_grad():
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            inputs = self.processor(images=input_img, return_tensors="pt").to(device, dtype=dtype)
            image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features


class Aegis(torch.nn.Module, ContentSafetyGuardrail):
    def __init__(
        self,
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        base_model_id: str = "meta-llama/LlamaGuard-7b",
        aegis_adapter: str = "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
    ) -> None:
        super().__init__()

        checkpoint_dir = snapshot_download(checkpoint_id)
        checkpoint_dir = (pathlib.Path(checkpoint_dir) / "aegis").as_posix()

        self.checkpoint_dir = checkpoint_dir
        self.dtype = torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, cache_dir=self.checkpoint_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=self.checkpoint_dir)
        self.model = PeftModel.from_pretrained(base_model, aegis_adapter, cache_dir=self.checkpoint_dir)

    def get_moderation_prompt(self, user_prompt: str) -> str:
        """Create the moderation prompt for the Aegis model."""
        full_prompt = f"""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the
below categories.

        <BEGIN UNSAFE CONTENT CATEGORIES> {"".join([f"{category['instruction']}" for category in UNSAFE_CATEGORIES])}
        <END UNSAFE CONTENT CATEGORIES>

        <BEGIN CONVERSATION>

        """
        full_prompt += f"User: {user_prompt}\n\n"
        full_prompt += """<END CONVERSATION>

        Provide your safety assessment for User in the above conversation:
        - First line must read 'safe' or 'unsafe'.
        - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""

        return full_prompt

    def get_aegis_block_message(self, moderation_output: str) -> str:
        """Extract the blocked category and reason from the Aegis model output."""
        block_msg = "Prompt blocked by Aegis."
        try:
            # Extract blocked category and reason
            if len(moderation_output.split("\n")) >= 2:
                moderation_output_list = moderation_output.split("\n")
                blocked_category = int(moderation_output_list[1].lstrip("O")) - 1
                if 0 <= blocked_category < len(UNSAFE_CATEGORIES):
                    blocked_reason = UNSAFE_CATEGORIES[blocked_category]["blocked_reason"]
                    blocked_category_name = UNSAFE_CATEGORIES[blocked_category]["category"]
                    block_msg = f"{blocked_category_name}: {blocked_reason}"
        except Exception as e:
            logger.warning(f"Unable to extract blocked category and reason from Aegis output: {e}")
        return block_msg

    def filter_aegis_output(self, prompt: str) -> tuple[bool, str]:
        """Filter the Aegis model output and return the safety status and message."""
        full_prompt = self.get_moderation_prompt(prompt)
        device = next(self.model.parameters()).device
        inputs = self.tokenizer([full_prompt], add_special_tokens=False, return_tensors="pt").to(device)
        output = self.model.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id)
        prompt_len = inputs["input_ids"].shape[-1]
        moderation_output = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        if "unsafe" in moderation_output.lower():
            block_msg = self.get_aegis_block_message(moderation_output)
            return False, block_msg
        else:
            return True, ""

    def is_safe(self, prompt: str) -> tuple[bool, str]:
        """Check if the input prompt is safe according to the Aegis model."""
        try:
            return self.filter_aegis_output(prompt)
        except Exception as e:
            logger.error(f"Unexpected error occurred when running Aegis guardrail: {e}")
            return True, "Unexpected error occurred when running Aegis guardrail."


class Blocklist(ContentSafetyGuardrail):
    def __init__(
        self,
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        guardrail_partial_match_min_chars: int = 4,
        guardrail_partial_match_letter_count: float = 0.5,
    ) -> None:
        checkpoint_dir = snapshot_download(checkpoint_id)
        checkpoint_dir = (pathlib.Path(checkpoint_dir) / "blocklist").as_posix()

        nltk.data.path.append(os.path.join(checkpoint_dir, "nltk_data"))
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.profanity = profanity
        self.checkpoint_dir = checkpoint_dir
        self.guardrail_partial_match_min_chars = guardrail_partial_match_min_chars
        self.guardrail_partial_match_letter_count = guardrail_partial_match_letter_count

        # Load blocklist and whitelist keywords
        self.blocklist_words = read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "custom"))
        self.whitelist_words = read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "whitelist"))
        self.exact_match_words = read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "exact_match"))

        self.profanity.load_censor_words(custom_words=self.blocklist_words, whitelist_words=self.whitelist_words)
        logger.debug(f"Loaded {len(self.blocklist_words)} words/phrases from blocklist")
        logger.debug(f"Whitelisted {len(self.whitelist_words)} words/phrases from whitelist")
        logger.debug(f"Loaded {len(self.exact_match_words)} exact match words/phrases from blocklist")

    def uncensor_whitelist(self, input_prompt: str, censored_prompt: str) -> str:
        """Explicitly uncensor words that are in the whitelist."""
        input_words = input_prompt.split()
        censored_words = censored_prompt.split()
        whitelist_words = set(self.whitelist_words)
        for i, token in enumerate(input_words):
            if token.strip(string.punctuation).lower() in whitelist_words:
                censored_words[i] = token
        censored_prompt = " ".join(censored_words)
        return censored_prompt

    def censor_prompt(self, input_prompt: str) -> tuple[bool, str]:
        """Censor the prompt using the blocklist with better-profanity fuzzy matching.

        Args:
            input_prompt: input prompt to censor

        Returns:
            bool: True if the prompt is blocked, False otherwise str: A message indicating why the prompt was blocked
        """
        censored_prompt = self.profanity.censor(input_prompt, censor_char=CENSOR)
        # Uncensor whitelisted words that were censored from blocklist fuzzy matching
        censored_prompt = self.uncensor_whitelist(input_prompt, censored_prompt)
        if CENSOR in censored_prompt:
            return True, f"Prompt blocked by censorship: Censored Prompt: {censored_prompt}"
        return False, ""

    @staticmethod
    def check_partial_match(
        normalized_prompt: str, normalized_word: str, guardrail_partial_match_letter_count: float
    ) -> tuple[bool, str]:
        """
        Check robustly if normalized word and the matching target have a difference of up to
        guardrail_partial_match_letter_count characters.

        Args:
            normalized_prompt: a string with many words
            normalized_word: a string with one or multiple words, its length is smaller than normalized_prompt
            guardrail_partial_match_letter_count:
                maximum allowed difference in characters (float to allow partial characters)

        Returns:
            bool: True if a match is found, False otherwise str: A message indicating why the prompt was blocked
        """
        prompt_words = normalized_prompt.split()
        word_length = len(normalized_word.split())
        max_similarity_ratio = (len(normalized_word) - float(guardrail_partial_match_letter_count)) / float(
            len(normalized_word)
        )

        for i in range(len(prompt_words) - word_length + 1):
            # Extract a substring from the prompt with the same number of words as the normalized_word
            substring = " ".join(prompt_words[i : i + word_length])
            similarity_ratio = SequenceMatcher(None, substring, normalized_word).ratio()
            if similarity_ratio >= max_similarity_ratio:
                return (
                    True,
                    f"Prompt blocked by partial match blocklist: Prompt: {normalized_prompt}, Partial Match Word: {normalized_word}",
                )

        return False, ""

    @staticmethod
    def check_against_whole_word_blocklist(
        prompt: str,
        blocklist: list[str],
        guardrail_partial_match_min_chars: int = 4,
        guardrail_partial_match_letter_count: float = 0.5,
    ) -> bool:
        """
        Check if the prompt contains any whole words from the blocklist. The match is case insensitive and robust to
        multiple spaces between words.

        Args:
            prompt: input prompt to check
            blocklist: list of words to check against
            guardrail_partial_match_min_chars: minimum number of characters in a word to check for partial match
            guardrail_partial_match_letter_count: maximum allowed difference in characters for partial match

        Returns:
            bool: True if a match is found, False otherwise str: A message indicating why the prompt was blocked
        """
        # Normalize spaces and convert to lowercase
        normalized_prompt = re.sub(r"\s+", " ", prompt).strip().lower()

        for word in blocklist:
            # Normalize spaces and convert to lowercase for each blocklist word
            normalized_word = re.sub(r"\s+", " ", word).strip().lower()

            # Use word boundaries to ensure whole word match
            if re.search(r"\b" + re.escape(normalized_word) + r"\b", normalized_prompt):
                return True, f"Prompt blocked by exact match blocklist: Prompt: {prompt}, Exact Match Word: {word}"

            # Check for partial match if the word is long enough
            if len(normalized_word) >= guardrail_partial_match_min_chars:
                match, message = Blocklist.check_partial_match(
                    normalized_prompt, normalized_word, guardrail_partial_match_letter_count
                )
                if match:
                    return True, message

        return False, ""

    def is_safe(self, input_prompt: str = "") -> tuple[bool, str]:
        """Check if the input prompt is safe using the blocklist."""
        # Check if the input is empty
        if not input_prompt:
            return False, "Input is empty"
        input_prompt = to_ascii(input_prompt)

        # Check full sentence for censored words
        censored, message = self.censor_prompt(input_prompt)
        if censored:
            return False, message

        # Check lemmatized words for censored words
        tokens = nltk.word_tokenize(input_prompt)
        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_prompt = " ".join(lemmas)
        censored, message = self.censor_prompt(lemmatized_prompt)
        if censored:
            return False, message

        # Check for exact match blocklist words
        censored, message = self.check_against_whole_word_blocklist(
            input_prompt,
            self.exact_match_words,
            self.guardrail_partial_match_min_chars,
            self.guardrail_partial_match_letter_count,
        )
        if censored:
            return False, message

        # If all these checks pass, the input is safe
        return True, "Input is safe"


class VideoContentSafetyFilter(torch.nn.Module, ContentSafetyGuardrail):
    def __init__(
        self,
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
    ) -> None:
        super().__init__()

        checkpoint_dir = snapshot_download(checkpoint_id)
        checkpoint_dir = (pathlib.Path(checkpoint_dir) / "video_content_safety_filter").as_posix()

        self.encoder = SigLIPEncoder(checkpoint_id=checkpoint_id)

        model_config = ModelConfig(input_size=1152, num_classes=7)
        self.model = VideoSafetyModel(model_config)

        safety_filter_local_path = os.path.join(checkpoint_dir, "safety_filter.pt")
        checkpoint = torch.load(safety_filter_local_path, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])

        self.eval()

    @torch.inference_mode()
    def __infer(self, pil_image: PIL.Image.Image) -> int:
        """Infer the class of the image."""
        image_embs = self.encoder.encode_image(pil_image)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        image_embs = image_embs.to(device=device, dtype=dtype)
        logits = self.model.network(image_embs)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        return predicted_class

    def is_safe_file(self, filepath: str) -> bool:
        """Check if the video file is safe."""
        video_data = load_video(filepath)

        # Sample frames at 2 FPS
        sample_rate = 2  # frames per second
        frame_interval = int(video_data.fps / sample_rate)
        frame_numbers = list(range(0, int(video_data.fps * video_data.duration), frame_interval))

        is_safe = True
        frame_scores = []

        for frame_number in frame_numbers:
            try:
                frame = video_data.frames[frame_number]
                pil_image = PIL.Image.fromarray(frame)
                predicted_class = self.__infer(pil_image)
                class_name = CLASS_IDX_TO_NAME.get(predicted_class, "Unknown")
                frame_scores.append({"frame_number": frame_number, "class": class_name})

                # If any frame is not "Safe", mark the video as unsafe
                if predicted_class != 0:
                    is_safe = False
                    break

            except Exception as e:
                logger.warning(
                    f"Warning: Failed to run safety classifier on frame_number {frame_number}. Exception: {e}"
                )
                continue

        # Prepare data for JSON
        video_data = {
            "filepath": filepath,
            "is_safe": is_safe,
            "video_length": video_data.duration,
            "fps": video_data.fps,
            "frame_scores": frame_scores,
        }

        logger.info(f"Video {filepath} is {'SAFE' if is_safe else 'UNSAFE'}.")
        logger.debug(f"Video data: {json.dumps(video_data, indent=4)}")
        return is_safe

    def is_safe_frames(self, frames: Iterable) -> bool:
        """Check if the video frames are safe."""
        is_safe = True
        frame_scores = []

        for frame_number, frame in enumerate(frames):
            try:
                pil_image = PIL.Image.fromarray(frame)
                predicted_class = self.__infer(pil_image)
                class_name = CLASS_IDX_TO_NAME.get(predicted_class, "Unknown")
                frame_scores.append({"frame_number": frame_number, "class": class_name})

                # If any frame is not "Safe", mark as not safe
                if predicted_class != 0:
                    is_safe = False
                    break

            except Exception as e:
                logger.warning(
                    f"Warning: Failed to run safety classifier on frame_number {frame_number}. Exception: {e}"
                )
                continue

        video_data = {
            "is_safe": is_safe,
            "frame_scores": frame_scores,
        }

        logger.debug(f"Frames data: {json.dumps(video_data, indent=4)}")
        return is_safe

    def is_safe(self, input: Union[str, Iterable]) -> Tuple[bool, str]:
        if isinstance(input, str):
            is_safe = self.is_safe_file(input)
            return is_safe, "safe video detected" if is_safe else "unsafe video detected"
        elif isinstance(input, Iterable):
            is_safe = self.is_safe_frames(input)
            return is_safe, "safe frames detected" if is_safe else "unsafe frames detected"
        else:
            raise ValueError(f"Input type {type(input)} not supported.")


class RetinaFaceFilter(torch.nn.Module, PostprocessingGuardrail):
    def __init__(
        self,
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        batch_size: int = 1,
        confidence_threshold: float = 0.7,
    ) -> None:
        super().__init__()

        checkpoint_dir = snapshot_download(checkpoint_id)
        checkpoint = pathlib.Path(checkpoint_dir) / "face_blur_filter/Resnet50_Final.pth"

        self.cfg = cfg_re50
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold

        # Disable loading ResNet pretrained weights
        self.cfg["pretrain"] = False
        self.net = RetinaFace(cfg=self.cfg, phase="test")

        # Load from RetinaFace pretrained checkpoint
        self.net = load_model(self.net, checkpoint)

        self.eval()

    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Preprocess a sequence of frames for face detection.

        Args:
            frames: Input frames

        Returns:
            Preprocessed frames tensor
        """
        device = next(self.net.parameters()).device
        dtype = next(self.net.parameters()).dtype

        with torch.no_grad():
            frames_tensor = torch.from_numpy(frames).to(device=device, dtype=dtype)  # Shape: [T, H, W, C]
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # Shape: [T, C, H, W]
            frames_tensor = frames_tensor[:, [2, 1, 0], :, :]  # RGB to BGR to match RetinaFace model input
            means = torch.tensor([104.0, 117.0, 123.0], device=device, dtype=dtype).view(1, 3, 1, 1)
            frames_tensor = frames_tensor - means  # Subtract mean BGR values for each channel
            return frames_tensor

    def blur_detected_faces(
        self,
        frames: np.ndarray,
        batch_loc: torch.Tensor,
        batch_conf: torch.Tensor,
        prior_data: torch.Tensor,
        scale: torch.Tensor,
        min_size: tuple[int] = (20, 20),
    ) -> list[np.ndarray]:
        """Blur detected faces in a batch of frames using RetinaFace predictions.

        Args:
            frames: Input frames
            batch_loc: Batched location predictions
            batch_conf: Batched confidence scores
            prior_data: Prior boxes for the video
            scale: Scale factor for resizing detections
            min_size: Minimum size of a detected face region in pixels

        Returns:
            Processed frames with pixelated faces
        """
        with torch.no_grad():
            batch_boxes = decode_batch(batch_loc, prior_data, self.cfg["variance"])
            batch_boxes = batch_boxes * scale

        blurred_frames = []
        for i, boxes in enumerate(batch_boxes):
            boxes = boxes.detach().cpu().numpy()
            scores = batch_conf[i, :, 1].detach().cpu().numpy()

            filtered_boxes = filter_detected_boxes(
                boxes,
                scores,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=NMS_THRESHOLD,
                top_k=TOP_K,
                keep_top_k=KEEP_TOP_K,
            )

            frame = frames[i]
            for box in filtered_boxes:
                x1, y1, x2, y2 = map(int, box)
                # Ignore bounding boxes smaller than the minimum size
                if x2 - x1 < min_size[0] or y2 - y1 < min_size[1]:
                    continue
                max_h, max_w = frame.shape[:2]
                face_roi = frame[max(y1, 0) : min(y2, max_h), max(x1, 0) : min(x2, max_w)]
                blurred_face = pixelate_face(face_roi)
                frame[max(y1, 0) : min(y2, max_h), max(x1, 0) : min(x2, max_w)] = blurred_face
            blurred_frames.append(frame)

        return blurred_frames

    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        """Blur faces in a sequence of frames.

        Args:
            frames: Input frames

        Returns:
            Processed frames with pixelated faces
        """
        # Create dataset and dataloader
        frames_tensor = self.preprocess_frames(frames)
        dataset = TensorDataset(frames_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        processed_frames, processed_batches = [], []
        device = next(self.net.parameters()).device
        dtype = next(self.net.parameters()).dtype

        prior_data, scale = None, None
        for i, batch in enumerate(dataloader):
            batch = batch[0]
            h, w = batch.shape[-2:]  # Batch shape: [C, H, W]

            with torch.no_grad():
                # Generate priors for the video
                if prior_data is None:
                    priorbox = PriorBox(self.cfg, image_size=(h, w))
                    priors = priorbox.forward()
                    priors = priors.to(device, dtype=dtype)
                    prior_data = priors.data

                # Get scale for resizing detections
                if scale is None:
                    scale = torch.Tensor([w, h, w, h])
                    scale = scale.to(device, dtype=dtype)

                batch_loc, batch_conf, _ = self.net(batch)

            # Blur detected faces in each batch of frames
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(frames))
            processed_batches.append(
                self.blur_detected_faces(frames[start_idx:end_idx], batch_loc, batch_conf, prior_data, scale)
            )

        processed_frames = [frame for batch in processed_batches for frame in batch]
        return np.array(processed_frames)


class CosmosSafetyChecker(torch.nn.Module):
    def __init__(
        self,
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        aegis_model_id: str = "meta-llama/LlamaGuard-7b",
        aegis_adapter_id: str = "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
    ) -> None:
        super().__init__()

        self.text_guardrail = GuardrailRunner(
            safety_models=[
                Blocklist(checkpoint_id),
                Aegis(checkpoint_id, aegis_model_id, aegis_adapter_id),
            ]
        )
        self.video_guardrail = GuardrailRunner(
            safety_models=[VideoContentSafetyFilter(checkpoint_id)],
            postprocessors=[RetinaFaceFilter(checkpoint_id)],
        )

    def check_text_safety(self, prompt: str) -> bool:
        is_safe, message = self.text_guardrail.run_safety_check(prompt)
        if not is_safe:
            logger.critical(f"GUARDRAIL BLOCKED: {message}")
        return is_safe

    def check_video_safety(self, frames: np.ndarray) -> np.ndarray:
        is_safe, message = self.video_guardrail.run_safety_check(frames)
        if not is_safe:
            logger.critical(f"GUARDRAIL BLOCKED: {message}")
            return None
        frames = self.video_guardrail.postprocess(frames)
        return frames

    def to(self, device: Union[str, torch.device] = None, dtype: torch.dtype = None) -> None:
        self.text_guardrail.safety_models[1].model.to(device=device, dtype=dtype)
        self.video_guardrail.safety_models[0].model.to(device=device, dtype=dtype)
        self.video_guardrail.postprocessors[0].to(device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        return self.text_guardrail.safety_models[1].model.device

    @property
    def dtype(self) -> torch.dtype:
        return self.text_guardrail.safety_models[1].model.dtype
