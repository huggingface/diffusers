import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer, SiglipModel, SiglipProcessor


class BaseRewardScorer(nn.Module):
    """
    Base interface for reward scorers.

    Subclasses are expected to implement a differentiable `forward` method that
    accepts a batch of images in the `[-1, 1]` range and a batch of prompt strings
    with the same batch dimension.
    """

    name: str = "base"
    default_model_id: Optional[str] = None
    supports_gradients: bool = True

    def __init__(self, model_id: Optional[str] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.model_id = model_id or self.default_model_id
        if self.model_id is None:
            raise ValueError(f"{self.__class__.__name__} requires `model_id` to be specified.")
        self._requested_device = torch.device(device) if device is not None else None

    @property
    def device(self) -> torch.device:
        parameters = list(self.parameters())
        if parameters:
            return parameters[0].device
        return self._requested_device or torch.device("cpu")

    def ensure_device(self) -> None:
        if self._requested_device is not None:
            self.to(self._requested_device)

    def forward(self, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        raise NotImplementedError


class ClipScorer(BaseRewardScorer):
    name = "clip"
    default_model_id = "openai/clip-vit-large-patch14"

    def __init__(self, model_id: Optional[str] = None, device: Optional[torch.device] = None):
        super().__init__(model_id=model_id, device=device)

        self.model = CLIPModel.from_pretrained(self.model_id)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_id)
        if self._requested_device is not None:
            self.model = self.model.to(self._requested_device)
        self.model = self.model.to(dtype=torch.float32)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        device = self.model.device
        pixel_values = self._preprocess_images(images).to(device=device, dtype=torch.float32)
        text_inputs = self.tokenizer(list(prompts), padding=True, truncation=True, return_tensors="pt").to(device)

        image_embeds = self.model.get_image_features(pixel_values=pixel_values)
        text_embeds = self.model.get_text_features(**text_inputs)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return (image_embeds * text_embeds).sum(dim=-1)

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = (images + 1) / 2
        pixel_values = torch.clamp(pixel_values, 0, 1)

        crop_size = self.image_processor.crop_size
        if isinstance(crop_size, dict):
            target_height = crop_size["height"]
            target_width = crop_size["width"]
        else:
            target_height = target_width = crop_size

        pixel_values = F.interpolate(
            pixel_values, size=(target_height, target_width), mode="bilinear", align_corners=False
        )

        mean = torch.tensor(
            self.image_processor.image_mean, device=pixel_values.device, dtype=pixel_values.dtype
        ).view(1, -1, 1, 1)
        std = torch.tensor(self.image_processor.image_std, device=pixel_values.device, dtype=pixel_values.dtype).view(
            1, -1, 1, 1
        )
        return (pixel_values - mean) / std


class SiglipScorer(BaseRewardScorer):
    name = "siglip"
    default_model_id = "google/siglip-so400m-patch14-384"

    def __init__(self, model_id: Optional[str] = None, device: Optional[torch.device] = None):
        super().__init__(model_id=model_id, device=device)

        self.processor = SiglipProcessor.from_pretrained(self.model_id)
        self.image_processor = self.processor.image_processor
        self.text_tokenizer = self.processor.tokenizer
        self.model = SiglipModel.from_pretrained(self.model_id)
        if self._requested_device is not None:
            self.model = self.model.to(self._requested_device)
        self.model = self.model.to(dtype=torch.float32)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:  # type: ignore[override]
        device = self.model.device
        pixel_values = self._preprocess_images(images).to(device=device, dtype=torch.float32)
        text_inputs = self.text_tokenizer(list(prompts), padding=True, truncation=True, return_tensors="pt").to(device)

        image_embeds = self.model.get_image_features(pixel_values=pixel_values)
        text_embeds = self.model.get_text_features(**text_inputs)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return (image_embeds * text_embeds).sum(dim=-1)

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = (images + 1) / 2
        pixel_values = torch.clamp(pixel_values, 0, 1)

        size = self.image_processor.size
        if isinstance(size, dict):
            target_height = size.get("shortest_edge") or size.get("height") or size.get("width")
            target_width = size.get("width") or target_height
            target_height = target_height or target_width
        else:
            target_height = target_width = size

        pixel_values = F.interpolate(
            pixel_values, size=(target_height, target_width), mode="bilinear", align_corners=False
        )

        mean = torch.tensor(
            self.image_processor.image_mean, device=pixel_values.device, dtype=pixel_values.dtype
        ).view(1, -1, 1, 1)
        std = torch.tensor(self.image_processor.image_std, device=pixel_values.device, dtype=pixel_values.dtype).view(
            1, -1, 1, 1
        )
        return (pixel_values - mean) / std


class PlaceholderScorer(BaseRewardScorer):
    """
    Helper scorer that surfaces a friendly error for scorers that require external dependencies.
    """

    name = "placeholder"
    supports_gradients = False

    def __init__(self, *args: Any, required_package: str, scorer_name: str, **kwargs: Any):
        self.required_package = required_package
        self.scorer_name = scorer_name
        raise ImportError(f"{scorer_name} requires the external package `{required_package}` which is not installed.")


SCORER_REGISTRY: Dict[str, Type[BaseRewardScorer]] = {ClipScorer.name: ClipScorer, SiglipScorer.name: SiglipScorer}


def available_scorers() -> Tuple[str, ...]:
    return tuple(sorted(SCORER_REGISTRY.keys()))


def build_scorer(
    name: str,
    model_id: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> BaseRewardScorer:
    if name not in SCORER_REGISTRY:
        raise ValueError(f"Unknown scorer `{name}`. Available scorers: {', '.join(available_scorers())}.")
    scorer_cls = SCORER_REGISTRY[name]
    device_obj = torch.device(device) if device is not None else None

    scorer = scorer_cls(model_id=model_id, device=device_obj, **kwargs)

    if not scorer.supports_gradients:
        warnings.warn(
            f"Scorer `{name}` does not declare gradient support. Adversarial refinement may not work as expected.",
            UserWarning,
        )

    return scorer
