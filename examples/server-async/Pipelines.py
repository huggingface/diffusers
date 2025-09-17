from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
import torch
import os
import logging
from pydantic import BaseModel
from dataclasses import dataclass,  field
from typing import List

logger = logging.getLogger(__name__)

class TextToImageInput(BaseModel):
    model: str
    prompt: str
    size: str | None = None
    n: int | None = None


@dataclass
class PresetModels:
    SD3: List[str] = field(default_factory=lambda: ['stabilityai/stable-diffusion-3-medium'])
    SD3_5: List[str] = field(default_factory=lambda: ['stabilityai/stable-diffusion-3.5-large', 'stabilityai/stable-diffusion-3.5-large-turbo', 'stabilityai/stable-diffusion-3.5-medium'])
    Flux: List[str] = field(default_factory=lambda: ['black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.1-schnell'])

class TextToImagePipelineSD3:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: StableDiffusion3Pipeline | None = None
        self.device: str | None = None

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-large"
            logger.info("Loading CUDA")
            self.device = "cuda"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-medium"
            logger.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class TextToImagePipelineFlux:
    def __init__(self, model_path: str | None = None, low_vram: bool = False):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: FluxPipeline | None = None
        self.device: str | None = None
        self.low_vram = low_vram

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-schnell"
            logger.info("Loading CUDA")
            self.device = "cuda" 
            self.pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
            if self.low_vram:
                self.pipeline.enable_model_cpu_offload()
            else:
                pass
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-schnell"
            logger.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class ModelPipelineInitializer:
    def __init__(self, model: str = '', type_models: str = 't2im'):
        self.model = model
        self.type_models = type_models
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model_type = None

    def initialize_pipeline(self):
        if not self.model:
            raise ValueError("Model name not provided")

        # Check if model exists in PresetModels
        preset_models = PresetModels()

        # Determine which model type we're dealing with
        if self.model in preset_models.SD3:
            self.model_type = "SD3"
        elif self.model in preset_models.SD3_5:
            self.model_type = "SD3_5"
        elif self.model in preset_models.Flux:
            self.model_type = "Flux"

        # Create appropriate pipeline based on model type and type_models
        if self.type_models == 't2im':
            if self.model_type in ["SD3", "SD3_5"]:
                self.pipeline = TextToImagePipelineSD3(self.model)
            elif self.model_type == "Flux":
                self.pipeline = TextToImagePipelineFlux(self.model)
            else:
                raise ValueError(f"Model type {self.model_type} not supported for text-to-image")
        elif self.type_models == 't2v':
            raise ValueError(f"Unsupported type_models: {self.type_models}")

        return self.pipeline