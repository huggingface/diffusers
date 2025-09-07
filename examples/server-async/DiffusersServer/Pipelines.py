# from https://github.com/F4k3r22/DiffusersServer/blob/main/DiffusersServer/Pipelines.py

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import torch
import os
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TextToImageInput(BaseModel):
    model: str
    prompt: str
    size: str | None = None
    n: int | None = None

class TextToImagePipelineSD3:
    def __init__(self, model_path: str | None = None):
        """
        Inicialización de la clase con la ruta del modelo.
        Si no se proporciona, se obtiene de la variable de entorno.
        """
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: StableDiffusion3Pipeline = None
        self.device: str = None

    def start(self):
        """
        Inicia el pipeline cargando el modelo en CUDA o MPS según esté disponible.
        Se utiliza la ruta del modelo definida en el __init__ y se asigna un valor predeterminado
        en función del dispositivo disponible si no se definió previamente.
        """
        if torch.cuda.is_available():
            # Si no se definió model_path, se asigna el valor por defecto para CUDA.
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-large"
            logger.info("Loading CUDA")
            self.device = "cuda"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
        elif torch.backends.mps.is_available():
            # Si no se definió model_path, se asigna el valor por defecto para MPS.
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
        """
        Inicialización de la clase con la ruta del modelo.
        Si no se proporciona, se obtiene de la variable de entorno.
        """
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: FluxPipeline = None
        self.device: str = None
        self.low_vram = low_vram

    def start(self):
        if torch.cuda.is_available():
            # Si no se definió model_path, se asigna el valor por defecto para CUDA.
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
            # Si no se definió model_path, se asigna el valor por defecto para MPS.
            model_path = self.model_path or "black-forest-labs/FLUX.1-schnell"
            logger.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class TextToImagePipelineSD:
    def __init__(self, model_path: str | None = None):
        """
        Inicialización de la clase con la ruta del modelo.
        Si no se proporciona, se obtiene de la variable de entorno.
        """
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: StableDiffusionPipeline = None
        self.device: str = None

    def start(self):
        if torch.cuda.is_available():
            # Si no se definió model_path, se asigna el valor por defecto para CUDA.
            model_path = self.model_path or "sd-legacy/stable-diffusion-v1-5"
            logger.info("Loading CUDA")
            self.device = "cuda" 
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
        elif torch.backends.mps.is_available():
            # Si no se definió model_path, se asigna el valor por defecto para MPS.
            model_path = self.model_path or "sd-legacy/stable-diffusion-v1-5"
            logger.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")