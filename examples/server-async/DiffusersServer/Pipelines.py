# Pipelines.py

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import torch
import os
import logging
from pydantic import BaseModel
import gc

logger = logging.getLogger(__name__)

class TextToImageInput(BaseModel):
    model: str
    prompt: str
    size: str | None = None
    n: int | None = None

class TextToImagePipelineSD3:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: StableDiffusion3Pipeline | None = None
        self.device: str | None = None
        
    def start(self):
        torch.set_float32_matmul_precision("high")
        
        if hasattr(torch._inductor, 'config'):
            if hasattr(torch._inductor.config, 'conv_1x1_as_mm'):
                torch._inductor.config.conv_1x1_as_mm = True
            if hasattr(torch._inductor.config, 'coordinate_descent_tuning'):
                torch._inductor.config.coordinate_descent_tuning = True
            if hasattr(torch._inductor.config, 'epilogue_fusion'):
                torch._inductor.config.epilogue_fusion = False
            if hasattr(torch._inductor.config, 'coordinate_descent_check_all_directions'):
                torch._inductor.config.coordinate_descent_check_all_directions = True
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
        
        
        if torch.cuda.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-large"
            logger.info(f"Loading CUDA with model: {model_path}")
            self.device = "cuda"
            
            torch.cuda.empty_cache()
            gc.collect()
            
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16" if "fp16" in model_path else None,
                low_cpu_mem_usage=True,
            )
            
            self.pipeline = self.pipeline.to(device=self.device)
            
            if hasattr(self.pipeline, 'transformer') and self.pipeline.transformer is not None:
                self.pipeline.transformer = self.pipeline.transformer.to(
                    memory_format=torch.channels_last
                )
                logger.info("Transformer optimized with channels_last format")
            
            if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None:
                self.pipeline.vae = self.pipeline.vae.to(
                    memory_format=torch.channels_last
                )
                logger.info("VAE optimized with channels_last format")
            
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("XFormers memory efficient attention enabled")
            except Exception as e:
                logger.info(f"XFormers not available: {e}")
            
            # --- Se descarta torch.compile pero se mantiene el resto ---
            if torch.__version__ >= "2.0.0":
                logger.info("Skipping torch.compile - running without compile optimizations by design")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("CUDA pipeline fully optimized and ready")
            
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-medium"
            logger.info(f"Loading MPS for Mac M Series with model: {model_path}")
            self.device = "mps"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            ).to(device=self.device)

            if hasattr(self.pipeline, 'transformer') and self.pipeline.transformer is not None:
                self.pipeline.transformer = self.pipeline.transformer.to(
                    memory_format=torch.channels_last
                )
            
            if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None:
                self.pipeline.vae = self.pipeline.vae.to(
                    memory_format=torch.channels_last
                )
            
                
            logger.info("MPS pipeline optimized and ready")
            
        else:
            raise Exception("No CUDA or MPS device available")
        
        # OPTIONAL WARMUP
        self._warmup()
        
        logger.info("Pipeline initialization completed successfully")
    
    def _warmup(self):
        if self.pipeline:
            logger.info("Running warmup inference...")
            with torch.no_grad():
                _ = self.pipeline(
                    prompt="warmup",
                    num_inference_steps=1,
                    height=512,
                    width=512,
                    guidance_scale=1.0,
                )
            torch.cuda.empty_cache() if self.device == "cuda" else None
            logger.info("Warmup completed")

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