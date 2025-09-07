# from https://github.com/F4k3r22/DiffusersServer/blob/main/DiffusersServer/serverasync.py

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse  
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from .Pipelines import TextToImagePipelineSD3, TextToImagePipelineFlux, TextToImagePipelineSD
import logging
from diffusers.utils.export_utils import export_to_video
from diffusers.pipelines.pipeline_utils import RequestScopedPipeline
from diffusers import *
from .superpipeline import *
import random
import uuid
import tempfile
from dataclasses import dataclass
import os
import torch
import threading
import gc
from typing import Optional, Dict, Any, Type
from dataclasses import dataclass, field
from typing import List

@dataclass
class PresetModels:
    SD3: List[str] = field(default_factory=lambda: ['stabilityai/stable-diffusion-3-medium'])
    SD3_5: List[str] = field(default_factory=lambda: ['stabilityai/stable-diffusion-3.5-large', 'stabilityai/stable-diffusion-3.5-large-turbo', 'stabilityai/stable-diffusion-3.5-medium'])
    Flux: List[str] = field(default_factory=lambda: ['black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.1-schnell'])

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
        else:
            self.model_type = "SD"

        # Create appropriate pipeline based on model type and type_models
        if self.type_models == 't2im':
            if self.model_type in ["SD3", "SD3_5"]:
                self.pipeline = TextToImagePipelineSD3(self.model)
            elif self.model_type == "Flux":
                self.pipeline = TextToImagePipelineFlux(self.model)
            elif self.model_type == "SD":
                self.pipeline = TextToImagePipelineSD(self.model)
            else:
                raise ValueError(f"Model type {self.model_type} not supported for text-to-image")
        elif self.type_models == 't2v':
            raise ValueError(f"Unsupported type_models: {self.type_models}")

        return self.pipeline

class Utils:
    def __init__(self, host: str = '0.0.0.0', port: int = 8500):
        self.service_url = f"http://{host}:{port}"
        self.image_dir = os.path.join(tempfile.gettempdir(), "images")
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.video_dir = os.path.join(tempfile.gettempdir(), "videos")
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def save_image(self, image):
        if hasattr(image, "to"):
            try:
                image = image.to("cpu")
            except Exception:
                pass

        if isinstance(image, torch.Tensor):
            from torchvision import transforms
            to_pil = transforms.ToPILImage()
            image = to_pil(image.squeeze(0).clamp(0, 1))

        filename = "img" + str(uuid.uuid4()).split("-")[0] + ".png"
        image_path = os.path.join(self.image_dir, filename)
        logger.info(f"Saving image to {image_path}")

        image.save(image_path, format="PNG", optimize=True)

        del image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return os.path.join(self.service_url, "images", filename)

    def save_video(self, video, fps):
        filename = "video" + str(uuid.uuid4()).split("-")[0] + ".mp4"
        video_path = os.path.join(self.video_dir, filename)
        export = export_to_video(video, video_path, fps=fps)
        logger.info(f"Saving video to {video_path}")
        return os.path.join(self.service_url, "video", filename)

@dataclass
class ServerConfigModels:
    model: str = 'stabilityai/stable-diffusion-3-medium' 
    type_models: str = 't2im' 
    custom_model : bool = False
    constructor_pipeline: Optional[Type] = None
    custom_pipeline: Optional[Type] = None  
    components: Optional[Dict[str, Any]] = None
    api_name: Optional[str] = 'custom_api'
    torch_dtype: Optional[torch.dtype] = None
    host: str = '0.0.0.0' 
    port: int = 8500

def create_app_fastapi(config: ServerConfigModels) -> FastAPI:
    app = FastAPI()

    class JSONBodyQueryAPI(BaseModel):
        model : str | None = None
        prompt : str
        negative_prompt : str | None = None
        num_inference_steps : int = 28
        num_images_per_prompt : int = 1

    logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger(__name__)

    server_config = config or ServerConfigModels()
    app.state.SERVER_CONFIG = server_config

    global utils_app

    utils_app = Utils(host=server_config.host, port=server_config.port)

    logger.info(f"Inicializando pipeline para el modelo: {server_config.model}")
    try:
        if server_config.custom_model:
            if server_config.constructor_pipeline is None:
                raise ValueError("constructor_pipeline cannot be None - a valid pipeline constructor is required")
            initializer = server_config.constructor_pipeline(
                model_path=server_config.model,
                pipeline=server_config.custom_pipeline,
                torch_dtype=server_config.torch_dtype,
                components=server_config.components,
            )
            model_pipeline = initializer.start()
            app.state.CUSTOM_PIPELINE = server_config.custom_pipeline
            app.state.MODEL_PIPELINE = model_pipeline
            app.state.MODEL_INITIALIZER = initializer
            logger.info(f"Pipeline personalizado inicializado. Tipo: {type(model_pipeline)}")
        else:
            initializer = ModelPipelineInitializer(
                model=server_config.model,
                type_models=server_config.type_models,
            )
            model_pipeline = initializer.initialize_pipeline()
            model_pipeline.start()

            app.state.REQUEST_PIPE = RequestScopedPipeline(model_pipeline.pipeline)

            # Lock for concurrency
            pipeline_lock = threading.Lock()

            app.state.MODEL_PIPELINE = model_pipeline
            app.state.PIPELINE_LOCK = pipeline_lock
            app.state.MODEL_INITIALIZER = initializer

        logger.info("Pipeline initialized and ready to receive requests")
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")
        raise


    @app.get("/")
    async def root():
        return {"message": "Welcome to the Diffusers Server"}

    @app.post("/api/diffusers/inference")
    async def api(json: JSONBodyQueryAPI):
        prompt                = json.prompt
        negative_prompt       = json.negative_prompt or ""
        num_steps             = json.num_inference_steps
        num_images_per_prompt = json.num_images_per_prompt

        wrapper     = app.state.MODEL_PIPELINE
        initializer = app.state.MODEL_INITIALIZER


        if not wrapper or not wrapper.pipeline:
            raise HTTPException(500, "Modelo no inicializado correctamente")
        if not prompt.strip():
            raise HTTPException(400, "No se proporcionó prompt")

        def make_generator():
            g = torch.Generator(device=initializer.device)
            return g.manual_seed(random.randint(0, 10_000_000))

        req_pipe = app.state.REQUEST_PIPE

        def infer():
            # This is called that because the RequestScoped Pipeline already internally 
            # handles everything necessary for inference and only the 
            # model pipeline needs to be passed, for example StableDiffusion3Pipeline
            gen = make_generator()
            return req_pipe.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=gen,
                num_inference_steps=num_steps,
                num_images_per_prompt=num_images_per_prompt,
                device=initializer.device
            )

        try:
            output = await run_in_threadpool(infer)

            urls = [utils_app.save_image(img) for img in output.images]
            return {"response": urls}

        except Exception as e:
            logger.error(f"Error durante la inferencia: {e}")
            raise HTTPException(500, f"Error en procesamiento: {e}")

        finally:
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    @app.get("/images/{filename}")
    async def serve_image(filename: str):
        file_path = os.path.join(utils_app.image_dir, filename)
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(file_path, media_type="image/png")

    @app.get("/api/models")
    async def list_models():
        return {
            "current_model" : server_config.model,
            "type" : server_config.type_models,
            "all_models": {
                "type": "T2Img",
                "SD3": PresetModels().SD3,
                "SD3_5": PresetModels().SD3_5,
                "Flux": PresetModels().Flux,
            }
        }

    @app.get("/api/status")
    async def get_status():
        memory_info = {}
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            memory_info = {
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
                "device": torch.cuda.get_device_name(0)
            }

        return {
            "current_model" : server_config.model,
            "type_models" : server_config.type_models,
            "memory" : memory_info}
        

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app