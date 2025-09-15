from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse  
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from .Pipelines import TextToImagePipelineSD3, TextToImagePipelineFlux, TextToImagePipelineSD, logger
import logging
from ..utils import RequestScopedPipeline
from diffusers import *
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
from contextlib import asynccontextmanager
import asyncio

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

    server_config = config or ServerConfigModels()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO)
        app.state.logger = logging.getLogger("diffusers-server")
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

        app.state.total_requests = 0
        app.state.active_inferences = 0
        app.state.metrics_lock = asyncio.Lock()
        app.state.metrics_task = None

        app.state.utils_app = Utils(
            host=server_config.host,
            port=server_config.port,
        )

        async def metrics_loop():
            try:
                while True:
                    async with app.state.metrics_lock:
                        total = app.state.total_requests
                        active = app.state.active_inferences
                    app.state.logger.info(f"[METRICS] total_requests={total} active_inferences={active}")
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                app.state.logger.info("Metrics loop cancelled")
                raise

        app.state.metrics_task = asyncio.create_task(metrics_loop())

        try:
            yield
        finally:
            task = app.state.metrics_task
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            try:
                stop_fn = getattr(model_pipeline, "stop", None) or getattr(model_pipeline, "close", None)
                if callable(stop_fn):
                    await run_in_threadpool(stop_fn)
            except Exception as e:
                app.state.logger.warning(f"Error during pipeline shutdown: {e}")

            app.state.logger.info("Lifespan shutdown complete")

    app = FastAPI(lifespan=lifespan)

    logger = logging.getLogger("DiffusersServer.Pipelines")

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
        request_pipe = None
        pipeline_lock = threading.Lock()

    else:
        initializer = ModelPipelineInitializer(
            model=server_config.model,
            type_models=server_config.type_models,
        )
        model_pipeline = initializer.initialize_pipeline()
        model_pipeline.start()

        request_pipe = RequestScopedPipeline(model_pipeline.pipeline)
        pipeline_lock = threading.Lock()

    logger.info(f"Pipeline initialized and ready to receive requests (model ={server_config.model})")

    app.state.MODEL_INITIALIZER = initializer
    app.state.MODEL_PIPELINE = model_pipeline
    app.state.REQUEST_PIPE = request_pipe
    app.state.PIPELINE_LOCK = pipeline_lock

    class JSONBodyQueryAPI(BaseModel):
        model : str | None = None
        prompt : str
        negative_prompt : str | None = None
        num_inference_steps : int = 28
        num_images_per_prompt : int = 1

    @app.middleware("http")
    async def count_requests_middleware(request: Request, call_next):
        async with app.state.metrics_lock:
            app.state.total_requests += 1
        response = await call_next(request)
        return response


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

        utils_app = app.state.utils_app


        if not wrapper or not wrapper.pipeline:
            raise HTTPException(500, "Model not initialized correctly")
        if not prompt.strip():
            raise HTTPException(400, "No prompt provided")


        def make_generator():
            g = torch.Generator(device=initializer.device)
            return g.manual_seed(random.randint(0, 10_000_000))

        req_pipe = app.state.REQUEST_PIPE

        def infer():
            gen = make_generator()
            return req_pipe.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=gen,
                num_inference_steps=num_steps,
                num_images_per_prompt=num_images_per_prompt,
                device=initializer.device,
                output_type="pil",
            )

        try:
            async with app.state.metrics_lock:
                app.state.active_inferences += 1

            output = await run_in_threadpool(infer)

            async with app.state.metrics_lock:
                app.state.active_inferences = max(0, app.state.active_inferences - 1)
                
            urls = [utils_app.save_image(img) for img in output.images]
            return {"response": urls}

        except Exception as e:
            async with app.state.metrics_lock:
                app.state.active_inferences = max(0, app.state.active_inferences - 1)
            logger.error(f"Error during inference: {e}")
            raise HTTPException(500, f"Error in processing: {e}")

        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.ipc_collect()
            gc.collect()


    @app.get("/images/{filename}")
    async def serve_image(filename: str):
        utils_app = app.state.utils_app
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