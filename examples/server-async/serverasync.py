import asyncio
import gc
import logging
import os
import random
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from Pipelines import ModelPipelineInitializer
from pydantic import BaseModel

from utils import RequestScopedPipeline, Utils


@dataclass
class ServerConfigModels:
    model: str = "stabilityai/stable-diffusion-3.5-medium"
    type_models: str = "t2im"
    constructor_pipeline: Optional[Type] = None
    custom_pipeline: Optional[Type] = None
    components: Optional[Dict[str, Any]] = None
    torch_dtype: Optional[torch.dtype] = None
    host: str = "0.0.0.0"
    port: int = 8500


server_config = ServerConfigModels()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    app.state.logger = logging.getLogger("diffusers-server")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

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
    model: str | None = None
    prompt: str
    negative_prompt: str | None = None
    num_inference_steps: int = 28
    num_images_per_prompt: int = 1


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
    prompt = json.prompt
    negative_prompt = json.negative_prompt or ""
    num_steps = json.num_inference_steps
    num_images_per_prompt = json.num_images_per_prompt

    wrapper = app.state.MODEL_PIPELINE
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


@app.get("/api/status")
async def get_status():
    memory_info = {}
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        memory_info = {
            "memory_allocated_gb": round(memory_allocated, 2),
            "memory_reserved_gb": round(memory_reserved, 2),
            "device": torch.cuda.get_device_name(0),
        }

    return {"current_model": server_config.model, "type_models": server_config.type_models, "memory": memory_info}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=server_config.host, port=server_config.port)
