# from https://github.com/F4k3r22/DiffusersServer/blob/main/DiffusersServer/create_server.py

from .Pipelines import *
from .serverasync import *
from .uvicorn_diffu import *
import asyncio

def create_inference_server_Async(
    model:str,
    type_model: str = 't2im',
    host: str = '0.0.0.0',
    port: int = 8500,
    threads=5,
    enable_memory_monitor=True,
    custom_model: bool = False,
    custom_pipeline: Optional[Type] | None = None,
    constructor_pipeline: Optional[Type] | None = None,
    components: Optional[Dict[str, Any]] = None,
    api_name: Optional[str] = 'custom_api',
    torch_dtype = torch.bfloat16
):
    config = ServerConfigModels(
        model=model,
        type_models=type_model,
        custom_model=custom_model,
        custom_pipeline=custom_pipeline,
        constructor_pipeline=constructor_pipeline,
        components=components,
        api_name=api_name,
        torch_dtype=torch_dtype,
        host=host,
        port=port
    )

    app = create_app_fastapi(config)

    asyncio.run(run_uvicorn_server(
        app, 
        host=host, 
        port=port, 
        workers=threads,
        enable_memory_monitor=enable_memory_monitor
    ))

    return app