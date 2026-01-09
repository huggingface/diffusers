
# Create a server

Diffusers' pipelines can be used as an inference engine for a server. It supports concurrent and multithreaded requests to generate images that may be requested by multiple users at the same time.

This guide will show you how to use the [`StableDiffusion3Pipeline`] in a server, but feel free to use any pipeline you want.


Start by navigating to the `examples/server` folder and installing all of the dependencies.

```py
pip install diffusers
pip install -r requirements.txt
```

Launch the server with the following command.

```py
python server.py
```

The server is accessed at http://localhost:8000. You can curl this model with the following command.
```
curl -X POST -H "Content-Type: application/json" --data '{"model": "something", "prompt": "a kitten in front of a fireplace"}' http://localhost:8000/v1/images/generations
```

If you need to upgrade some dependencies, you can use either [pip-tools](https://github.com/jazzband/pip-tools) or [uv](https://github.com/astral-sh/uv). For example, upgrade the dependencies with `uv` using the following command.

```
uv pip compile requirements.in -o requirements.txt
```


The server is built with [FastAPI](https://fastapi.tiangolo.com/async/). The endpoint for `v1/images/generations` is shown below.
```py
@app.post("/v1/images/generations")
async def generate_image(image_input: TextToImageInput):
    try:
        loop = asyncio.get_event_loop()
        scheduler = shared_pipeline.pipeline.scheduler.from_config(shared_pipeline.pipeline.scheduler.config)
        pipeline = StableDiffusion3Pipeline.from_pipe(shared_pipeline.pipeline, scheduler=scheduler)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(random.randint(0, 10000000))
        output = await loop.run_in_executor(None, lambda: pipeline(image_input.prompt, generator = generator))
        logger.info(f"output: {output}")
        image_url = save_image(output.images[0])
        return {"data": [{"url": image_url}]}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        elif hasattr(e, 'message'):
            raise HTTPException(status_code=500, detail=e.message + traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e) + traceback.format_exc())
```
The `generate_image` function is defined as asynchronous with the [async](https://fastapi.tiangolo.com/async/) keyword so that FastAPI knows that whatever is happening in this function won't necessarily return a result right away. Once it hits some point in the function that it needs to await some other [Task](https://docs.python.org/3/library/asyncio-task.html#asyncio.Task), the main thread goes back to answering other HTTP requests. This is shown in the code below with the [await](https://fastapi.tiangolo.com/async/#async-and-await) keyword.
```py
output = await loop.run_in_executor(None, lambda: pipeline(image_input.prompt, generator = generator))
```
At this point, the execution of the pipeline function is placed onto a [new thread](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor), and the main thread performs other things until a result is returned from the `pipeline`.

Another important aspect of this implementation is creating a `pipeline` from `shared_pipeline`. The goal behind this is to avoid loading the underlying model more than once onto the GPU while still allowing for each new request that is running on a separate thread to have its own generator and scheduler. The scheduler, in particular, is not thread-safe, and it will cause errors like: `IndexError: index 21 is out of bounds for dimension 0 with size 21` if you try to use the same scheduler across multiple threads.
