
# Create a server

Diffusers' pipelines can be used as an inference engine for a server. It supports concurrent and multithreaded requests to generate images that may be requested by multiple users at the same time.

This guide will show you how to use the [`StableDiffusion3Pipeline`] in a server, but feel free to use any pipeline you want.


Start by navigating to the `examples/server` folder and installing all of the dependencies.

``py
pip install .
pip install -f requirements.txt
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

## How does this Server Work?

The server is built with [FastAPI](https://fastapi.tiangolo.com/async/). The endpoint for `v1/images/generations` is defined like this:
```py
@app.post("/v1/images/generations")
async def generate_image(image_input: TextToImageInput):
    try:
        loop = asyncio.get_event_loop()
        scheduler = shared_pipeline.pipeline.scheduler.from_config(shared_pipeline.pipeline.scheduler.config)
        pipeline = StableDiffusion3Pipeline.from_pipe(shared_pipeline.pipeline, scheduler=scheduler)
        generator =torch.Generator(device="cuda")
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
Above, the `generate_image` function is defined as asynchronous with the `async` keyword so that [FastAPI](https://fastapi.tiangolo.com/async/) knows that whatever is happening in this function is not going to necessarily return a result right away. Once it hits some point in the function that it needs to await some other [Task](https://docs.python.org/3/library/asyncio-task.html#asyncio.Task), the main thread goes back to answering other HTTP requests. For us, this happens when it hits this part of the function:
```py
output = await loop.run_in_executor(None, lambda: pipeline(image_input.prompt, generator = generator))
```
At this point, we are tossing the execution of the pipeline function [onto a new thread](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor) and the main thread knows to go and do some other things until a result is returned from the `pipeline`.

Another important aspect of this implementation is the portion which creates a Pipeline from the `shared_pipeline`. The goal behind this is to avoid loading the underlying model more than once into the GPU while still allowing for each new request that is running on its own thread to have its own generator and scheduler. The scheduler in particular, at the time of this writing (November 2024), is not thread safe, and it will cause errors like: `IndexError: index 21 is out of bounds for dimension 0 with size 21` if you do try to use the same scheduler across multiple threads.
