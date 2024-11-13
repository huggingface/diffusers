
## OpenAI Compatible `/v1/images/generations` Server

This is a concurrent, multithreaded solution for running a server that can generate images using the `diffusers` library. This examples uses the Stable Diffusion 3 pipeline, but you can use any pipeline that you would like by swapping out the model and pipeline to be the ones that you want to use.

### Installing Dependencies

Start by going to the base of the repo and installing it with:
``py
pip install .
```

The pipeline can then have its dependencies installed with:
```py
pip install -f requirements.txt
```

### Running the server

This server can be run with:
```py
python server.py
```
The server will be spun up at http://localhost:8000. You can `curl` this model with the following command:
```
curl -X POST -H "Content-Type: application/json" --data '{"model": "something", "prompt": "a kitten in front of a fireplace"}' http://localhost:8000/v1/images/generations
```

### Upgrading Dependencies

If you need to upgrade some dependencies, you can do that with either [pip-tools](https://github.com/jazzband/pip-tools) or [uv](https://github.com/astral-sh/uv). With `uv`, this looks like:
```
uv pip compile requirements.in -o requirements.txt
```

