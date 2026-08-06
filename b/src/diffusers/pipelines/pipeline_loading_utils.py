

def _fetch_class_library_tuple(module):
    # Guard against non-module values (bool, int, str, etc.) that can be passed
    # when a positional argument shift occurs in subclasses (e.g. issue #6969).
    if not isinstance(module, (torch.nn.Module, type)):
        return (None, None)

    # import it here to avoid circular import
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    pipelines = getattr(diffusers_module, "pipelines")
