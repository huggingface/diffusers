from ...utils import is_torch_available, is_transformers_available


if is_torch_available():
    from .single_file_model import FromOriginalModelMixin

    if is_transformers_available():
        from .single_file import FromSingleFileMixin
