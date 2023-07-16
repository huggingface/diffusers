from ...utils import is_faiss_available, is_torch_available, is_transformers_available


if is_transformers_available() and is_torch_available():
    from .pipeline_rdm import RDMPipeline
if is_transformers_available() and is_torch_available() and is_faiss_available():
    from .retriever import Index, IndexConfig, Retriever
