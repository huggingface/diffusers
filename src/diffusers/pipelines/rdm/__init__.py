from ...utils import is_torch_available, is_transformers_available, is_faiss_available


if is_transformers_available() and is_torch_available() and is_faiss_available():
    from .pipeline_rdm import RDMPipeline
    from .retriever import IndexConfig, Index, Retriever
