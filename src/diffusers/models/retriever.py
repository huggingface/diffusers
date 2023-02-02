"""
Idea for structure
 Retriever aggregates an Index class and a RetrieverConfig class
 The Index class aggregates a Dataset and RetrieverConfig class
 from_pretrained in the retriever's class, it takes in a huggingface path to a dataset, optional path to an index file+config file in huggingface if there is one
 If an index file is provided, add that index to the dataset.
 If the dataset doesn't have the column embedding or a corresponding index file, in the Index class, the index is computed based on the clip model defined in the config. Then add that to the index of the dataset. This is done in the Index class
 In retrieve we just call the retrieve method in the Index class that gets knn based on the faiss embedding.
 In the save_pretrained method, save index using save_faiss_index. Save this dataset along with config.
 The call method will just call retrieve.
 I'll also have a way to pass the clip model and its components via default arguments.
 Test save_pretrained and from_pretrained methods on new dataset.
"""

from transformers import CLIPModel, CLIPFeatureExtractor, CLIPTokenizer
from datasets import load_dataset, Image, load_dataset_builder, load_from_disk, Dataset
import torch
from typing import Callable, List, Optional, Union
import numpy as np
from ..utils import deprecate, logging
from transformers.models.rag.retrieval_rag import LegacyIndex, CustomHFIndex, CanonicalHFIndex
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
from diffusers.pipelines.rdm import preprocess_images


class IndexConfig:
    def __init__(self, clip_name_or_path="openai/clip-vit-large-patch14", dataset_name="Isamu136/oxford_pets_with_l14_emb", \
                 image_column="image", index_name="embeddings", index_path=None):
        self.clip_name_or_path = clip_name_or_path
        self.dataset_name = dataset_name
        self.image_column = image_column
        self.index_name = index_name
        self.index_path = index_path

class Index:
    """
    Each index for a retrieval model is specific to the clip model used and the dataset used.
    """
    def __init__(self, config:IndexConfig, dataset: Dataset, clip_model:CLIPModel=None, \
                 feature_extractor:CLIPFeatureExtractor=None, tokenizer:CLIPTokenizer=None):
        self.config = config
        self.dataset = dataset
        self.index_initialized = False
        self.clip_model = clip_model
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.index_name = config.index_name
        self.index_path = config.index_path
        self.init_index()
    def set_index_name(self, index_name:str):
        self.index_name = index_name
    def init_index(self):
        if not self.index_initialized:
            if self.index_path and self.index_name:
                try:
                    self.dataset.load_faiss_index(self.index_name, self.index_path)
                    self.index_initialized = True
                except FileNotFoundError as e:
                    raise FileNotFoundError("Invalid index name")
            if self.index_name in self.dataset.features:
                self.dataset.add_faiss_index(column=self.index_name)
                self.index_initialized = True
    def build_index(self, device:str="cuda", torch_dtype=torch.float32):
        if not self.index_initialized:
            self.clip_model = self.clip_model or CLIPModel.from_pretrained(self.config.clip_name_or_path).to(device=device, dtype=torch_dtype)
            self.feature_extractor = self.feature_extractor or CLIPFeatureExtractor.from_pretrained(self.config.clip_name_or_path).to(device=device, dtype=torch_dtype)
            self.dataset = get_dataset_with_emb(self.dataset, self.clip_model, self.feature_extractor, device=device, image_column=self.config.image_column, embedding_column=self.config.embeddings_column)
            self.init_index()
    def get_knn(self, vec, k=20):
        vec = np.array(vec).astype(np.float32)
        return self.dataset.get_nearest_examples(self.index_name, vec, k=k)
    def get_knn_from_text(self, prompt, k=20):
        vec = map_txt_to_clip_feature(self.clip_model, self.tokenizer, prompt, self.device)
        return self.get_knn(vec, k)

class Retriever:
    def __init__(self, config:IndexConfig, index:Index=None):
        self.config = config
        self.index = index or self._build_index(config)
    @classmethod
    def from_pretrained(cls, retriever_name_or_path:str, dataset:Dataset=None, clip_model:CLIPModel=None,\
                         feature_extractor:CLIPFeatureExtractor=None, tokenizer:CLIPTokenizer=None, **kwargs):
        config = kwargs.pop("config", None) or IndexConfig.from_pretrained(retriever_name_or_path, **kwargs)
        clip_model = clip_model or CLIPModel.from_pretrained(config.clip_name_or_path)
        tokenizer = tokenizer or CLIPTokenizer.from_pretrained(config.clip_name_or_path)
        feature_extractor = feature_extractor or CLIPFeatureExtractor.from_pretrained(config.clip_name_or_path)
        dataset = dataset or load_dataset(config.dataset_name)
        index =cls._build_index(config, dataset, clip_model=clip_model, feature_extractor=feature_extractor, tokenizer=tokenizer)
        return cls(
            config,
            index=index
        )
    @staticmethod
    def _build_index(config:IndexConfig, dataset:Dataset=None, clip_model:CLIPModel=None,\
                         feature_extractor:CLIPFeatureExtractor=None, tokenizer:CLIPTokenizer=None):
        dataset = dataset or load_dataset(config.dataset_name)
        index =Index(config, dataset, clip_model=clip_model, feature_extractor=feature_extractor, tokenizer=tokenizer)
        index.build_index()
        return index

    def save_pretrained(self, save_directory):
        if isinstance(self.index, CustomHFIndex):
            if self.config.index_path is None:
                index_path = os.path.join(save_directory, "hf_dataset_index.faiss")
                self.index.dataset.get_index("embeddings").save(index_path)
                self.config.index_path = index_path
            if self.config.passages_path is None:
                passages_path = os.path.join(save_directory, "hf_dataset")
                # datasets don't support save_to_disk with indexes right now
                faiss_index = self.index.dataset._indexes.pop("embeddings")
                self.index.dataset.save_to_disk(passages_path)
                self.index.dataset._indexes["embeddings"] = faiss_index
                self.config.passages_path = passages_path
        self.config.save_pretrained(save_directory)
        rag_tokenizer = RagTokenizer(
            question_encoder=self.question_encoder_tokenizer,
            generator=self.generator_tokenizer,
        )
        rag_tokenizer.save_pretrained(save_directory)

    def init_retrieval(self):
        """
        Retriever initialization function. It loads the index into memory.
        """

        logger.info("initializing retrieval")
        self.index.init_index()

    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
        r"""
        Postprocessing retrieved `docs` and combining them with `input_strings`.
        Args:
            docs  (`dict`):
                Retrieved documents.
            input_strings (`str`):
                Input strings decoded by `preprocess_query`.
            prefix (`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.
        Return:
            `tuple(tensors)`: a tuple consisting of two elements: contextualized `input_ids` and a compatible
            `attention_mask`.
        """

        def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
            # TODO(Patrick): if we train more RAG models, I want to put the input first to take advantage of effortless truncation
            # TODO(piktus): better handling of truncation
            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]
            if prefix is None:
                prefix = ""
            out = (prefix + doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string).replace(
                "  ", " "
            )
            return out

        rag_input_strings = [
            cat_input_and_doc(
                docs[i]["title"][j],
                docs[i]["text"][j],
                input_strings[i],
                prefix,
            )
            for i in range(len(docs))
            for j in range(n_docs)
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=self.config.max_combined_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def _chunk_tensor(self, t: Iterable, chunk_size: int) -> List[Iterable]:
        return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]

    def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray]:
        question_hidden_states_batched = self._chunk_tensor(question_hidden_states, self.batch_size)
        ids_batched = []
        vectors_batched = []
        for question_hidden_states in question_hidden_states_batched:
            start_time = time.time()
            ids, vectors = self.index.get_top_docs(question_hidden_states, n_docs)
            logger.debug(
                f"index search time: {time.time() - start_time} sec, batch size {question_hidden_states.shape}"
            )
            ids_batched.extend(ids)
            vectors_batched.extend(vectors)
        return (
            np.array(ids_batched),
            np.array(vectors_batched),
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:
        """
        Retrieves documents for specified `question_hidden_states`.
        Args:
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`):
                A batch of query vectors to retrieve with.
            n_docs (`int`):
                The number of docs retrieved per query.
        Return:
            `Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:
            - **retrieved_doc_embeds** (`np.ndarray` of shape `(batch_size, n_docs, dim)`) -- The retrieval embeddings
              of the retrieved docs per query.
            - **doc_ids** (`np.ndarray` of shape `(batch_size, n_docs)`) -- The ids of the documents in the index
            - **doc_dicts** (`List[dict]`): The `retrieved_doc_embeds` examples per query.
        """

        doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
        return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids)

    def set_ctx_encoder_tokenizer(self, ctx_encoder_tokenizer: PreTrainedTokenizer):
        # used in end2end retriever training
        self.ctx_encoder_tokenizer = ctx_encoder_tokenizer
        self.return_tokenized_docs = True

    def __call__(
        self,
        question_input_ids: List[List[int]],
        question_hidden_states: np.ndarray,
        prefix=None,
        n_docs=None,
        return_tensors=None,
    ) -> BatchEncoding:
        """
        Retrieves documents for specified `question_hidden_states`.
        Args:
            question_input_ids: (`List[List[int]]`) batch of input ids
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix: (`str`, *optional*):
                The prefix used by the generator's tokenizer.
            n_docs (`int`, *optional*):
                The number of docs retrieved per query.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        Returns: [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
            - **context_input_ids** -- List of token ids to be fed to a model.
              [What are input IDs?](../glossary#input-ids)
            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model
            (when `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).
              [What are attention masks?](../glossary#attention-mask)
            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        """

        n_docs = n_docs if n_docs is not None else self.n_docs
        prefix = prefix if prefix is not None else self.config.generator.prefix
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)

        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
        context_input_ids, context_attention_mask = self.postprocess_docs(
            docs, input_strings, prefix, n_docs, return_tensors=return_tensors
        )

        if self.return_tokenized_docs:
            retrieved_doc_text = []
            retrieved_doc_title = []

            for b_idx in range(len(docs)):
                for doc_idx in range(n_docs):
                    retrieved_doc_text.append(docs[b_idx]["text"][doc_idx])
                    retrieved_doc_title.append(docs[b_idx]["title"][doc_idx])

            tokenized_docs = self.ctx_encoder_tokenizer(
                retrieved_doc_title,
                retrieved_doc_text,
                truncation=True,
                padding="longest",
                return_tensors=return_tensors,
            )

            return BatchEncoding(
                {
                    "context_input_ids": context_input_ids,
                    "context_attention_mask": context_attention_mask,
                    "retrieved_doc_embeds": retrieved_doc_embeds,
                    "doc_ids": doc_ids,
                    "tokenized_doc_ids": tokenized_docs["input_ids"],
                    "tokenized_doc_attention_mask": tokenized_docs["attention_mask"],
                },
                tensor_type=return_tensors,
            )

        else:
            return BatchEncoding(
                {
                    "context_input_ids": context_input_ids,
                    "context_attention_mask": context_attention_mask,
                    "retrieved_doc_embeds": retrieved_doc_embeds,
                    "doc_ids": doc_ids,
                },
                tensor_type=return_tensors,
            )
def map_img_to_clip_feature(clip, feature_extractor, imgs, device="cuda"):
    for i, image in enumerate(imgs):
        if not image.mode == "RGB":
            imgs[i] = image.convert("RGB")
    retrieved_images = preprocess_images(imgs, feature_extractor).to(device)
    image_embeddings = clip.get_image_features(retrieved_images)
    image_embeddings = image_embeddings / torch.linalg.norm(image_embeddings, dim=-1, keepdim=True)
    image_embeddings = image_embeddings[None, ...]
    return image_embeddings
def map_txt_to_clip_feature(clip, tokenizer, prompt, device="cuda"):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    if text_input_ids.shape[-1] > tokenizer.model_max_length:
        removed_text = tokenizer.batch_decode(text_input_ids[:, tokenizer.model_max_length :])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )
        text_input_ids = text_input_ids[:, :tokenizer.model_max_length]
    text_embeddings = clip.get_text_features(text_input_ids.to(device))
    text_embeddings = text_embeddings / torch.linalg.norm(text_embeddings, dim=-1, keepdim=True)
    text_embeddings = text_embeddings[:, None, :]
    return text_embeddings[0][0].cpu().detach().numpy()
def get_dataset_with_emb(dataset, clip_model, feature_extractor, device="cuda", image_column="image", embedding_column="embeddings"):
    return dataset.map(lambda example: {embedding_column: map_img_to_clip_feature(clip_model, feature_extractor, [example[image_column]], device).cpu().detach().numpy()[0][0]})