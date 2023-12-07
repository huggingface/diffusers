# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Dict, Optional, Union
from uuid import uuid4

from huggingface_hub import (
    ModelCard,
    ModelCardData,
    create_repo,
    get_full_repo_name,
    hf_hub_download,
    upload_folder,
)
from huggingface_hub.constants import HF_HUB_CACHE, HF_HUB_DISABLE_TELEMETRY, HF_HUB_OFFLINE
from huggingface_hub.file_download import REGEX_COMMIT_HASH
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    is_jinja_available,
    validate_hf_hub_args,
)
from packaging import version
from requests import HTTPError

from .. import __version__
from .constants import (
    DEPRECATED_REVISION_ARGS,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
)
from .import_utils import (
    ENV_VARS_TRUE_VALUES,
    _flax_version,
    _jax_version,
    _onnxruntime_version,
    _torch_version,
    is_flax_available,
    is_onnx_available,
    is_torch_available,
)
from .logging import get_logger


logger = get_logger(__name__)


MODEL_CARD_TEMPLATE_PATH = Path(__file__).parent / "model_card_template.md"
SESSION_ID = uuid4().hex


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f"diffusers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if HF_HUB_DISABLE_TELEMETRY or HF_HUB_OFFLINE:
        return ua + "; telemetry/off"
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    if is_flax_available():
        ua += f"; jax/{_jax_version}"
        ua += f"; flax/{_flax_version}"
    if is_onnx_available():
        ua += f"; onnxruntime/{_onnxruntime_version}"
    # CI will set this value to True
    if os.environ.get("DIFFUSERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
        ua += "; is_ci/true"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def create_model_card(args, model_name):
    if not is_jinja_available():
        raise ValueError(
            "Modelcard rendering is based on Jinja templates."
            " Please make sure to have `jinja` installed before using `create_model_card`."
            " To install it, please run `pip install Jinja2`."
        )

    if hasattr(args, "local_rank") and args.local_rank not in [-1, 0]:
        return

    hub_token = args.hub_token if hasattr(args, "hub_token") else None
    repo_name = get_full_repo_name(model_name, token=hub_token)

    model_card = ModelCard.from_template(
        card_data=ModelCardData(  # Card metadata object that will be converted to YAML block
            language="en",
            license="apache-2.0",
            library_name="diffusers",
            tags=[],
            datasets=args.dataset_name,
            metrics=[],
        ),
        template_path=MODEL_CARD_TEMPLATE_PATH,
        model_name=model_name,
        repo_name=repo_name,
        dataset_name=args.dataset_name if hasattr(args, "dataset_name") else None,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps if hasattr(args, "gradient_accumulation_steps") else None
        ),
        adam_beta1=args.adam_beta1 if hasattr(args, "adam_beta1") else None,
        adam_beta2=args.adam_beta2 if hasattr(args, "adam_beta2") else None,
        adam_weight_decay=args.adam_weight_decay if hasattr(args, "adam_weight_decay") else None,
        adam_epsilon=args.adam_epsilon if hasattr(args, "adam_epsilon") else None,
        lr_scheduler=args.lr_scheduler if hasattr(args, "lr_scheduler") else None,
        lr_warmup_steps=args.lr_warmup_steps if hasattr(args, "lr_warmup_steps") else None,
        ema_inv_gamma=args.ema_inv_gamma if hasattr(args, "ema_inv_gamma") else None,
        ema_power=args.ema_power if hasattr(args, "ema_power") else None,
        ema_max_decay=args.ema_max_decay if hasattr(args, "ema_max_decay") else None,
        mixed_precision=args.mixed_precision,
    )

    card_path = os.path.join(args.output_dir, "README.md")
    model_card.save(card_path)


def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str] = None):
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


# Old default cache path, potentially to be migrated.
# This logic was more or less taken from `transformers`, with the following differences:
# - Diffusers doesn't use custom environment variables to specify the cache path.
# - There is no need to migrate the cache format, just move the files to the new location.
hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
old_diffusers_cache = os.path.join(hf_cache_home, "diffusers")


def move_cache(old_cache_dir: Optional[str] = None, new_cache_dir: Optional[str] = None) -> None:
    if new_cache_dir is None:
        new_cache_dir = HF_HUB_CACHE
    if old_cache_dir is None:
        old_cache_dir = old_diffusers_cache

    old_cache_dir = Path(old_cache_dir).expanduser()
    new_cache_dir = Path(new_cache_dir).expanduser()
    for old_blob_path in old_cache_dir.glob("**/blobs/*"):
        if old_blob_path.is_file() and not old_blob_path.is_symlink():
            new_blob_path = new_cache_dir / old_blob_path.relative_to(old_cache_dir)
            new_blob_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old_blob_path, new_blob_path)
            try:
                os.symlink(new_blob_path, old_blob_path)
            except OSError:
                logger.warning(
                    "Could not create symlink between old cache and new cache. If you use an older version of diffusers again, files will be re-downloaded."
                )
    # At this point, old_cache_dir contains symlinks to the new cache (it can still be used).


cache_version_file = os.path.join(HF_HUB_CACHE, "version_diffusers_cache.txt")
if not os.path.isfile(cache_version_file):
    cache_version = 0
else:
    with open(cache_version_file) as f:
        try:
            cache_version = int(f.read())
        except ValueError:
            cache_version = 0

if cache_version < 1:
    old_cache_is_not_empty = os.path.isdir(old_diffusers_cache) and len(os.listdir(old_diffusers_cache)) > 0
    if old_cache_is_not_empty:
        logger.warning(
            "The cache for model files in Diffusers v0.14.0 has moved to a new location. Moving your "
            "existing cached models. This is a one-time operation, you can interrupt it or run it "
            "later by calling `diffusers.utils.hub_utils.move_cache()`."
        )
        try:
            move_cache()
        except Exception as e:
            trace = "\n".join(traceback.format_tb(e.__traceback__))
            logger.error(
                f"There was a problem when trying to move your cache:\n\n{trace}\n{e.__class__.__name__}: {e}\n\nPlease "
                "file an issue at https://github.com/huggingface/diffusers/issues/new/choose, copy paste this whole "
                "message and we will do our best to help."
            )

if cache_version < 1:
    try:
        os.makedirs(HF_HUB_CACHE, exist_ok=True)
        with open(cache_version_file, "w") as f:
            f.write("1")
    except Exception:
        logger.warning(
            f"There was a problem when trying to write in your cache folder ({HF_HUB_CACHE}). Please, ensure "
            "the directory exists and can be written to."
        )


def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


@validate_hf_hub_args
def _get_model_file(
    pretrained_model_name_or_path: Union[str, Path],
    *,
    weights_name: str,
    subfolder: Optional[str],
    cache_dir: Optional[str],
    force_download: bool,
    proxies: Optional[Dict],
    resume_download: bool,
    local_files_only: bool,
    token: Optional[str],
    user_agent: Union[Dict, str, None],
    revision: Optional[str],
    commit_hash: Optional[str] = None,
):
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    elif os.path.isdir(pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
            # Load from a PyTorch checkpoint
            model_file = os.path.join(pretrained_model_name_or_path, weights_name)
            return model_file
        elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
        ):
            model_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
            return model_file
        else:
            raise EnvironmentError(
                f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}."
            )
    else:
        # 1. First check if deprecated way of loading from branches is used
        if (
            revision in DEPRECATED_REVISION_ARGS
            and (weights_name == WEIGHTS_NAME or weights_name == SAFETENSORS_WEIGHTS_NAME)
            and version.parse(version.parse(__version__).base_version) >= version.parse("0.22.0")
        ):
            try:
                model_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=_add_variant(weights_name, revision),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    subfolder=subfolder,
                    revision=revision or commit_hash,
                )
                warnings.warn(
                    f"Loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'` is deprecated. Loading instead from `revision='main'` with `variant={revision}`. Loading model variants via `revision='{revision}'` will be removed in diffusers v1. Please use `variant='{revision}'` instead.",
                    FutureWarning,
                )
                return model_file
            except:  # noqa: E722
                warnings.warn(
                    f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'`. This behavior is deprecated and will be removed in diffusers v1. One should use `variant='{revision}'` instead. However, it appears that {pretrained_model_name_or_path} currently does not have a {_add_variant(weights_name, revision)} file in the 'main' branch of {pretrained_model_name_or_path}. \n The Diffusers team and community would be very grateful if you could open an issue: https://github.com/huggingface/diffusers/issues/new with the title '{pretrained_model_name_or_path} is missing {_add_variant(weights_name, revision)}' so that the correct variant file can be added.",
                    FutureWarning,
                )
        try:
            # 2. Load model file as usual
            model_file = hf_hub_download(
                pretrained_model_name_or_path,
                filename=weights_name,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                subfolder=subfolder,
                revision=revision or commit_hash,
            )
            return model_file

        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                "token having permission to this repo with `token` or log in with `huggingface-cli "
                "login`."
            )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                "this model name. Check the model page at "
                f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
            )
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {weights_name}."
            )
        except HTTPError as err:
            raise EnvironmentError(
                f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n{err}"
            )
        except ValueError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                f" directory containing a file named {weights_name} or"
                " \nCheckout your internet connection or see how to run the library in"
                " offline mode at 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a file named {weights_name}"
            )


class PushToHubMixin:
    """
    A Mixin to push a model, scheduler, or pipeline to the Hugging Face Hub.
    """

    def _upload_folder(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all files in `working_dir` to `repo_id`.
        """
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Scheduler" in self.__class__.__name__:
                commit_message = "Upload scheduler"
            else:
                commit_message = f"Upload {self.__class__.__name__}"

        logger.info(f"Uploading the files of {working_dir} to {repo_id}.")
        return upload_folder(
            repo_id=repo_id, folder_path=working_dir, token=token, commit_message=commit_message, create_pr=create_pr
        )

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[str] = None,
        create_pr: bool = False,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
    ) -> str:
        """
        Upload model, scheduler, or pipeline files to the 🤗 Hugging Face Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your model, scheduler, or pipeline files to. It should
                contain your organization name when pushing to an organization. `repo_id` can also be a path to a local
                directory.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. The token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether or not to convert the model weights to the `safetensors` format.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.

        Examples:

        ```python
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="unet")

        # Push the `unet` to your namespace with the name "my-finetuned-unet".
        unet.push_to_hub("my-finetuned-unet")

        # Push the `unet` to an organization with the name "my-finetuned-unet".
        unet.push_to_hub("your-org/my-finetuned-unet")
        ```
        """
        repo_id = create_repo(repo_id, private=private, token=token, exist_ok=True).repo_id

        # Save all files.
        save_kwargs = {"safe_serialization": safe_serialization}
        if "Scheduler" not in self.__class__.__name__:
            save_kwargs.update({"variant": variant})

        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir, **save_kwargs)

            return self._upload_folder(
                tmpdir,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )
