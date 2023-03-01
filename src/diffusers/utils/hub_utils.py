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
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Union
from uuid import uuid4

from huggingface_hub import HfFolder, ModelCard, ModelCardData, whoami
from huggingface_hub.utils import is_jinja_available

from .. import __version__
from .constants import DIFFUSERS_CACHE, HUGGINGFACE_CO_RESOLVE_ENDPOINT
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
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "").upper() in ENV_VARS_TRUE_VALUES
DISABLE_TELEMETRY = os.getenv("DISABLE_TELEMETRY", "").upper() in ENV_VARS_TRUE_VALUES
HUGGINGFACE_CO_TELEMETRY = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/api/telemetry/"


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f"diffusers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if DISABLE_TELEMETRY or HF_HUB_OFFLINE:
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


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


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
        new_cache_dir = DIFFUSERS_CACHE
    if old_cache_dir is None:
        old_cache_dir = old_diffusers_cache

    old_cache_dir = Path(old_cache_dir).expanduser()
    new_cache_dir = Path(new_cache_dir).expanduser()
    for old_blob_path in old_cache_dir.glob("**/blobs/*"):  #  move file blob by blob
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


cache_version_file = os.path.join(DIFFUSERS_CACHE, "version_diffusers_cache.txt")
if not os.path.isfile(cache_version_file):
    cache_version = 0
else:
    with open(cache_version_file) as f:
        cache_version = int(f.read())

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
        os.makedirs(DIFFUSERS_CACHE, exist_ok=True)
        with open(cache_version_file, "w") as f:
            f.write("1")
    except Exception:
        logger.warning(
            f"There was a problem when trying to write in your cache folder ({DIFFUSERS_CACHE}). Please, ensure "
            "the directory exists and can be written to."
        )
