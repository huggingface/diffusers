# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Union
from uuid import uuid4

from huggingface_hub import CommitOperationAdd, HfFolder, create_commit, create_repo, whoami

from .. import __version__
from .constants import HUGGINGFACE_CO_RESOLVE_ENDPOINT
from .import_utils import (
    ENV_VARS_TRUE_VALUES,
    _flax_version,
    _jax_version,
    _onnxruntime_version,
    _torch_version,
    is_flax_available,
    is_modelcards_available,
    is_onnx_available,
    is_torch_available,
)
from .logging import get_logger


if is_modelcards_available():
    from modelcards import CardData, ModelCard


logger = get_logger(__name__)


MODEL_CARD_TEMPLATE_PATH = Path(__file__).parent / "utils" / "model_card_template.md"
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
    if not is_modelcards_available:
        raise ValueError(
            "Please make sure to have `modelcards` installed when using the `create_model_card` function. You can"
            " install the package with `pip install modelcards`."
        )

    if hasattr(args, "local_rank") and args.local_rank not in [-1, 0]:
        return

    hub_token = args.hub_token if hasattr(args, "hub_token") else None
    repo_name = get_full_repo_name(model_name, token=hub_token)

    model_card = ModelCard.from_template(
        card_data=CardData(  # Card metadata object that will be converted to YAML block
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
        gradient_accumulation_steps=args.gradient_accumulation_steps
        if hasattr(args, "gradient_accumulation_steps")
        else None,
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


@contextmanager
def working_or_temp_dir(working_dir, use_temp_dir: bool = False):
    if use_temp_dir:
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    else:
        yield working_dir


class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or pipeline to the hub.
    """

    def _create_repo(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ):
        """
        Create the repo if needed, cleans up repo_id, retrieves the token.
        """
        token = HfFolder.get_token() if use_auth_token is True else use_auth_token
        url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)

        # If the namespace is not there, add it or `upload_file` will complain
        if "/" not in repo_id and url != f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/{repo_id}":
            repo_id = get_full_repo_name(repo_id, token=token)
        return repo_id, token

    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        """
        Returns the list of files with their last modification timestamp.
        """
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

    def _upload_modified_files(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        files_timestamps: Dict[str, float],
        commit_message: Optional[str] = None,
        token: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """
        if commit_message is None:
            commit_message = f"Upload {self.__class__.__name__}"
        modified_files = [
            f
            for f in os.listdir(working_dir)
            if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]
        ]
        operations = []
        for file in modified_files:
            operations.append(CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file))
        logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")
        return create_commit(
            repo_id=repo_id, operations=operations, commit_message=commit_message, token=token, create_pr=create_pr
        )

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
    ) -> str:
        """
        Parameters:
        Upload the {object_files} to the 🤗 Hub while synchronizing a local clone of the repo in `repo_path_or_name`.
            repo_id (`str`):
                The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            use_auth_token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
        Examples:
        ```python
        from diffusers import {object_class}
        {object} = {object_class}.from_pretrained("stabilityai/stable-diffusion-2")
        # Push the {object} to your namespace with the name "my-finetuned-diffusion".
        {object}.push_to_hub("my-finetuned-diffusion")
        # Push the {object} to an organization with the name "my-finetuned-diffusion".
        {object}.push_to_hub("huggingface/my-finetuned-diffusion")
        ```
        """

        if os.path.isdir(repo_id):
            working_dir = repo_id
            repo_id = repo_id.split(os.path.sep)[-1]
        else:
            working_dir = repo_id.split("/")[-1]

        repo_id, token = self._create_repo(repo_id, private=private, use_auth_token=use_auth_token)

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            self.save_pretrained(work_dir)

            return self._upload_modified_files(
                work_dir, repo_id, files_timestamps, commit_message=commit_message, token=token, create_pr=create_pr
            )
