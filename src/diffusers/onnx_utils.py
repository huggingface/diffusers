# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np

from huggingface_hub import HfApi, HfFolder, hf_hub_download

from .utils import is_onnx_available, logging


if is_onnx_available():
    import onnxruntime as ort


ONNX_WEIGHTS_NAME = "model.onnx"


logger = logging.get_logger(__name__)


class OnnxModel:
    base_model_prefix = "onnx_model"

    def __init__(self, model=None, **kwargs):
        self.model = model
        self.model_save_dir = kwargs.get("model_save_dir", None)
        self.latest_model_name = kwargs.get("latest_model_name", "model.onnx")

    def __call__(self, **kwargs):
        inputs = {k: np.array(v) for k, v in kwargs.items()}
        return self.model.run(None, inputs)

    @staticmethod
    def load_model(path: Union[str, Path], provider=None):
        """
        Arguments:
        loads ONNX Inference session with Provider. Default Provider is if CUDAExecutionProvider GPU available else
        `CPUExecutionProvider`
            path (`str` or `Path`):
                Directory from which to load
            provider(`str`, *optional*):
                Onnxruntime provider to use for loading the model, defaults to `CUDAExecutionProvider` if GPU is
                available else `CPUExecutionProvider`
        """
        if provider is None:
            provider = "CPUExecutionProvider"

        return ort.InferenceSession(path, providers=[provider])

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        """
        Arguments:
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_ort.ORTModel.from_pretrained`] class method. It will always save the
        latest_model_name.
            save_directory (`str` or `Path`):
                Directory where to save the model file.
            file_name(`str`, *optional*):
                Overwrites the default model file name from `"model.onnx"` to `file_name`. This allows you to save the
                model with a different name.
        """
        model_file_name = file_name if file_name is not None else ONNX_WEIGHTS_NAME

        src_path = self.model_save_dir.joinpath(self.latest_model_name)
        dst_path = Path(save_directory).joinpath(model_file_name)
        if not src_path.samefile(dst_path):
            shutil.copyfile(src_path, dst_path)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Arguments:
        Save a model to a directory, so that it can be re-loaded using the [`~OnnxModel.from_pretrained`] class method.:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                <Tip warning={true}>

                Using `push_to_hub=True` will synchronize the repository you are pushing to with `save_directory`,
                which requires `save_directory` to be a local clone of the repo you are pushing to if it's an existing
                folder. Pass along `temp_dir=True` to use a temporary directory instead.

                </Tip>
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # saving model weights/files
        self._save_pretrained(save_directory, **kwargs)

        if push_to_hub:
            return self.push_to_hub(save_directory, **kwargs)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs,
    ):
        """
        Arguments:
        Load a model from a directory or the HF Hub. Implements:
        https:
            //github.com/huggingface/huggingface_hub/blob/e67de48368bc1843e40afc1cc9d236402b9609ee/src/huggingface_hub/hub_mixin.py#L73
            model_id (`str` or `Path`):
                Directory from which to load
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            cache_dir (`Union[str, Path]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            file_name(`str`):
                Overwrites the default model file name from `"model.onnx"` to `file_name`. This allows you to load
                different model files from the same repository or directory.
            provider(`str`):
                The ONNX runtime provider, e.g. `CPUExecutionProvider` or `CUDAExecutionProvider`.
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        model_file_name = file_name if file_name is not None else ONNX_WEIGHTS_NAME
        # load model from local directory
        if os.path.isdir(model_id):
            model = OnnxModel.load_model(os.path.join(model_id, model_file_name))
            kwargs["model_save_dir"] = Path(model_id)
        # load model from hub
        else:
            # download model
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=model_file_name,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
            )
            kwargs["model_save_dir"] = Path(model_cache_path).parent
            kwargs["latest_model_name"] = Path(model_cache_path).name
            model = OnnxModel.load_model(model_cache_path, provider=provider)
        return cls(model=model, **kwargs)

    def push_to_hub(
        self,
        save_directory: str = None,
        repository_id: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> str:
        if isinstance(use_auth_token, str):
            huggingface_token = use_auth_token
        elif use_auth_token:
            huggingface_token = HfFolder.get_token()
        else:
            raise ValueError("You need to proivde `use_auth_token` to be able to push to the hub")
        api = HfApi()

        user = api.whoami(huggingface_token)
        self._git_config_username_and_email(git_email=user["email"], git_user=user["fullname"])

        api.create_repo(
            token=huggingface_token,
            name=repository_id,
            organization=user["name"],
            exist_ok=True,
            private=private,
        )
        for path, subdirs, files in os.walk(save_directory):
            for name in files:
                local_file_path = os.path.join(path, name)
                _, hub_file_path = os.path.split(local_file_path)
                # FIXME: when huggingface_hub fixes the return of upload_file
                try:
                    api.upload_file(
                        token=huggingface_token,
                        repo_id=f"{user['name']}/{repository_id}",
                        path_or_fileobj=os.path.join(os.getcwd(), local_file_path),
                        path_in_repo=hub_file_path,
                    )
                except KeyError:
                    pass
                except NameError:
                    pass

    def _git_config_username_and_email(self, git_user: str = None, git_email: str = None):
        """
        Set git user name and email (only in the current repo)
        """
        try:
            if git_user is not None:
                subprocess.run(
                    ["git", "config", "--global", "user.name", git_user],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                )
            if git_email is not None:
                subprocess.run(
                    ["git", "config", "--global", "user.email", git_email],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                )
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        force_download: bool = True,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **model_kwargs,
    ):
        revision = None
        if len(str(model_id).split("@")) == 2:
            model_id, revision = model_id.split("@")

        return cls._from_pretrained(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            use_auth_token=use_auth_token,
            **model_kwargs,
        )
