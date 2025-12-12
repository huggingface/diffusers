# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import tempfile
import uuid

import pytest
import torch
from huggingface_hub.utils import ModelCard, delete_repo, is_jinja_available

from ...others.test_utils import TOKEN, USER, is_staging_test


@is_staging_test
class ModelPushToHubTesterMixin:
    """
    Mixin class for testing push_to_hub functionality on models.

    Expected class attributes to be set by subclasses:
        - model_class: The model class to test

    Expected methods to be implemented by subclasses:
        - get_init_dict(): Returns dict of arguments to initialize the model
    """

    identifier = uuid.uuid4()
    repo_id = f"test-model-{identifier}"
    org_repo_id = f"valid_org/{repo_id}-org"

    def test_push_to_hub(self):
        """Test pushing model to hub and loading it back."""
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)
        model.push_to_hub(self.repo_id, token=TOKEN)

        new_model = self.model_class.from_pretrained(f"{USER}/{self.repo_id}")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2), "Parameters don't match after push_to_hub and from_pretrained"

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.repo_id)

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, repo_id=self.repo_id, push_to_hub=True, token=TOKEN)

        new_model = self.model_class.from_pretrained(f"{USER}/{self.repo_id}")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2), (
                "Parameters don't match after save_pretrained with push_to_hub and from_pretrained"
            )

        # Reset repo
        delete_repo(self.repo_id, token=TOKEN)

    def test_push_to_hub_in_organization(self):
        """Test pushing model to hub in organization namespace."""
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)
        model.push_to_hub(self.org_repo_id, token=TOKEN)

        new_model = self.model_class.from_pretrained(self.org_repo_id)
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2), "Parameters don't match after push_to_hub to org and from_pretrained"

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.org_repo_id)

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, push_to_hub=True, token=TOKEN, repo_id=self.org_repo_id)

        new_model = self.model_class.from_pretrained(self.org_repo_id)
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2), (
                "Parameters don't match after save_pretrained with push_to_hub to org and from_pretrained"
            )

        # Reset repo
        delete_repo(self.org_repo_id, token=TOKEN)

    def test_push_to_hub_library_name(self):
        """Test that library_name in model card is set to 'diffusers'."""
        if not is_jinja_available():
            pytest.skip("Model card tests cannot be performed without Jinja installed.")

        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)
        model.push_to_hub(self.repo_id, token=TOKEN)

        model_card = ModelCard.load(f"{USER}/{self.repo_id}", token=TOKEN).data
        assert model_card.library_name == "diffusers", (
            f"Expected library_name 'diffusers', got {model_card.library_name}"
        )

        # Reset repo
        delete_repo(self.repo_id, token=TOKEN)
