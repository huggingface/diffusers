import os
import re
import unittest

import yaml


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
docker_path = os.path.join(git_repo_path, "docker")
workflow_path = os.path.join(git_repo_path, ".github", "workflows", "build_docker_images.yml")


class TestDockerImages(unittest.TestCase):
    def setUp(self):
        self.image_dirs = set(os.listdir(docker_path))
        with open(workflow_path, "r") as f:
            self.workflow = yaml.safe_load(f)

    def test_images_are_built_on_pull_requests(self):
        run = self.workflow["jobs"]["test-build-docker-images"]["steps"][-1]["run"]
        allowed_images = re.search(r"ALLOWED_IMAGES=\(\n(.*?)\)", run, flags=re.DOTALL).group(1)
        self.assertSetEqual(set(allowed_images.split()), self.image_dirs)

    def test_images_are_pushed_on_schedule(self):
        matrix = self.workflow["jobs"]["build-and-push-docker-images"]["strategy"]["matrix"]
        self.assertSetEqual(set(matrix["image-name"]), self.image_dirs)
