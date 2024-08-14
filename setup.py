# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/main/setup.py

To create the package for PyPI.

1. Run `make pre-release` (or `make pre-patch` for a patch release) then run `make fix-copies` to fix the index of the
   documentation.

   If releasing on a special branch, copy the updated README.md on the main branch for the commit you will make
   for the post-release and run `make fix-copies` on the main branch as well.

2. Unpin specific versions from setup.py that use a git install.

3. Checkout the release branch (v<RELEASE>-release, for example v4.19-release), and commit these changes with the
   message: "Release: <RELEASE>" and push.

4. Manually trigger the "Nightly and release tests on main/release branch" workflow from the release branch. Wait for
   the tests to complete. We can safely ignore the known test failures.

5. Wait for the tests on main to be completed and be green (otherwise revert and fix bugs).

6. Add a tag in git to mark the release: "git tag v<RELEASE> -m 'Adds tag v<RELEASE> for PyPI'"
   Push the tag to git: git push --tags origin v<RELEASE>-release

7. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory
   (This will build a wheel for the Python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

   Long story cut short, you need to run both before you can upload the distribution to the
   test PyPI and the actual PyPI servers:

   python setup.py bdist_wheel && python setup.py sdist

8. Check that everything looks correct by uploading the package to the PyPI test server:

   twine upload dist/* -r pypitest
   (pypi suggests using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi diffusers

   If you are testing from a Colab Notebook, for instance, then do:
   pip install diffusers && pip uninstall diffusers
   pip install -i https://testpypi.python.org/pypi diffusers

   Check you can run the following commands:
   python -c "from diffusers import __version__; print(__version__)"
   python -c "from diffusers import DiffusionPipeline; pipe = DiffusionPipeline.from_pretrained('fusing/unet-ldm-dummy-update'); pipe()"
   python -c "from diffusers import DiffusionPipeline; pipe = DiffusionPipeline.from_pretrained('hf-internal-testing/tiny-stable-diffusion-pipe', safety_checker=None); pipe('ah suh du')"
   python -c "from diffusers import *"

9. Upload the final version to the actual PyPI:
   twine upload dist/* -r pypi

10. Prepare the release notes and publish them on GitHub once everything is looking hunky-dory. You can use the following
    Space to fetch all the commits applicable for the release: https://huggingface.co/spaces/lysandre/github-release. Repo should
    be `huggingface/diffusers`. `tag` should be the previous release tag (v0.26.1, for example), and `branch` should be
    the latest release branch (v0.27.0-release, for example). It denotes all commits that have happened on branch
    v0.27.0-release after the tag v0.26.1 was created.

11. Run `make post-release` (or, for a patch release, `make post-patch`). If you were on a branch for the release,
    you need to go back to main before executing this.
"""

import os
import re
import sys

from setuptools import Command, find_packages, setup


# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
# 2. once modified, run: `make deps_table_update` to update src/diffusers/dependency_versions_table.py
_deps = [
    "Pillow",  # keep the PIL.Image.Resampling deprecation away
    "accelerate>=0.31.0",
    "compel==0.1.8",
    "datasets",
    "filelock",
    "flax>=0.4.1",
    "hf-doc-builder>=0.3.0",
    "huggingface-hub>=0.23.2",
    "requests-mock==1.10.0",
    "importlib_metadata",
    "invisible-watermark>=0.2.0",
    "isort>=5.5.4",
    "jax>=0.4.1",
    "jaxlib>=0.4.1",
    "Jinja2",
    "k-diffusion>=0.0.12",
    "torchsde",
    "note_seq",
    "librosa",
    "numpy",
    "parameterized",
    "peft>=0.6.0",
    "protobuf>=3.20.3,<4",
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "python>=3.8.0",
    "ruff==0.1.5",
    "safetensors>=0.3.1",
    "sentencepiece>=0.1.91,!=0.1.92",
    "GitPython<3.1.19",
    "scipy",
    "onnx",
    "regex!=2019.12.17",
    "requests",
    "tensorboard",
    "torch>=1.4",
    "torchvision",
    "transformers>=4.41.2",
    "urllib3<=2.0.0",
    "black",
]

# this is a lookup table with items like:
#
# tokenizers: "huggingface-hub==0.8.0"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

# since we save this data in src/diffusers/dependency_versions_table.py it can be easily accessed from
# anywhere. If you need to quickly access the data from this table in a shell, you can do so easily with:
#
# python -c 'import sys; from diffusers.dependency_versions_table import deps; \
# print(" ".join([deps[x] for x in sys.argv[1:]]))' tokenizers datasets
#
# Just pass the desired package names to that script as it's shown with 2 packages above.
#
# If diffusers is not yet installed and the work is done from the cloned repo remember to add `PYTHONPATH=src` to the script above
#
# You can then feed this for example to `pip`:
#
# pip install -U $(python -c 'import sys; from diffusers.dependency_versions_table import deps; \
# print(" ".join([deps[x] for x in sys.argv[1:]]))' tokenizers datasets)
#


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


class DepsTableUpdateCommand(Command):
    """
    A custom command that updates the dependency table.
    usage: python setup.py deps_table_update
    """

    description = "build runtime dependency table"
    user_options = [
        # format: (long option, short option, description).
        (
            "dep-table-update",
            None,
            "updates src/diffusers/dependency_versions_table.py",
        ),
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        entries = "\n".join([f'    "{k}": "{v}",' for k, v in deps.items()])
        content = [
            "# THIS FILE HAS BEEN AUTOGENERATED. To update:",
            "# 1. modify the `_deps` dict in setup.py",
            "# 2. run `make deps_table_update`",
            "deps = {",
            entries,
            "}",
            "",
        ]
        target = "src/diffusers/dependency_versions_table.py"
        print(f"updating {target}")
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(content))


extras = {}
extras["quality"] = deps_list("urllib3", "isort", "ruff", "hf-doc-builder")
extras["docs"] = deps_list("hf-doc-builder")
extras["training"] = deps_list("accelerate", "datasets", "protobuf", "tensorboard", "Jinja2", "peft")
extras["test"] = deps_list(
    "compel",
    "GitPython",
    "datasets",
    "Jinja2",
    "invisible-watermark",
    "k-diffusion",
    "librosa",
    "parameterized",
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "requests-mock",
    "safetensors",
    "sentencepiece",
    "scipy",
    "torchvision",
    "transformers",
)
extras["torch"] = deps_list("torch", "accelerate")

if os.name == "nt":  # windows
    extras["flax"] = []  # jax is not supported on windows
else:
    extras["flax"] = deps_list("jax", "jaxlib", "flax")

extras["dev"] = (
    extras["quality"] + extras["test"] + extras["training"] + extras["docs"] + extras["torch"] + extras["flax"]
)

install_requires = [
    deps["importlib_metadata"],
    deps["filelock"],
    deps["huggingface-hub"],
    deps["numpy"],
    deps["regex"],
    deps["requests"],
    deps["safetensors"],
    deps["Pillow"],
]

version_range_max = max(sys.version_info[1], 10) + 1

setup(
    name="diffusers",
    version="0.31.0.dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="State-of-the-art diffusion in PyTorch and JAX.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning diffusion jax pytorch stable diffusion audioldm",
    license="Apache 2.0 License",
    author="The Hugging Face team (past and future) with the help of all our contributors (https://github.com/huggingface/diffusers/graphs/contributors)",
    author_email="diffusers@huggingface.co",
    url="https://github.com/huggingface/diffusers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"diffusers": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=list(install_requires),
    extras_require=extras,
    entry_points={"console_scripts": ["diffusers-cli=diffusers.commands.diffusers_cli:main"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, version_range_max)],
    cmdclass={"deps_table_update": DepsTableUpdateCommand},
)
