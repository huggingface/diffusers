# Copyright 2022 The HuggingFace Team. All rights reserved.
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

To create the package for pypi.

1. Run `make pre-release` (or `make pre-patch` for a patch release) then run `make fix-copies` to fix the index of the
   documentation.

   If releasing on a special branch, copy the updated README.md on the main branch for your the commit you will make
   for the post-release and run `make fix-copies` on the main branch as well.

2. Run Tests for Amazon Sagemaker. The documentation is located in `./tests/sagemaker/README.md`, otherwise @philschmid.

3. Unpin specific versions from setup.py that use a git install.

4. Checkout the release branch (v<RELEASE>-release, for example v4.19-release), and commit these changes with the
   message: "Release: <RELEASE>" and push.

5. Wait for the tests on main to be completed and be green (otherwise revert and fix bugs)

6. Add a tag in git to mark the release: "git tag v<RELEASE> -m 'Adds tag v<RELEASE> for pypi' "
   Push the tag to git: git push --tags origin v<RELEASE>-release

7. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

8. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi diffusers

   Check you can run the following commands:
   python -c "from diffusers import pipeline; classifier = pipeline('text-classification'); print(classifier('What a nice release'))"
   python -c "from diffusers import *"

9. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

10. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

11. Run `make post-release` (or, for a patch release, `make post-patch`). If you were on a branch for the release,
    you need to go back to main before executing this.
"""

import re
from distutils.core import Command

from setuptools import find_packages, setup

# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
# 2. once modified, run: `make deps_table_update` to update src/diffusers/dependency_versions_table.py
_deps = [
    "Pillow",
    "black~=22.0,>=22.3",
    "filelock",
    "flake8>=3.8.3",
    "huggingface-hub",
    "importlib_metadata",
    "isort>=5.5.4",
    "numpy",
    "pytest",
    "regex!=2019.12.17",
    "requests",
    "torch>=1.4",
    "tensorboard",
    "modelcards==0.1.4"
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
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets
#
# Just pass the desired package names to that script as it's shown with 2 packages above.
#
# If diffusers is not yet installed and the work is done from the cloned repo remember to add `PYTHONPATH=src` to the script above
#
# You can then feed this for example to `pip`:
#
# pip install -U $(python -c 'import sys; from diffusers.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets)
#


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


class DepsTableUpdateCommand(Command):
    """
    A custom distutils command that updates the dependency table.
    usage: python setup.py deps_table_update
    """

    description = "build runtime dependency table"
    user_options = [
        # format: (long option, short option, description).
        ("dep-table-update", None, "updates src/diffusers/dependency_versions_table.py"),
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
            "# 2. run `make deps_table_update``",
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


extras = {}
extras["quality"] = ["black ~= 22.0", "isort >= 5.5.4", "flake8 >= 3.8.3"]
extras["docs"] = []
extras["training"] = ["tensorboard", "modelcards"]
extras["test"] = [
    "pytest",
]
extras["dev"] = extras["quality"] + extras["test"] + extras["training"]

install_requires = [
    deps["importlib_metadata"],
    deps["filelock"],
    deps["huggingface-hub"],
    deps["numpy"],
    deps["regex"],
    deps["requests"],
    deps["torch"],
    deps["Pillow"],
]

setup(
    name="diffusers",
    version="0.1.2",
    description="Diffusers",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="The HuggingFace team",
    author_email="patrick@huggingface.co",
    url="https://github.com/huggingface/diffusers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.6.0",
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    cmdclass={"deps_table_update": DepsTableUpdateCommand},
)

# Release checklist
# 1. Change the version in __init__.py and setup.py.
# 2. Commit these changes with the message: "Release: Release"
# 3. Add a tag in git to mark the release: "git tag RELEASE -m 'Adds tag RELEASE for pypi' "
#    Push the tag to git: git push --tags origin main
# 4. Run the following commands in the top-level directory:
#      python setup.py bdist_wheel
#      python setup.py sdist
# 5. Upload the package to the pypi test server first:
#      twine upload dist/* -r pypitest
#      twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
# 6. Check that you can install it in a virtualenv by running:
#      pip install -i https://testpypi.python.org/pypi diffusers
#      diffusers env
#      diffusers test
# 7. Upload the final version to actual pypi:
#      twine upload dist/* -r pypi
# 8. Add release notes to the tag in github once everything is looking hunky-dory.
# 9. Update the version in __init__.py, setup.py to the new version "-dev" and push to master
