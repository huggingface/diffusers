# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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

import importlib
import inspect
import os
import re
import warnings
from collections import OrderedDict
from difflib import get_close_matches
from pathlib import Path

from diffusers.utils import ENV_VARS_TRUE_VALUES, is_flax_available, is_tf_available, is_torch_available


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_repo.py
PATH_TO_DIFFUSERS = "src/diffusers"
PATH_TO_TESTS = "tests"
PATH_TO_DOC = "docs/source/en"

# Update this list with models that are supposed to be private.
PRIVATE_MODELS = []

# Update this list for models that are not tested with a comment explaining the reason it should not be.
# Being in this list is an exception and should **not** be the rule.
IGNORE_NON_TESTED = PRIVATE_MODELS.copy() + []

# Update this list with test files that don't have a tester with a `all_model_classes` variable and which don't
# trigger the common tests.
TEST_FILES_WITH_NO_COMMON_TESTS = []

# Update this list for models that are not in any of the auto MODEL_XXX_MAPPING. Being in this list is an exception and
# should **not** be the rule.
IGNORE_NON_AUTO_CONFIGURED = PRIVATE_MODELS.copy() + []


# This is to make sure the transformers module imported is the one in the repo.
spec = importlib.util.spec_from_file_location(
    "diffusers",
    os.path.join(PATH_TO_DIFFUSERS, "__init__.py"),
    submodule_search_locations=[PATH_TO_DIFFUSERS],
)
diffusers = spec.loader.load_module()


def check_modules_are_in_local_init():
    """Check the model list inside the diffusers library."""
    # Get the modules from the directory structure of `src/diffusers/<models,schedulers,pipelines>/`
    modules_dirs = [os.path.join(PATH_TO_DIFFUSERS, subdir) for subdir in ["models", "pipelines", "schedulers"]]
    for modules_dir in modules_dirs:
        _modules = []
        for module in os.listdir(modules_dir):
            module_dir = os.path.join(modules_dir, module)
            if os.path.isdir(module_dir) and "__init__.py" in os.listdir(module_dir):
                _modules.append(module)
            elif os.path.isfile(module_dir) and not module.startswith("_") and module_dir.endswith(".py") :
                _modules.append(module.replace(".py", ""))

        # Get the modules from the directory structure of `src/diffusers/<models,schedulers,pipelines>/`
        module_dirs = dir(diffusers.models) + dir(diffusers.pipelines) + dir(diffusers.schedulers)
        modules = [module for module in module_dirs if not module.startswith("__")]

        missing_modules = sorted(list(set(_modules).difference(modules)))
        if missing_modules:
            raise Exception(
                f"The following modules should be included in {modules_dir}/__init__.py: {','.join(missing_modules)}."
            )


# If some modeling modules should be ignored for all checks, they should be added in the nested list
# _ignore_modules of this function.
def get_model_modules():
    """Get the model modules inside the diffusers library."""
    _ignore_modules = []
    modules = []
    for model in dir(diffusers.models):
        # There are some magic dunder attributes in the dir, we ignore them
        if not model.startswith("__"):
            model_module = getattr(diffusers.models, model)
            if inspect.ismodule(model_module):
                modules.append(model_module)
    return modules


def get_modules(module):
    """Get the objects in module that are models/schedulers/pipelines."""
    objects = []
    objects_classes = (diffusers.modeling_utils.ModelMixin, diffusers.SchedulerMixin, diffusers.DiffusionPipeline)
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, objects_classes) and attr.__module__ == module.__name__:
            objects.append((attr_name, attr))
    return objects


def check_modules_are_in_global_init():
    """Checks all models defined in the library are in the main init."""
    modules_not_in_init = []
    dir_diffusers = dir(diffusers)
    for module in get_model_modules():
        modules_not_in_init += [
            module[0] for module in get_modules(module) if module[0] not in dir_diffusers
        ]

    if len(modules_not_in_init) > 0:
        raise Exception(f"The following models should be in the main init: {','.join(modules_not_in_init)}.")


# If some test files should be ignored when checking models are all tested, they should be added in the
# nested list _ignore_files of this function.
def get_module_test_files():
    """Get the model/scheduler/pipeline test files.

    The returned files should NOT contain the `tests` (i.e. `PATH_TO_TESTS` defined in this script). They will be
    considered as paths relative to `tests`. A caller has to use `os.path.join(PATH_TO_TESTS, ...)` to access the files.
    """

    _ignore_files = []
    test_files = []

    model_test_root = os.path.join(PATH_TO_TESTS)
    model_test_dirs = []
    for x in os.listdir(model_test_root):
        x = os.path.join(model_test_root, x)
        if os.path.isdir(x):
            model_test_dirs.append(x)

    for target_dir in model_test_dirs:
        for file_or_dir in os.listdir(target_dir):
            path = os.path.join(target_dir, file_or_dir)
            if os.path.isfile(path):
                filename = os.path.split(path)[-1]
                if "test_" in filename and not os.path.splitext(filename)[0] in _ignore_files:
                    file = os.path.join(*path.split(os.sep)[1:])
                    test_files.append(file)

    return test_files


def check_models_are_tested(module, test_file):
    """Check models defined in module are tested in test_file."""
    # XxxModelMixin are not tested
    defined_models = get_modules(module)
    tested_models = find_tested_models(test_file)
    if tested_models is None:
        if test_file.replace(os.path.sep, "/") in TEST_FILES_WITH_NO_COMMON_TESTS:
            return
        return [
            f"{test_file} should define `all_model_classes` to apply common tests to the models it tests. "
            + "If this intentional, add the test filename to `TEST_FILES_WITH_NO_COMMON_TESTS` in the file "
            + "`utils/check_repo.py`."
        ]
    failures = []
    for model_name, _ in defined_models:
        if model_name not in tested_models and model_name not in IGNORE_NON_TESTED:
            failures.append(
                f"{model_name} is defined in {module.__name__} but is not tested in "
                + f"{os.path.join(PATH_TO_TESTS, test_file)}. Add it to the all_model_classes in that file."
                + "If common tests should not applied to that model, add its name to `IGNORE_NON_TESTED`"
                + "in the file `utils/check_repo.py`."
            )
    return failures


def check_all_modules_are_tested():
    """Check all models/schedulers/pipelines are properly tested."""
    modules = get_model_modules()
    test_files = get_module_test_files()
    failures = []
    for module in modules:
        test_file = [file for file in test_files if f"test_{module.__name__.split('.')[-1]}.py" in file]
        if len(test_file) == 0:
            failures.append(f"{module.__name__} does not have its corresponding test file {test_file}.")
        elif len(test_file) > 1:
            failures.append(f"{module.__name__} has several test files: {test_file}.")
        else:
            test_file = test_file[0]
            new_failures = check_models_are_tested(module, test_file)
            if new_failures is not None:
                failures += new_failures
    if len(failures) > 0:
        raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))

_re_decorator = re.compile(r"^\s*@(\S+)\s+$")


def check_decorator_order(filename):
    """Check that in the test file `filename` the slow decorator is always last."""
    with open(filename, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    decorator_before = None
    errors = []
    for i, line in enumerate(lines):
        search = _re_decorator.search(line)
        if search is not None:
            decorator_name = search.groups()[0]
            if decorator_before is not None and decorator_name.startswith("parameterized"):
                errors.append(i)
            decorator_before = decorator_name
        elif decorator_before is not None:
            decorator_before = None
    return errors


def check_all_decorator_order():
    """Check that in all test files, the slow decorator is always last."""
    errors = []
    for fname in os.listdir(PATH_TO_TESTS):
        if fname.endswith(".py"):
            filename = os.path.join(PATH_TO_TESTS, fname)
            new_errors = check_decorator_order(filename)
            errors += [f"- {filename}, line {i}" for i in new_errors]
    if len(errors) > 0:
        msg = "\n".join(errors)
        raise ValueError(
            "The parameterized decorator (and its variants) should always be first, but this is not the case in the"
            f" following files:\n{msg}"
        )


def find_all_documented_objects():
    """Parse the content of all doc files to detect which classes and functions it documents"""
    documented_obj = []
    for doc_file in Path(PATH_TO_DOC).glob("**/*.rst"):
        with open(doc_file, "r", encoding="utf-8", newline="\n") as f:
            content = f.read()
        raw_doc_objs = re.findall(r"(?:autoclass|autofunction):: transformers.(\S+)\s+", content)
        documented_obj += [obj.split(".")[-1] for obj in raw_doc_objs]
    for doc_file in Path(PATH_TO_DOC).glob("**/*.mdx"):
        with open(doc_file, "r", encoding="utf-8", newline="\n") as f:
            content = f.read()
        raw_doc_objs = re.findall("\[\[autodoc\]\]\s+(\S+)\s+", content)
        documented_obj += [obj.split(".")[-1] for obj in raw_doc_objs]
    return documented_obj


# One good reason for not being documented is to be deprecated. Put in this list deprecated objects.
DEPRECATED_OBJECTS = []

# Exceptionally, some objects should not be documented after all rules passed.
# ONLY PUT SOMETHING IN THIS LIST AS A LAST RESORT!
UNDOCUMENTED_OBJECTS = []

# This list should be empty. Objects in it should get their own doc page.
SHOULD_HAVE_THEIR_OWN_PAGE = []


def ignore_undocumented(name):
    """Rules to determine if `name` should be undocumented."""
    # NOT DOCUMENTED ON PURPOSE.
    # Constants uppercase are not documented.
    if name.isupper():
        return True
    # ModelMixins / Encoders / Decoders / Layers / Embeddings / Attention are not documented.
    if (
        name.endswith("ModelMixin")
        or name.endswith("Decoder")
        or name.endswith("Encoder")
        or name.endswith("Layer")
        or name.endswith("Embeddings")
        or name.endswith("Attention")
    ):
        return True
    # Submodules are not documented.
    if os.path.isdir(os.path.join(PATH_TO_DIFFUSERS, name)) or os.path.isfile(
        os.path.join(PATH_TO_DIFFUSERS, f"{name}.py")
    ):
        return True
    # All load functions are not documented.
    if name.startswith("load_tf") or name.startswith("load_pytorch"):
        return True
    # is_xxx_available functions are not documented.
    if name.startswith("is_") and name.endswith("_available"):
        return True
    # Deprecated objects are not documented.
    if name in DEPRECATED_OBJECTS or name in UNDOCUMENTED_OBJECTS:
        return True
    # MMBT model does not really work.
    if name.startswith("MMBT"):
        return True
    if name in SHOULD_HAVE_THEIR_OWN_PAGE:
        return True
    return False


def check_all_objects_are_documented():
    """Check all models are properly documented."""
    documented_objs = find_all_documented_objects()
    undocumented_objs = [c for c in dir(diffusers) if c not in documented_objs and not ignore_undocumented(c) and not c.startswith("_")]
    if len(undocumented_objs) > 0:
        raise Exception(
            "The following objects are in the public init so should be documented:\n - "
            + "\n - ".join(undocumented_objs)
        )
    check_docstrings_are_in_md()
    check_model_type_doc_match()


def check_model_type_doc_match():
    """Check all doc pages have a corresponding model type."""
    model_doc_folder = Path(PATH_TO_DOC) / "model_doc"
    model_docs = [m.stem for m in model_doc_folder.glob("*.mdx")]

    model_types = list(diffusers.models.auto.configuration_auto.MODEL_NAMES_MAPPING.keys())

    errors = []
    for m in model_docs:
        if m not in model_types and m != "auto":
            close_matches = get_close_matches(m, model_types)
            error_message = f"{m} is not a proper model identifier."
            if len(close_matches) > 0:
                close_matches = "/".join(close_matches)
                error_message += f" Did you mean {close_matches}?"
            errors.append(error_message)

    if len(errors) > 0:
        raise ValueError(
            "Some model doc pages do not match any existing model type:\n"
            + "\n".join(errors)
            + "\nYou can add any missing model type to the `MODEL_NAMES_MAPPING` constant in "
            "models/auto/configuration_auto.py."
        )


# Re pattern to catch :obj:`xx`, :class:`xx`, :func:`xx` or :meth:`xx`.
_re_rst_special_words = re.compile(r":(?:obj|func|class|meth):`([^`]+)`")
# Re pattern to catch things between double backquotes.
_re_double_backquotes = re.compile(r"(^|[^`])``([^`]+)``([^`]|$)")
# Re pattern to catch example introduction.
_re_rst_example = re.compile(r"^\s*Example.*::\s*$", flags=re.MULTILINE)


def is_rst_docstring(docstring):
    """
    Returns `True` if `docstring` is written in rst.
    """
    if _re_rst_special_words.search(docstring) is not None:
        return True
    if _re_double_backquotes.search(docstring) is not None:
        return True
    if _re_rst_example.search(docstring) is not None:
        return True
    return False


def check_docstrings_are_in_md():
    """Check all docstrings are in md"""
    files_with_rst = []
    for file in Path(PATH_TO_DIFFUSERS).glob("**/*.py"):
        with open(file, "r") as f:
            code = f.read()
        docstrings = code.split('"""')

        for idx, docstring in enumerate(docstrings):
            if idx % 2 == 0 or not is_rst_docstring(docstring):
                continue
            files_with_rst.append(file)
            break

    if len(files_with_rst) > 0:
        raise ValueError(
            "The following files have docstrings written in rst:\n"
            + "\n".join([f"- {f}" for f in files_with_rst])
            + "\nTo fix this run `doc-builder convert path_to_py_file` after installing `doc-builder`\n"
            "(`pip install git+https://github.com/huggingface/doc-builder`)"
        )


def check_repo_quality():
    """Check all models are properly tested and documented."""
    print("Checking all models, schedulers and pipelines are included.")
    check_modules_are_in_local_init()
    print("Checking all models, schedulers and pipelines are public.")
    check_modules_are_in_global_init()
    print("Checking all models, schedulers and pipelines are properly tested.")
    check_all_decorator_order()
    check_all_modules_are_tested()
    print("Checking all objects are properly documented.")
    check_all_objects_are_documented()


if __name__ == "__main__":
    check_repo_quality()
