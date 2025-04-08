# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

from diffusers.models.auto import get_values
from diffusers.utils import ENV_VARS_TRUE_VALUES, is_flax_available, is_torch_available


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_repo.py
PATH_TO_DIFFUSERS = "src/diffusers"
PATH_TO_TESTS = "tests"
PATH_TO_DOC = "docs/source/en"

# Update this list with models that are supposed to be private.
PRIVATE_MODELS = [
    "DPRSpanPredictor",
    "RealmBertModel",
    "T5Stack",
    "TFDPRSpanPredictor",
]

# Update this list for models that are not tested with a comment explaining the reason it should not be.
# Being in this list is an exception and should **not** be the rule.
IGNORE_NON_TESTED = PRIVATE_MODELS.copy() + [
    # models to ignore for not tested
    "OPTDecoder",  # Building part of bigger (tested) model.
    "DecisionTransformerGPT2Model",  # Building part of bigger (tested) model.
    "SegformerDecodeHead",  # Building part of bigger (tested) model.
    "PLBartEncoder",  # Building part of bigger (tested) model.
    "PLBartDecoder",  # Building part of bigger (tested) model.
    "PLBartDecoderWrapper",  # Building part of bigger (tested) model.
    "BigBirdPegasusEncoder",  # Building part of bigger (tested) model.
    "BigBirdPegasusDecoder",  # Building part of bigger (tested) model.
    "BigBirdPegasusDecoderWrapper",  # Building part of bigger (tested) model.
    "DetrEncoder",  # Building part of bigger (tested) model.
    "DetrDecoder",  # Building part of bigger (tested) model.
    "DetrDecoderWrapper",  # Building part of bigger (tested) model.
    "M2M100Encoder",  # Building part of bigger (tested) model.
    "M2M100Decoder",  # Building part of bigger (tested) model.
    "Speech2TextEncoder",  # Building part of bigger (tested) model.
    "Speech2TextDecoder",  # Building part of bigger (tested) model.
    "LEDEncoder",  # Building part of bigger (tested) model.
    "LEDDecoder",  # Building part of bigger (tested) model.
    "BartDecoderWrapper",  # Building part of bigger (tested) model.
    "BartEncoder",  # Building part of bigger (tested) model.
    "BertLMHeadModel",  # Needs to be setup as decoder.
    "BlenderbotSmallEncoder",  # Building part of bigger (tested) model.
    "BlenderbotSmallDecoderWrapper",  # Building part of bigger (tested) model.
    "BlenderbotEncoder",  # Building part of bigger (tested) model.
    "BlenderbotDecoderWrapper",  # Building part of bigger (tested) model.
    "MBartEncoder",  # Building part of bigger (tested) model.
    "MBartDecoderWrapper",  # Building part of bigger (tested) model.
    "MegatronBertLMHeadModel",  # Building part of bigger (tested) model.
    "MegatronBertEncoder",  # Building part of bigger (tested) model.
    "MegatronBertDecoder",  # Building part of bigger (tested) model.
    "MegatronBertDecoderWrapper",  # Building part of bigger (tested) model.
    "PegasusEncoder",  # Building part of bigger (tested) model.
    "PegasusDecoderWrapper",  # Building part of bigger (tested) model.
    "DPREncoder",  # Building part of bigger (tested) model.
    "ProphetNetDecoderWrapper",  # Building part of bigger (tested) model.
    "RealmBertModel",  # Building part of bigger (tested) model.
    "RealmReader",  # Not regular model.
    "RealmScorer",  # Not regular model.
    "RealmForOpenQA",  # Not regular model.
    "ReformerForMaskedLM",  # Needs to be setup as decoder.
    "Speech2Text2DecoderWrapper",  # Building part of bigger (tested) model.
    "TFDPREncoder",  # Building part of bigger (tested) model.
    "TFElectraMainLayer",  # Building part of bigger (tested) model (should it be a TFModelMixin ?)
    "TFRobertaForMultipleChoice",  # TODO: fix
    "TrOCRDecoderWrapper",  # Building part of bigger (tested) model.
    "SeparableConv1D",  # Building part of bigger (tested) model.
    "FlaxBartForCausalLM",  # Building part of bigger (tested) model.
    "FlaxBertForCausalLM",  # Building part of bigger (tested) model. Tested implicitly through FlaxRobertaForCausalLM.
    "OPTDecoderWrapper",
]

# Update this list with test files that don't have a tester with a `all_model_classes` variable and which don't
# trigger the common tests.
TEST_FILES_WITH_NO_COMMON_TESTS = [
    "models/decision_transformer/test_modeling_decision_transformer.py",
    "models/camembert/test_modeling_camembert.py",
    "models/mt5/test_modeling_flax_mt5.py",
    "models/mbart/test_modeling_mbart.py",
    "models/mt5/test_modeling_mt5.py",
    "models/pegasus/test_modeling_pegasus.py",
    "models/camembert/test_modeling_tf_camembert.py",
    "models/mt5/test_modeling_tf_mt5.py",
    "models/xlm_roberta/test_modeling_tf_xlm_roberta.py",
    "models/xlm_roberta/test_modeling_flax_xlm_roberta.py",
    "models/xlm_prophetnet/test_modeling_xlm_prophetnet.py",
    "models/xlm_roberta/test_modeling_xlm_roberta.py",
    "models/vision_text_dual_encoder/test_modeling_vision_text_dual_encoder.py",
    "models/vision_text_dual_encoder/test_modeling_flax_vision_text_dual_encoder.py",
    "models/decision_transformer/test_modeling_decision_transformer.py",
]

# Update this list for models that are not in any of the auto MODEL_XXX_MAPPING. Being in this list is an exception and
# should **not** be the rule.
IGNORE_NON_AUTO_CONFIGURED = PRIVATE_MODELS.copy() + [
    # models to ignore for model xxx mapping
    "DPTForDepthEstimation",
    "DecisionTransformerGPT2Model",
    "GLPNForDepthEstimation",
    "ViltForQuestionAnswering",
    "ViltForImagesAndTextClassification",
    "ViltForImageAndTextRetrieval",
    "ViltForMaskedLM",
    "XGLMEncoder",
    "XGLMDecoder",
    "XGLMDecoderWrapper",
    "PerceiverForMultimodalAutoencoding",
    "PerceiverForOpticalFlow",
    "SegformerDecodeHead",
    "FlaxBeitForMaskedImageModeling",
    "PLBartEncoder",
    "PLBartDecoder",
    "PLBartDecoderWrapper",
    "BeitForMaskedImageModeling",
    "CLIPTextModel",
    "CLIPVisionModel",
    "TFCLIPTextModel",
    "TFCLIPVisionModel",
    "FlaxCLIPTextModel",
    "FlaxCLIPVisionModel",
    "FlaxWav2Vec2ForCTC",
    "DetrForSegmentation",
    "DPRReader",
    "FlaubertForQuestionAnswering",
    "FlavaImageCodebook",
    "FlavaTextModel",
    "FlavaImageModel",
    "FlavaMultimodalModel",
    "GPT2DoubleHeadsModel",
    "LukeForMaskedLM",
    "LukeForEntityClassification",
    "LukeForEntityPairClassification",
    "LukeForEntitySpanClassification",
    "OpenAIGPTDoubleHeadsModel",
    "RagModel",
    "RagSequenceForGeneration",
    "RagTokenForGeneration",
    "RealmEmbedder",
    "RealmForOpenQA",
    "RealmScorer",
    "RealmReader",
    "TFDPRReader",
    "TFGPT2DoubleHeadsModel",
    "TFOpenAIGPTDoubleHeadsModel",
    "TFRagModel",
    "TFRagSequenceForGeneration",
    "TFRagTokenForGeneration",
    "Wav2Vec2ForCTC",
    "HubertForCTC",
    "SEWForCTC",
    "SEWDForCTC",
    "XLMForQuestionAnswering",
    "XLNetForQuestionAnswering",
    "SeparableConv1D",
    "VisualBertForRegionToPhraseAlignment",
    "VisualBertForVisualReasoning",
    "VisualBertForQuestionAnswering",
    "VisualBertForMultipleChoice",
    "TFWav2Vec2ForCTC",
    "TFHubertForCTC",
    "MaskFormerForInstanceSegmentation",
]

# Update this list for models that have multiple model types for the same
# model doc
MODEL_TYPE_TO_DOC_MAPPING = OrderedDict(
    [
        ("data2vec-text", "data2vec"),
        ("data2vec-audio", "data2vec"),
        ("data2vec-vision", "data2vec"),
    ]
)


# This is to make sure the transformers module imported is the one in the repo.
spec = importlib.util.spec_from_file_location(
    "diffusers",
    os.path.join(PATH_TO_DIFFUSERS, "__init__.py"),
    submodule_search_locations=[PATH_TO_DIFFUSERS],
)
diffusers = spec.loader.load_module()


def check_model_list():
    """Check the model list inside the transformers library."""
    # Get the models from the directory structure of `src/diffusers/models/`
    models_dir = os.path.join(PATH_TO_DIFFUSERS, "models")
    _models = []
    for model in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model)
        if os.path.isdir(model_dir) and "__init__.py" in os.listdir(model_dir):
            _models.append(model)

    # Get the models from the directory structure of `src/transformers/models/`
    models = [model for model in dir(diffusers.models) if not model.startswith("__")]

    missing_models = sorted(set(_models).difference(models))
    if missing_models:
        raise Exception(
            f"The following models should be included in {models_dir}/__init__.py: {','.join(missing_models)}."
        )


# If some modeling modules should be ignored for all checks, they should be added in the nested list
# _ignore_modules of this function.
def get_model_modules():
    """Get the model modules inside the transformers library."""
    _ignore_modules = [
        "modeling_auto",
        "modeling_encoder_decoder",
        "modeling_marian",
        "modeling_mmbt",
        "modeling_outputs",
        "modeling_retribert",
        "modeling_utils",
        "modeling_flax_auto",
        "modeling_flax_encoder_decoder",
        "modeling_flax_utils",
        "modeling_speech_encoder_decoder",
        "modeling_flax_speech_encoder_decoder",
        "modeling_flax_vision_encoder_decoder",
        "modeling_transfo_xl_utilities",
        "modeling_tf_auto",
        "modeling_tf_encoder_decoder",
        "modeling_tf_outputs",
        "modeling_tf_pytorch_utils",
        "modeling_tf_utils",
        "modeling_tf_transfo_xl_utilities",
        "modeling_tf_vision_encoder_decoder",
        "modeling_vision_encoder_decoder",
    ]
    modules = []
    for model in dir(diffusers.models):
        # There are some magic dunder attributes in the dir, we ignore them
        if not model.startswith("__"):
            model_module = getattr(diffusers.models, model)
            for submodule in dir(model_module):
                if submodule.startswith("modeling") and submodule not in _ignore_modules:
                    modeling_module = getattr(model_module, submodule)
                    if inspect.ismodule(modeling_module):
                        modules.append(modeling_module)
    return modules


def get_models(module, include_pretrained=False):
    """Get the objects in module that are models."""
    models = []
    model_classes = (diffusers.ModelMixin, diffusers.TFModelMixin, diffusers.FlaxModelMixin)
    for attr_name in dir(module):
        if not include_pretrained and ("Pretrained" in attr_name or "PreTrained" in attr_name):
            continue
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, model_classes) and attr.__module__ == module.__name__:
            models.append((attr_name, attr))
    return models


def is_a_private_model(model):
    """Returns True if the model should not be in the main init."""
    if model in PRIVATE_MODELS:
        return True

    # Wrapper, Encoder and Decoder are all privates
    if model.endswith("Wrapper"):
        return True
    if model.endswith("Encoder"):
        return True
    if model.endswith("Decoder"):
        return True
    return False


def check_models_are_in_init():
    """Checks all models defined in the library are in the main init."""
    models_not_in_init = []
    dir_transformers = dir(diffusers)
    for module in get_model_modules():
        models_not_in_init += [
            model[0] for model in get_models(module, include_pretrained=True) if model[0] not in dir_transformers
        ]

    # Remove private models
    models_not_in_init = [model for model in models_not_in_init if not is_a_private_model(model)]
    if len(models_not_in_init) > 0:
        raise Exception(f"The following models should be in the main init: {','.join(models_not_in_init)}.")


# If some test_modeling files should be ignored when checking models are all tested, they should be added in the
# nested list _ignore_files of this function.
def get_model_test_files():
    """Get the model test files.

    The returned files should NOT contain the `tests` (i.e. `PATH_TO_TESTS` defined in this script). They will be
    considered as paths relative to `tests`. A caller has to use `os.path.join(PATH_TO_TESTS, ...)` to access the files.
    """

    _ignore_files = [
        "test_modeling_common",
        "test_modeling_encoder_decoder",
        "test_modeling_flax_encoder_decoder",
        "test_modeling_flax_speech_encoder_decoder",
        "test_modeling_marian",
        "test_modeling_tf_common",
        "test_modeling_tf_encoder_decoder",
    ]
    test_files = []
    # Check both `PATH_TO_TESTS` and `PATH_TO_TESTS/models`
    model_test_root = os.path.join(PATH_TO_TESTS, "models")
    model_test_dirs = []
    for x in os.listdir(model_test_root):
        x = os.path.join(model_test_root, x)
        if os.path.isdir(x):
            model_test_dirs.append(x)

    for target_dir in [PATH_TO_TESTS] + model_test_dirs:
        for file_or_dir in os.listdir(target_dir):
            path = os.path.join(target_dir, file_or_dir)
            if os.path.isfile(path):
                filename = os.path.split(path)[-1]
                if "test_modeling" in filename and os.path.splitext(filename)[0] not in _ignore_files:
                    file = os.path.join(*path.split(os.sep)[1:])
                    test_files.append(file)

    return test_files


# This is a bit hacky but I didn't find a way to import the test_file as a module and read inside the tester class
# for the all_model_classes variable.
def find_tested_models(test_file):
    """Parse the content of test_file to detect what's in all_model_classes"""
    # This is a bit hacky but I didn't find a way to import the test_file as a module and read inside the class
    with open(os.path.join(PATH_TO_TESTS, test_file), "r", encoding="utf-8", newline="\n") as f:
        content = f.read()
    all_models = re.findall(r"all_model_classes\s+=\s+\(\s*\(([^\)]*)\)", content)
    # Check with one less parenthesis as well
    all_models += re.findall(r"all_model_classes\s+=\s+\(([^\)]*)\)", content)
    if len(all_models) > 0:
        model_tested = []
        for entry in all_models:
            for line in entry.split(","):
                name = line.strip()
                if len(name) > 0:
                    model_tested.append(name)
        return model_tested


def check_models_are_tested(module, test_file):
    """Check models defined in module are tested in test_file."""
    # XxxModelMixin are not tested
    defined_models = get_models(module)
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


def check_all_models_are_tested():
    """Check all models are properly tested."""
    modules = get_model_modules()
    test_files = get_model_test_files()
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


def get_all_auto_configured_models():
    """Return the list of all models in at least one auto class."""
    result = set()  # To avoid duplicates we concatenate all model classes in a set.
    if is_torch_available():
        for attr_name in dir(diffusers.models.auto.modeling_auto):
            if attr_name.startswith("MODEL_") and attr_name.endswith("MAPPING_NAMES"):
                result = result | set(get_values(getattr(diffusers.models.auto.modeling_auto, attr_name)))
    if is_flax_available():
        for attr_name in dir(diffusers.models.auto.modeling_flax_auto):
            if attr_name.startswith("FLAX_MODEL_") and attr_name.endswith("MAPPING_NAMES"):
                result = result | set(get_values(getattr(diffusers.models.auto.modeling_flax_auto, attr_name)))
    return list(result)


def ignore_unautoclassed(model_name):
    """Rules to determine if `name` should be in an auto class."""
    # Special white list
    if model_name in IGNORE_NON_AUTO_CONFIGURED:
        return True
    # Encoder and Decoder should be ignored
    if "Encoder" in model_name or "Decoder" in model_name:
        return True
    return False


def check_models_are_auto_configured(module, all_auto_models):
    """Check models defined in module are each in an auto class."""
    defined_models = get_models(module)
    failures = []
    for model_name, _ in defined_models:
        if model_name not in all_auto_models and not ignore_unautoclassed(model_name):
            failures.append(
                f"{model_name} is defined in {module.__name__} but is not present in any of the auto mapping. "
                "If that is intended behavior, add its name to `IGNORE_NON_AUTO_CONFIGURED` in the file "
                "`utils/check_repo.py`."
            )
    return failures


def check_all_models_are_auto_configured():
    """Check all models are each in an auto class."""
    missing_backends = []
    if not is_torch_available():
        missing_backends.append("PyTorch")
    if not is_flax_available():
        missing_backends.append("Flax")
    if len(missing_backends) > 0:
        missing = ", ".join(missing_backends)
        if os.getenv("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
            raise Exception(
                "Full quality checks require all backends to be installed (with `pip install -e .[dev]` in the "
                f"Transformers repo, the following are missing: {missing}."
            )
        else:
            warnings.warn(
                "Full quality checks require all backends to be installed (with `pip install -e .[dev]` in the "
                f"Transformers repo, the following are missing: {missing}. While it's probably fine as long as you "
                "didn't make any change in one of those backends modeling files, you should probably execute the "
                "command above to be on the safe side."
            )
    modules = get_model_modules()
    all_auto_models = get_all_auto_configured_models()
    failures = []
    for module in modules:
        new_failures = check_models_are_auto_configured(module, all_auto_models)
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
    for doc_file in Path(PATH_TO_DOC).glob("**/*.md"):
        with open(doc_file, "r", encoding="utf-8", newline="\n") as f:
            content = f.read()
        raw_doc_objs = re.findall(r"\[\[autodoc\]\]\s+(\S+)\s+", content)
        documented_obj += [obj.split(".")[-1] for obj in raw_doc_objs]
    return documented_obj


# One good reason for not being documented is to be deprecated. Put in this list deprecated objects.
DEPRECATED_OBJECTS = [
    "AutoModelWithLMHead",
    "BartPretrainedModel",
    "DataCollator",
    "DataCollatorForSOP",
    "GlueDataset",
    "GlueDataTrainingArguments",
    "LineByLineTextDataset",
    "LineByLineWithRefDataset",
    "LineByLineWithSOPTextDataset",
    "PretrainedBartModel",
    "PretrainedFSMTModel",
    "SingleSentenceClassificationProcessor",
    "SquadDataTrainingArguments",
    "SquadDataset",
    "SquadExample",
    "SquadFeatures",
    "SquadV1Processor",
    "SquadV2Processor",
    "TFAutoModelWithLMHead",
    "TFBartPretrainedModel",
    "TextDataset",
    "TextDatasetForNextSentencePrediction",
    "Wav2Vec2ForMaskedLM",
    "Wav2Vec2Tokenizer",
    "glue_compute_metrics",
    "glue_convert_examples_to_features",
    "glue_output_modes",
    "glue_processors",
    "glue_tasks_num_labels",
    "squad_convert_examples_to_features",
    "xnli_compute_metrics",
    "xnli_output_modes",
    "xnli_processors",
    "xnli_tasks_num_labels",
    "TFTrainer",
    "TFTrainingArguments",
]

# Exceptionally, some objects should not be documented after all rules passed.
# ONLY PUT SOMETHING IN THIS LIST AS A LAST RESORT!
UNDOCUMENTED_OBJECTS = [
    "AddedToken",  # This is a tokenizers class.
    "BasicTokenizer",  # Internal, should never have been in the main init.
    "CharacterTokenizer",  # Internal, should never have been in the main init.
    "DPRPretrainedReader",  # Like an Encoder.
    "DummyObject",  # Just picked by mistake sometimes.
    "MecabTokenizer",  # Internal, should never have been in the main init.
    "ModelCard",  # Internal type.
    "SqueezeBertModule",  # Internal building block (should have been called SqueezeBertLayer)
    "TFDPRPretrainedReader",  # Like an Encoder.
    "TransfoXLCorpus",  # Internal type.
    "WordpieceTokenizer",  # Internal, should never have been in the main init.
    "absl",  # External module
    "add_end_docstrings",  # Internal, should never have been in the main init.
    "add_start_docstrings",  # Internal, should never have been in the main init.
    "cached_path",  # Internal used for downloading models.
    "convert_tf_weight_name_to_pt_weight_name",  # Internal used to convert model weights
    "logger",  # Internal logger
    "logging",  # External module
    "requires_backends",  # Internal function
]

# This list should be empty. Objects in it should get their own doc page.
SHOULD_HAVE_THEIR_OWN_PAGE = [
    # Benchmarks
    "PyTorchBenchmark",
    "PyTorchBenchmarkArguments",
    "TensorFlowBenchmark",
    "TensorFlowBenchmarkArguments",
]


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
    modules = diffusers._modules
    objects = [c for c in dir(diffusers) if c not in modules and not c.startswith("_")]
    undocumented_objs = [c for c in objects if c not in documented_objs and not ignore_undocumented(c)]
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
    model_docs = [m.stem for m in model_doc_folder.glob("*.md")]

    model_types = list(diffusers.models.auto.configuration_auto.MODEL_NAMES_MAPPING.keys())
    model_types = [MODEL_TYPE_TO_DOC_MAPPING[m] if m in MODEL_TYPE_TO_DOC_MAPPING else m for m in model_types]

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
    print("Checking all models are included.")
    check_model_list()
    print("Checking all models are public.")
    check_models_are_in_init()
    print("Checking all models are properly tested.")
    check_all_decorator_order()
    check_all_models_are_tested()
    print("Checking all objects are properly documented.")
    check_all_objects_are_documented()
    print("Checking all models are in at least one auto class.")
    check_all_models_are_auto_configured()


if __name__ == "__main__":
    check_repo_quality()
