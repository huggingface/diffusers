import argparse
import inspect
import sys
from pathlib import Path
from typing import List, Type


root_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_dir))

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default=None)
args = parser.parse_args()


def get_test_methods_from_class(cls: Type) -> List[str]:
    """
    Get all test method names from a given class.
    Only returns methods that start with 'test_'.
    """
    test_methods = []
    for name, obj in inspect.getmembers(cls):
        if name.startswith("test_") and inspect.isfunction(obj):
            test_methods.append(name)
    return sorted(test_methods)


def generate_pytest_pattern(test_methods: List[str]) -> str:
    """Generate pytest pattern string for the -k flag."""
    return " or ".join(test_methods)


def generate_pattern_for_mixins(mixin_classes: List[Type]) -> str:
    """
    Generate a pytest pattern covering the test methods of all the given mixin classes.
    """
    test_methods = set()
    for mixin_class in mixin_classes:
        test_methods.update(get_test_methods_from_class(mixin_class))
    return generate_pytest_pattern(sorted(test_methods))


if __name__ == "__main__":
    mixin_classes = []
    if args.type == "pipeline":
        from tests.pipelines.test_pipelines_common import PipelineTesterMixin

        mixin_classes = [PipelineTesterMixin]

    elif args.type == "models":
        # The model tester suite is split across several mixins under `tests/models/testing_utils`,
        # so aggregate their test methods to reconstruct the full coverage.
        from tests.models.testing_utils import (
            AttentionTesterMixin,
            LoraTesterMixin,
            MemoryTesterMixin,
            ModelTesterMixin,
            TrainingTesterMixin,
        )

        mixin_classes = [
            ModelTesterMixin,
            MemoryTesterMixin,
            TrainingTesterMixin,
            AttentionTesterMixin,
            LoraTesterMixin,
        ]

    elif args.type == "lora":
        from tests.lora.utils import PeftLoraLoaderMixinTests

        mixin_classes = [PeftLoraLoaderMixinTests]

    pattern = generate_pattern_for_mixins(mixin_classes)
    print(pattern)
