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


def generate_pattern_for_mixin(mixin_class: Type) -> str:
    """
    Generate pytest pattern for a specific mixin class.
    """
    if mixin_cls is None:
        return ""
    test_methods = get_test_methods_from_class(mixin_class)
    return generate_pytest_pattern(test_methods)


if __name__ == "__main__":
    mixin_cls = None
    if args.type == "pipeline":
        from tests.pipelines.test_pipelines_common import PipelineTesterMixin

        mixin_cls = PipelineTesterMixin

    elif args.type == "models":
        from tests.models.test_modeling_common import ModelTesterMixin

        mixin_cls = ModelTesterMixin

    elif args.type == "lora":
        from tests.lora.utils import PeftLoraLoaderMixinTests

        mixin_cls = PeftLoraLoaderMixinTests

    pattern = generate_pattern_for_mixin(mixin_cls)
    print(pattern)
