# list_deprecated_test_classes.py
import ast
import importlib
import inspect
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.resolve()
TESTS_DIR = PROJECT_ROOT / "tests" / "pipelines"
PIPELINE_PACKAGE = "diffusers"

def find_deprecated_test_classes():
    """
    Finds test files and the specific test class names that test deprecated pipelines.
    """
    print(f"🔍 Project root set to: {PROJECT_ROOT}")
    print(f"🔍 Recursively searching for deprecated test classes in: {TESTS_DIR}\n")
    deprecated_tests = {}

    # 1. Get the DeprecatedPipelineMixin class to check against.
    try:
        mixin_module = importlib.import_module(f"{PIPELINE_PACKAGE}.pipelines.pipeline_utils")
        DeprecatedPipelineMixin = getattr(mixin_module, 'DeprecatedPipelineMixin')
    except (ImportError, AttributeError):
        print("❌ Error: Could not import DeprecatedPipelineMixin.")
        return {}

    if not TESTS_DIR.is_dir():
        print(f"❌ Error: The directory '{TESTS_DIR}' does not exist.")
        return {}

    # 2. Recursively find all test files.
    for filepath in sorted(TESTS_DIR.rglob("test_*.py")):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source_code = f.read()
            tree = ast.parse(source_code)
            
            relative_path = str(filepath.relative_to(PROJECT_ROOT))
            deprecated_pipelines_in_file = set()

            # 3. First pass: find all imported pipelines in the file that are deprecated.
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(PIPELINE_PACKAGE):
                    for alias in node.names:
                        try:
                            pipeline_module = importlib.import_module(node.module)
                            pipeline_class = getattr(pipeline_module, alias.name)
                            if inspect.isclass(pipeline_class) and issubclass(pipeline_class, DeprecatedPipelineMixin):
                                deprecated_pipelines_in_file.add(pipeline_class.__name__)
                        except (AttributeError, ImportError, TypeError):
                            continue
            
            if not deprecated_pipelines_in_file:
                continue

            # 4. Second pass: find test classes that use these deprecated pipelines.
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Heuristic: Check if the class name contains the name of a deprecated pipeline.
                    # This is a robust way to link a test class to the pipeline it tests.
                    for pipeline_name in deprecated_pipelines_in_file:
                        # e.g., Pipeline: BlipDiffusionPipeline, Test Class: BlipDiffusionPipelineTests
                        if pipeline_name in node.name:
                            if relative_path not in deprecated_tests:
                                deprecated_tests[relative_path] = []
                            deprecated_tests[relative_path].append(node.name)
                            
        except Exception as e:
            print(f"⚠️  Could not process {filepath.name}: {e}")
            continue

    return deprecated_tests

if __name__ == "__main__":
    found_tests = find_deprecated_test_classes()
    
    if found_tests:
        print("\n" + "="*70)
        print("✅ Found Test Files and Classes for Deprecated Pipelines:")
        print("="*70)
        for file, classes in found_tests.items():
            print(f"\n📄 File: {file}")
            for cls in sorted(list(set(classes))):
                print(f"   - Test Class: {cls}")
    else:
        print("\nNo deprecated pipeline test classes were found.")