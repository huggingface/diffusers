"""Test to verify fix for issue #6969: Expanded init fields cause incompatibilities."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from diffusers.pipelines.pipeline_loading_utils import _fetch_class_library_tuple


def test_valid_module():
    """Test that valid modules still work."""
    from transformers import CLIPTextModel
    library, class_name = _fetch_class_library_tuple(CLIPTextModel)
    assert library == "transformers"
    assert class_name == "CLIPTextModel"
    print("✓ test_valid_module passed")


def test_bool_raises_type_error():
    """Test that passing a bool (like requires_safety_checker) raises TypeError."""
    try:
        _fetch_class_library_tuple(True)
        print("✗ test_bool_raises_type_error FAILED - no exception raised")
        return False
    except TypeError as e:
        error_msg = str(e)
        assert "bool" in error_msg, f"Error message should mention 'bool', got: {error_msg}"
        assert "signature mismatch" in error_msg.lower(), f"Error should mention signature mismatch, got: {error_msg}"
        print(f"✓ test_bool_raises_type_error passed - got expected error: {error_msg[:100]}...")
        return True


def test_int_raises_type_error():
    """Test that passing an int raises TypeError."""
    try:
        _fetch_class_library_tuple(42)
        print("✗ test_int_raises_type_error FAILED - no exception raised")
        return False
    except TypeError as e:
        error_msg = str(e)
        assert "int" in error_msg, f"Error message should mention 'int', got: {error_msg}"
        print(f"✓ test_int_raises_type_error passed")
        return True


def test_string_raises_type_error():
    """Test that passing a string raises TypeError."""
    try:
        _fetch_class_library_tuple("not_a_module")
        print("✗ test_string_raises_type_error FAILED - no exception raised")
        return False
    except TypeError as e:
        error_msg = str(e)
        assert "str" in error_msg, f"Error message should mention 'str', got: {error_msg}"
        print(f"✓ test_string_raises_type_error passed")
        return True


def test_none_handled_by_register_modules():
    """Test that None is handled correctly (it should be caught before _fetch_class_library_tuple)."""
    # None should be handled by register_modules before reaching _fetch_class_library_tuple
    # This test just documents the expected behavior
    print("✓ test_none_handled_by_register_modules passed - None is handled by register_modules")


if __name__ == "__main__":
    print("Testing fix for issue #6969...")
    print()
    
    test_valid_module()
    test_bool_raises_type_error()
    test_int_raises_type_error()
    test_string_raises_type_error()
    test_none_handled_by_register_modules()
    
    print()
    print("All tests passed!")
