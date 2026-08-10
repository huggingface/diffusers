import os
import sys
from unittest.mock import mock_open, patch


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

from check_support_list import check_documentation  # noqa: E402


# Mock doc and source contents that we can reuse
DOC_CONTENT = """# Documentation
## FooProcessor

[[autodoc]] module.FooProcessor

## BarProcessor

[[autodoc]] module.BarProcessor
"""
SOURCE_CONTENT = """
class FooProcessor(nn.Module):
    pass

class BarProcessor(nn.Module):
    pass
"""


class TestCheckSupportList:
    def test_check_documentation_all_documented(self):
        # In this test, both FooProcessor and BarProcessor are documented
        with patch("builtins.open", mock_open(read_data=DOC_CONTENT)) as doc_file:
            doc_file.side_effect = [
                mock_open(read_data=DOC_CONTENT).return_value,
                mock_open(read_data=SOURCE_CONTENT).return_value,
            ]

            undocumented = check_documentation(
                doc_path="fake_doc.md",
                src_path="fake_source.py",
                doc_regex=r"\[\[autodoc\]\]\s([^\n]+)",
                src_regex=r"class\s+(\w+Processor)\(.*?nn\.Module.*?\):",
            )
            assert len(undocumented) == 0, f"Expected no undocumented classes, got {undocumented}"

    def test_check_documentation_missing_class(self):
        # In this test, only FooProcessor is documented, but BarProcessor is missing from the docs
        doc_content_missing = """# Documentation
## FooProcessor

[[autodoc]] module.FooProcessor
"""
        with patch("builtins.open", mock_open(read_data=doc_content_missing)) as doc_file:
            doc_file.side_effect = [
                mock_open(read_data=doc_content_missing).return_value,
                mock_open(read_data=SOURCE_CONTENT).return_value,
            ]

            undocumented = check_documentation(
                doc_path="fake_doc.md",
                src_path="fake_source.py",
                doc_regex=r"\[\[autodoc\]\]\s([^\n]+)",
                src_regex=r"class\s+(\w+Processor)\(.*?nn\.Module.*?\):",
            )
            assert "BarProcessor" in undocumented, f"BarProcessor should be undocumented, got {undocumented}"
