import pytest

from diffusers.pipelines.longcat_image.pipeline_longcat_image import (
    logger as base_logger,
    prepare_pos_ids as prepare_pos_ids_base,
)
from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import (
    logger as edit_logger,
    prepare_pos_ids as prepare_pos_ids_edit,
)
from ...testing_utils import CaptureLogger


@pytest.mark.parametrize(
    ("prepare_pos_ids", "logger"),
    (
        (prepare_pos_ids_base, base_logger),
        (prepare_pos_ids_edit, edit_logger),
    ),
)
def test_prepare_pos_ids_logs_ignored_text_dimensions(prepare_pos_ids, logger):
    with CaptureLogger(logger) as captured:
        prepare_pos_ids(type="text", num_token=4, height=8, width=8)

    assert 'ignored when `type="text"`' in captured.out


@pytest.mark.parametrize(
    ("prepare_pos_ids", "logger"),
    (
        (prepare_pos_ids_base, base_logger),
        (prepare_pos_ids_edit, edit_logger),
    ),
)
def test_prepare_pos_ids_logs_ignored_image_num_token(prepare_pos_ids, logger):
    with CaptureLogger(logger) as captured:
        prepare_pos_ids(type="image", num_token=4, height=2, width=3)

    assert 'ignored when `type="image"`' in captured.out


@pytest.mark.parametrize("prepare_pos_ids", (prepare_pos_ids_base, prepare_pos_ids_edit))
def test_prepare_pos_ids_unknown_type_message(prepare_pos_ids):
    with pytest.raises(KeyError, match='Unknown type audio'):
        prepare_pos_ids(type="audio", num_token=4)
