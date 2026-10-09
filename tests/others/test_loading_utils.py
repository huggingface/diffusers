from io import BytesIO

import pytest
from PIL import Image, UnidentifiedImageError

from diffusers.utils import load_video


class FakeResponse:
    status_code = 200

    def __init__(self, data, interrupt=False):
        self.data = data
        self.interrupt = interrupt

    def iter_content(self, chunk_size=8192):
        yield self.data
        if self.interrupt:
            raise ConnectionError("Download interrupted")


def mock_remote(monkeypatch, tmp_path, response):
    monkeypatch.setattr(
        "diffusers.utils.loading_utils.requests.get",
        lambda *args, **kwargs: response,
    )
    monkeypatch.setattr(
        "diffusers.utils.loading_utils.tempfile.tempdir",
        str(tmp_path),
    )


@pytest.mark.parametrize("interrupt", [False, True])
def test_remote_video_failure_cleans_tempfile(tmp_path, monkeypatch, interrupt):
    response = FakeResponse(b"invalid-gif-data", interrupt=interrupt)
    mock_remote(monkeypatch, tmp_path, response)

    expected = ConnectionError if interrupt else UnidentifiedImageError

    with pytest.raises(expected):
        load_video("https://example.com/broken.gif")

    assert not list(tmp_path.iterdir())


def test_remote_video_success_cleans_tempfile(tmp_path, monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (4, 4), "red").save(
        buffer, format="GIF", duration=100
    )

    mock_remote(monkeypatch, tmp_path, FakeResponse(buffer.getvalue()))

    frames, fps = load_video(
        "https://example.com/valid.gif", return_fps=True
    )

    assert len(frames) == 1
    assert fps == pytest.approx(10.0)
    assert not list(tmp_path.iterdir())


def test_local_video_is_preserved(tmp_path):
    video_path = tmp_path / "local.gif"
    Image.new("RGB", (4, 4), "red").save(video_path, format="GIF")

    frames = load_video(str(video_path))

    assert len(frames) == 1
    assert video_path.exists()
