from io import BytesIO

import pytest
from fastapi import HTTPException

from backend.app.upload_utils import parse_allowed_extensions, save_upload, supported_extension


def test_extensions_are_matched_exactly(tmp_path):
    assert parse_allowed_extensions(".pdf, .txt") == {".pdf", ".txt"}
    assert supported_extension("report.PDF", ".pdf,.txt") == ".pdf"
    with pytest.raises(HTTPException) as exc:
        supported_extension("report.pdf.exe", ".pdf,.txt")
    assert exc.value.status_code == 400


def test_upload_is_saved_within_limit(tmp_path):
    target = tmp_path / "small.txt"
    save_upload(BytesIO(b"hello"), str(target), 5)
    assert target.read_bytes() == b"hello"


def test_upload_stops_at_limit_plus_one_and_rejects(tmp_path):
    class TrackedStream(BytesIO):
        bytes_read = 0

        def read(self, size=-1):
            chunk = super().read(size)
            self.bytes_read += len(chunk)
            return chunk

    source = TrackedStream(b"x" * 100)
    target = tmp_path / "large.txt"
    with pytest.raises(HTTPException) as exc:
        save_upload(source, str(target), 8)
    assert exc.value.status_code == 413
    assert source.bytes_read == 9
    assert target.read_bytes() == b""
