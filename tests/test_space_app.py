"""Smoke-test the Hugging Face Space handlers (skipped when gradio is not installed)."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("gradio")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "space"))
import app  # noqa: E402


def test_custom_run_and_history_analysis():
    md, frame = app.run_custom("collapse", 5, 0.2, False, 3, 10)
    assert "Ground truth" in md and "/page/" in frame
    md, frame = app.run_custom("healthy (no collapse)", 5, 0.2, False, 3, 8)
    assert "no collapse" in md
    with pytest.raises(Exception, match="onset"):
        app.run_custom("collapse", 25, 0.1, False, 1, 10)
    md, frame = app.analyze(app.sample_history(), "ALARM")
    assert "REGRESS regression check" in md


def test_http_routes():
    client = TestClient(app.app)
    assert client.get("/healthz").json()["ok"] is True
    assert client.get("/page/does-not-exist").status_code == 404
    _, frame = app.run_custom("collapse", 4, 0.2, False, 1, 8)
    key = frame.split("/page/")[1].split('"')[0]
    page = client.get(f"/page/{key}")
    assert page.status_code == 200 and "const DATA" in page.text


def test_bad_upload_is_a_friendly_error(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("not json")
    with pytest.raises(Exception, match="Could not read"):
        app.analyze(str(bad), "ALARM")
    with pytest.raises(Exception, match="Upload"):
        app.analyze(None, "ALARM")
