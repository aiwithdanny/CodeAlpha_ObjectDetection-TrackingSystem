"""Web API tests (tracker mocked, koi model load nahi hota)."""

import io

import pytest

from tests.fakes import install_fake_ultralytics

install_fake_ultralytics()

from app import create_app  # noqa: E402
from config import Config  # noqa: E402


class TestConfig(Config):
    TESTING = True
    MAX_UPLOAD_MB = 1


@pytest.fixture
def client(tmp_path):
    app = create_app(TestConfig)
    app.config["UPLOAD_FOLDER"] = str(tmp_path / "up")
    app.config["OUTPUT_FOLDER"] = str(tmp_path / "out")
    import os

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
    return app.test_client()


def test_health(client):
    assert client.get("/health").status_code == 200


def test_upload_no_file_400(client):
    assert client.post("/api/upload").status_code == 400


def test_upload_bad_extension_400(client):
    data = {"video": (io.BytesIO(b"xyz"), "evil.exe")}
    res = client.post("/api/upload", data=data,
                      content_type="multipart/form-data")
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_job_not_found_404(client):
    assert client.get("/api/job/nope123").status_code == 404


def test_result_not_ready_404(client):
    assert client.get("/result/nope123").status_code == 404
