"""HTTP contract tests; no model or GPU is required."""
import pytest
from fastapi.testclient import TestClient
from src.server import MAX_BODY_BYTES, create_app

RESULT = {"issue": "Database unavailable", "root_cause": "Connection timed out", "severity": "high", "fix": "Check database connectivity", "confidence": "70%"}


class FakeRuntime:
    def analyze(self, logs):
        assert logs == "database timeout"
        return RESULT


@pytest.fixture
def client():
    with TestClient(create_app(loader=FakeRuntime, api_key="test-key")) as client:
        yield client


def test_requires_auth_and_accepts_request(client):
    assert client.post("/v1/analyze", json={"logs": "database timeout"}).status_code == 401
    response = client.post("/v1/analyze", json={"logs": "database timeout"}, headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json() == RESULT


def test_validation_and_body_size(client):
    headers = {"Authorization": "Bearer test-key"}
    assert client.post("/v1/analyze", json={"logs": " "}, headers=headers).status_code == 422
    assert client.post("/v1/analyze", json={"logs": ""}, headers=headers).status_code == 422
    assert client.post("/v1/analyze", content=b"x" * (MAX_BODY_BYTES + 1), headers=headers).status_code == 413


def test_health(client):
    assert client.get("/health").json() == {"status": "ready"}


def test_bad_generation_is_not_exposed():
    from src.inference import InferenceOutputError

    class Broken:
        def analyze(self, logs):
            raise InferenceOutputError("sensitive model output")

    with TestClient(create_app(loader=Broken, api_key="secret")) as client:
        response = client.post("/v1/analyze", json={"logs": "secret log"}, headers={"Authorization": "Bearer secret"})
        assert response.status_code == 502
        assert "sensitive" not in response.text
        assert "secret log" not in response.text


def test_key_is_required():
    with pytest.raises(RuntimeError, match="LOGSAGE_API_KEY"):
        with TestClient(create_app(loader=FakeRuntime, api_key="")):
            pass
