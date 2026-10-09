from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import extractor
from app.main import app

client = TestClient(app)
TEXT = "Acme busca Backend Dev Senior, remoto, Python y FastAPI. USD 4000-5000 por mes."


def fake_response(args: str, call_id: str = "c1"):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name="save_job_offer", arguments=args))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))])


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")


def patch_model(monkeypatch, responses):
    calls = []

    def fake(client_, messages):
        calls.append(list(messages))
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(extractor, "_call_model", fake)
    return calls


def test_ok(monkeypatch):
    patch_model(monkeypatch, [fake_response('{"title":"Dev","company":"Acme"}')])
    r = client.post("/extract", json={"text": TEXT})
    assert r.status_code == 200
    assert r.json()["company"] == "Acme"
    assert r.json()["location"] is None


def test_retry_fixes_missing_field(monkeypatch):
    calls = patch_model(
        monkeypatch,
        [fake_response('{"title":"Dev"}'), fake_response('{"title":"Dev","company":"Acme"}', "c2")],
    )
    r = client.post("/extract", json={"text": TEXT})
    assert r.status_code == 200
    assert len(calls) == 2
    assert "company" in calls[1][-1]["content"]  # el modelo vio qué campo faltaba


def test_gives_up_with_422(monkeypatch):
    calls = patch_model(monkeypatch, [fake_response('{"title":"Dev"}')])
    r = client.post("/extract", json={"text": TEXT})
    assert r.status_code == 422
    assert r.json()["detail"]["errors"][0]["field"] == "company"
    assert len(calls) == extractor.MAX_RETRIES + 1


def test_invalid_json_422(monkeypatch):
    patch_model(monkeypatch, [fake_response("no es json")])
    assert client.post("/extract", json={"text": TEXT}).status_code == 422


def test_missing_key_503(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    assert client.post("/extract", json={"text": TEXT}).status_code == 503


def test_upstream_502(monkeypatch):
    def boom(*a):
        raise extractor.UpstreamError("Groq caído")

    monkeypatch.setattr(extractor, "_call_model", boom)
    assert client.post("/extract", json={"text": TEXT}).status_code == 502


def test_short_text_422():
    assert client.post("/extract", json={"text": "hola"}).status_code == 422


def test_placeholder_company_is_rejected(monkeypatch):
    patch_model(monkeypatch, [fake_response('{"title":"Dev","company":"Desconocida"}')])
    r = client.post("/extract", json={"text": TEXT})
    assert r.status_code == 422
    assert r.json()["detail"]["errors"][0]["field"] == "company"
