from pathlib import Path
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


# ---- URL extraction ----
from app import fetcher  # noqa: E402

JOB_HTML = """<html><head><script type="application/ld+json">
{"@context":"https://schema.org","@type":"JobPosting","title":"Backend Developer",
 "hiringOrganization":{"@type":"Organization","name":"Acme"},
 "jobLocation":{"address":{"addressLocality":"Rosario","addressCountry":"AR"}},
 "description":"<p>We need someone with Python and FastAPI experience to build APIs for our platform team.</p>"}
</script></head><body>menu</body></html>"""


def test_jobposting_jsonld_is_preferred():
    text = fetcher.html_to_job_text(JOB_HTML)
    assert "Company: Acme" in text and "Rosario" in text and "FastAPI" in text and "<p>" not in text


def test_plain_html_fallback_drops_scripts():
    html = "<html><body><nav>menu</nav><script>var x=1</script><h1>Dev</h1><p>Python</p></body></html>"
    text = fetcher.html_to_job_text(html)
    assert "Dev" in text and "Python" in text and "var x" not in text and "menu" not in text


@pytest.mark.parametrize("url", ["http://localhost/x", "http://127.0.0.1/", "http://169.254.169.254/", "file:///etc/passwd", "ftp://a.com"])
def test_ssrf_blocked(url):
    with pytest.raises(fetcher.FetchError):
        fetcher._check_public_url(url)


def test_extract_url_ok(monkeypatch):
    monkeypatch.setattr("app.main.fetch_job_text", lambda url: "Acme busca Dev. " * 20)
    patch_model(monkeypatch, [fake_response('{"title":"Dev","company":"Acme"}')])
    r = client.post("/extract/url", json={"url": "https://example.com/job"})
    assert r.status_code == 200 and r.json()["company"] == "Acme"


def test_extract_url_unreadable_page(monkeypatch):
    def blocked(url):
        raise fetcher.FetchError("bloqueado", 422)

    monkeypatch.setattr("app.main.fetch_job_text", blocked)
    r = client.post("/extract/url", json={"url": "https://example.com/job"})
    assert r.status_code == 422 and r.json()["detail"]["message"] == "bloqueado"


def test_ui_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "jobspec" in r.text


# ---- Extraction quality (LLM mocked) ----
import json  # noqa: E402

from app.models import JobOffer  # noqa: E402

FLATIRON = (Path(__file__).resolve().parent.parent / "samples" / "flatiron_software_engineer.txt").read_text()


def extract_with(monkeypatch, payload: dict, text: str = FLATIRON) -> dict:
    patch_model(monkeypatch, [fake_response(json.dumps(payload))])
    r = client.post("/extract", json={"text": text})
    assert r.status_code == 200, r.text
    return r.json()


def test_seniority_inferred_from_years_is_dropped(monkeypatch):
    # El modelo deduce "senior" desde "4+ years"; el texto nunca dice senior.
    out = extract_with(monkeypatch, {"title": "Software Engineer", "company": "Flatiron", "seniority": "senior", "years_experience": 4})
    assert out["seniority"] is None
    assert out["years_experience"] == 4


@pytest.mark.parametrize(
    "word,level",
    [("Senior", "senior"), ("Sr.", "senior"), ("Junior", "junior"), ("Semi-Senior", "semi_senior"), ("Tech Lead", "lead")],
)
def test_explicit_seniority_is_kept(monkeypatch, word, level):
    text = f"Acme busca {word} Backend Developer, remoto. Python y FastAPI, 3 años de experiencia."
    out = extract_with(monkeypatch, {"title": "Backend Developer", "company": "Acme", "seniority": level}, text)
    assert out["seniority"] == level


def test_alternatives_are_not_all_required(monkeypatch):
    out = extract_with(
        monkeypatch,
        {
            "title": "Software Engineer",
            "company": "Flatiron",
            "required_skills": ["React", "AWS", "GCP", "Azure", "postgresql"],
            "skill_alternatives": [["AWS", "GCP", "Azure"], ["MySQL", "MongoDB", "PostgreSQL"], ["Solo"]],
        },
    )
    # Las skills que están en un grupo salen de required_skills; el grupo de 1 se descarta.
    assert out["required_skills"] == ["React"]
    assert out["skill_alternatives"] == [["AWS", "GCP", "Azure"], ["MySQL", "MongoDB", "PostgreSQL"]]


def test_new_fields_languages_and_timezone(monkeypatch):
    out = extract_with(
        monkeypatch,
        {"title": "Software Engineer", "company": "Flatiron", "languages": ["English (advanced)", "english (advanced)"], "timezone": " US hours "},
    )
    assert out["languages"] == ["English (advanced)"]
    assert out["timezone"] == "US hours"


def test_new_fields_default_to_empty(monkeypatch):
    out = extract_with(monkeypatch, {"title": "Dev", "company": "Acme"}, TEXT)
    assert out["languages"] == [] and out["timezone"] is None and out["skill_alternatives"] == []


def test_schema_and_prompt_carry_the_rules():
    props = extractor.TOOL["function"]["parameters"]["properties"]
    assert "NUNCA" in props["seniority"]["description"]
    assert "skill_alternatives" in props["required_skills"]["description"]
    assert {"skill_alternatives", "languages", "timezone"} <= set(props)
    assert "años de experiencia" in extractor.SYSTEM_PROMPT and "Express" in extractor.SYSTEM_PROMPT
    assert "'AI'" in extractor.SYSTEM_PROMPT


def test_sample_is_saved():
    assert "Flatiron" in FLATIRON and "NestJS" in FLATIRON
