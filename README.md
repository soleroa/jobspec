# jobspec

A small FastAPI service that turns a free-text job posting into validated JSON: company, title, salary, required skills, work mode, and more.

It uses LLM **tool calling** (via [Groq](https://groq.com)) to force the model to answer with a fixed schema, and [Pydantic](https://docs.pydantic.dev) to validate the result. If the model can't fill a required field, the API returns a clear error instead of making something up.

## How it works

```
POST /extract  ──►  forced tool call (Groq)  ──►  Pydantic validation  ──►  JSON
                          ▲                               │
                          └──── retry with the exact ─────┘
                                validation errors
```

1. The JSON schema of the `JobOffer` Pydantic model is sent to the LLM as a tool (`save_job_offer`), and `tool_choice` forces the model to call it, so it can't answer with free text.
2. The tool arguments are validated with Pydantic (types, enums, salary range, currency code, placeholder values).
3. If validation fails, the model gets the failing fields back and is asked to fix only those, up to `MAX_RETRIES` times.
4. If it still fails, the API responds with an HTTP error that names the problem.

**Missing data is never invented.** Only `title` and `company` are required. Everything else is `null` (or an empty list) when the text doesn't mention it. Filler values such as `"Unknown"` or `"N/A"` in required fields are rejected.

## Requirements

- Python 3.9+
- A Groq API key (free at <https://console.groq.com/keys>)

## Setup

```bash
git clone <your-repo-url> jobspec
cd jobspec

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY=gsk_...
```

### Configuration

| Variable       | Default               | Description                                         |
| -------------- | --------------------- | --------------------------------------------------- |
| `GROQ_API_KEY` | _(required)_          | Your Groq API key                                   |
| `GROQ_MODEL`   | `openai/gpt-oss-120b` | Any Groq model that supports tool calling           |
| `MAX_RETRIES`  | `2`                   | Extra attempts when the model returns invalid data  |

## Run

Make sure the virtual environment is active, then start the server:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

There is no separate frontend to build or start: the web UI is a single static page served by the same FastAPI app. Once the server is running, open:

- **Web UI:** <http://localhost:8000/> — paste a job posting link or its text
- **Interactive API docs:** <http://localhost:8000/docs>

To use a different port: `uvicorn app.main:app --reload --port 8080`.

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /extract`

**Request**

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Mercado Libre busca Data Engineer semi senior en Buenos Aires, modalidad híbrida. Requisitos: SQL, Python, Airflow. Se valora conocimiento de Spark. Salario ARS 2.500.000 mensuales."}'
```

`text` must be between 20 and 20,000 characters.

**Response `200 OK`**

```json
{
  "title": "Data Engineer",
  "company": "Mercado Libre",
  "seniority": "semi_senior",
  "work_mode": "hybrid",
  "location": {"city": "Buenos Aires", "country": null},
  "salary": {"min": 2500000.0, "max": 2500000.0, "currency": "ARS", "period": "month"},
  "required_skills": ["SQL", "Python", "Airflow"],
  "nice_to_have_skills": ["Spark"],
  "years_experience": null
}
```

Fields the text doesn't mention come back as `null` (or `[]` for skill lists).

### `POST /extract/url`

Same output as `/extract`, but takes a link. The server downloads the page and extracts the posting from it, preferring the page's structured `JobPosting` data (schema.org JSON-LD) and falling back to the visible text.

```bash
curl -X POST http://localhost:8000/extract/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://jobs.lever.co/company/1234-abcd"}'
```

Limitations:

- Works with public pages that include the posting in their HTML (company career pages, Greenhouse, Lever, many local job boards).
- Sites like LinkedIn, Indeed or Glassdoor usually require login, block bots or render with JavaScript. The API then returns a `422` asking you to paste the text with `/extract` instead.
- Only public `http(s)` addresses are fetched; localhost and private networks are blocked (SSRF protection).

### Output schema

| Field                 | Type                                                  | Required |
| --------------------- | ----------------------------------------------------- | -------- |
| `title`               | string                                                | yes      |
| `company`             | string                                                | yes      |
| `seniority`           | `intern` \| `junior` \| `semi_senior` \| `senior` \| `lead` | no |
| `work_mode`           | `remote` \| `hybrid` \| `onsite`                      | no       |
| `location`            | `{city, country}`                                     | no       |
| `salary`              | `{min, max, currency (ISO 4217), period (hour/month/year)}` | no |
| `required_skills`     | string[]                                              | no (`[]`) |
| `nice_to_have_skills` | string[]                                              | no (`[]`) |
| `years_experience`    | integer (0–50)                                        | no       |

### Errors

| Status | When                                                                          |
| ------ | ----------------------------------------------------------------------------- |
| `422`  | Invalid request body (e.g. text too short), **or** the text doesn't contain a required field and the model couldn't fill it after retries, **or** (`/extract/url`) the page is unreadable, blocked, expired or not allowed |
| `502`  | Groq returned an error or couldn't be reached, or (`/extract/url`) the site failed or timed out |
| `503`  | `GROQ_API_KEY` is not configured                                              |

**Example: a posting with no company name**

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Buscamos desarrollador Python con experiencia en Django. Trabajo remoto, pago a convenir. Enviar CV."}'
```

```json
{
  "detail": {
    "message": "No se pudo extraer una oferta válida del texto",
    "errors": [
      {"field": "company", "error": "expected string, but got null"}
    ]
  }
}
```

## Tests

The tests mock the LLM, so they run offline and don't need an API key:

```bash
pytest
```

They cover the happy path, a successful retry, giving up after the retries, invalid JSON from the model, placeholder values, a missing API key, an upstream failure, too-short input, URL extraction (JSON-LD, plain HTML, blocked addresses) and the UI being served.

## Project structure

```
app/
├── main.py        # FastAPI app and HTTP error mapping
├── fetcher.py     # URL download, SSRF protection, HTML -> job text
├── static/        # web UI (index.html)
├── models.py      # Pydantic models and validators (JobOffer, Salary, ...)
└── extractor.py   # Tool calling, retries and error handling
tests/             # pytest suite (LLM mocked)
samples/           # real job postings for manual testing
```
