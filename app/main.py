from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.extractor import extract_job_offer
from app.models import JobOffer

app = FastAPI(
    title="jobspec",
    description="Extrae datos estructurados de ofertas de trabajo en texto libre.",
)


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=20000, description="Texto de la oferta")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/extract", response_model=JobOffer)
def extract(req: ExtractRequest) -> JobOffer:
    return extract_job_offer(req.text)
