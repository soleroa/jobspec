from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.extractor import (
    ConfigError,
    ExtractionError,
    UpstreamError,
    extract_job_offer,
)
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
    try:
        return extract_job_offer(req.text)
    except ExtractionError as e:
        # El texto no tenía datos suficientes (ej: falta la empresa) o el modelo no se corrigió.
        raise HTTPException(
            status_code=422,
            detail={"message": "No se pudo extraer una oferta válida del texto", "errors": e.details},
        )
    except UpstreamError as e:
        raise HTTPException(status_code=502, detail={"message": str(e)})
    except ConfigError as e:
        raise HTTPException(status_code=503, detail={"message": str(e)})
