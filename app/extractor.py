import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError

from app.models import JobOffer

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TOOL_NAME = "save_job_offer"

SYSTEM_PROMPT = (
    "Sos un extractor de datos de ofertas de trabajo. Leé el texto del usuario y "
    "llamá a la función save_job_offer con los datos que aparezcan en el texto. "
    "Reglas: no inventes nada; si un dato no está en el texto, dejalo en null "
    "(o lista vacía). Los skills van como nombres cortos (ej: 'Python', 'Docker'). "
    "Los salarios son números sin símbolos ni separadores."
)


class ExtractionError(Exception):
    """El modelo no devolvió una oferta válida."""


def _inline_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Pydantic genera $defs/$ref para los modelos anidados; los reemplazamos por
    el contenido, así el esquema es autocontenido y lo entiende cualquier proveedor."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Guarda los datos estructurados de una oferta de trabajo.",
        "parameters": _inline_refs(JobOffer.model_json_schema()),
    },
}


def _client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ExtractionError("Falta GROQ_API_KEY en el entorno (.env)")
    return Groq(api_key=key)


def extract_job_offer(text: str) -> JobOffer:
    """Extrae una JobOffer de texto libre forzando al modelo a llamar a la herramienta."""
    response = _client().chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        tools=[TOOL],
        # Obliga al modelo a llamar a esta función (no puede responder con texto libre).
        tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise ExtractionError("El modelo no llamó a la herramienta")

    try:
        args = json.loads(tool_calls[0].function.arguments)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Argumentos no son JSON válido: {e}") from e

    try:
        return JobOffer.model_validate(args)
    except ValidationError as e:
        raise ExtractionError(f"La salida no cumple el esquema: {e}") from e
