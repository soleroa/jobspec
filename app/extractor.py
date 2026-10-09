import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import groq
from groq import Groq
from pydantic import ValidationError

from app.models import JobOffer

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Reintentos extra (además del primer intento) cuando la salida es inválida.
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
TOOL_NAME = "save_job_offer"

SYSTEM_PROMPT = (
    "Sos un extractor de datos de ofertas de trabajo. Leé el texto del usuario y "
    "llamá a la función save_job_offer con los datos que aparezcan en el texto. "
    "Reglas: no inventes nada; si un dato no está en el texto, dejalo en null "
    "(o lista vacía). Los skills van como nombres cortos (ej: 'Python', 'Docker'). "
    "Los salarios son números sin símbolos ni separadores."
)

RETRY_RULE = (
    "Corregí solo lo que falló. Si el dato NO aparece en el texto original, no lo inventes "
    "ni lo deduzcas: dejalo en null si el esquema lo permite. Usá únicamente información del texto."
)


class ExtractorError(Exception):
    """Base de los errores de extracción."""


class ConfigError(ExtractorError):
    """Falta configuración (ej: GROQ_API_KEY). -> 503"""


class UpstreamError(ExtractorError):
    """Groq no respondió o devolvió un error. -> 502"""


class ExtractionError(ExtractorError):
    """El modelo no logró devolver una oferta válida tras los reintentos. -> 422"""

    def __init__(self, message: str, details: Optional[list] = None):
        super().__init__(message)
        self.details = details or []


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
        raise ConfigError("Falta GROQ_API_KEY en el entorno (.env)")
    return Groq(api_key=key)


def _call_model(client: Groq, messages: List[dict]):
    try:
        return client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=messages,
            tools=[TOOL],
            # Obliga al modelo a llamar a esta función (no puede responder con texto libre).
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
    except groq.BadRequestError as e:
        # Groq devuelve 400 (tool_use_failed) cuando el modelo genera una llamada mal formada.
        # El mensaje de Groq trae los campos que fallaron: "`/company`: expected string, but got null".
        details = [
            {"field": f.replace("/", "."), "error": err.strip()}
            for f, err in re.findall(r"`/([\w./]+)`: ([^`\]]+)", str(e))
        ]
        raise _BadModelOutput("El modelo generó una llamada inválida", details) from e
    except groq.APIError as e:
        raise UpstreamError(f"Error al llamar a Groq: {e}") from e


class _BadModelOutput(Exception):
    """Salida inválida del modelo; se puede reintentar."""

    def __init__(self, message: str, details: Optional[list] = None):
        super().__init__(message)
        self.details = details or []


def _parse(tool_call) -> JobOffer:
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        raise _BadModelOutput(f"Los argumentos no son JSON válido: {e}") from e
    try:
        return JobOffer.model_validate(args)
    except ValidationError as e:
        details = [
            {"field": ".".join(str(p) for p in err["loc"]), "error": err["msg"]}
            for err in e.errors()
        ]
        raise _BadModelOutput("La salida no cumple el esquema", details) from e


def extract_job_offer(text: str) -> JobOffer:
    """Extrae una JobOffer de texto libre forzando al modelo a llamar a la herramienta.

    Si la salida es inválida, reintenta mostrándole al modelo qué campos fallaron.
    """
    client = _client()
    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    last: Optional[_BadModelOutput] = None

    for _ in range(MAX_RETRIES + 1):
        call = None
        try:
            response = _call_model(client, messages)
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise _BadModelOutput("El modelo no llamó a la herramienta")
            call = tool_calls[0]
            return _parse(call)
        except _BadModelOutput as e:
            last = e
            if call is None:
                # No hay llamada a la que responder (Groq rechazó la llamada): reintentamos
                # con un mensaje más específico en vez de repetir el mismo pedido.
                if e.details:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Tu llamada fue rechazada. Detalle: {json.dumps(e.details, ensure_ascii=False)}. {RETRY_RULE}",
                        }
                    )
                continue
            # Devolvemos el error como resultado de la herramienta para que el modelo se corrija.
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                    ],
                }
            )
            details = json.dumps(e.details, ensure_ascii=False)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": f"Error: {e}. Detalle: {details}. {RETRY_RULE} Volvé a llamar a la función.",
                }
            )

    raise ExtractionError(str(last), last.details if last else None)
