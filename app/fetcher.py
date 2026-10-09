import ipaddress
import json
import socket
from typing import Any, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

MAX_BYTES = 2_000_000
MAX_REDIRECTS = 5
MIN_TEXT_CHARS = 200
MAX_TEXT_CHARS = 20000
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; jobspec/0.1)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "es,en;q=0.8",
}


class FetchError(Exception):
    """No se pudo obtener o leer la página. `status` es el código HTTP sugerido."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def _check_public_url(url: str) -> None:
    """Evita SSRF: solo http(s) y solo hosts que resuelvan a IPs públicas."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise FetchError("La URL debe empezar con http:// o https://", 422)
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror:
        raise FetchError("No se pudo resolver el dominio de esa URL", 422)
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise FetchError("Esa URL apunta a una dirección no permitida", 422)


def fetch_html(url: str) -> str:
    """Descarga una página siguiendo redirects a mano, validando cada salto."""
    try:
        with httpx.Client(timeout=15, headers=HEADERS, follow_redirects=False) as client:
            for _ in range(MAX_REDIRECTS + 1):
                _check_public_url(url)
                with client.stream("GET", url) as r:
                    if r.is_redirect:
                        url = urljoin(url, r.headers.get("location", ""))
                        continue
                    if r.status_code in (401, 403, 429, 999):
                        raise FetchError(
                            "El sitio bloqueó el acceso (pide login o bloquea bots). "
                            "Copiá el texto de la oferta y pegalo en la otra pestaña.",
                            422,
                        )
                    if r.status_code in (404, 410):
                        raise FetchError("Esa oferta ya no existe (el link está roto o venció)", 422)
                    if r.status_code >= 400:
                        raise FetchError(f"El sitio respondió con error {r.status_code}", 502)
                    ctype = r.headers.get("content-type", "")
                    if "html" not in ctype and "text" not in ctype:
                        raise FetchError("La URL no es una página web de texto", 422)
                    body = b""
                    for chunk in r.iter_bytes():
                        body += chunk
                        if len(body) > MAX_BYTES:
                            break
                    return body[:MAX_BYTES].decode(r.encoding or "utf-8", errors="replace")
            raise FetchError("Demasiadas redirecciones", 422)
    except httpx.TimeoutException:
        raise FetchError("El sitio tardó demasiado en responder", 502)
    except httpx.HTTPError as e:
        raise FetchError(f"No se pudo descargar la página: {e}", 502)


def _jobposting_nodes(data: Any) -> List[dict]:
    """Busca objetos schema.org JobPosting dentro de un JSON-LD (puede venir anidado)."""
    found: List[dict] = []
    if isinstance(data, list):
        for item in data:
            found += _jobposting_nodes(item)
    elif isinstance(data, dict):
        t = data.get("@type")
        if t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t):
            found.append(data)
        found += _jobposting_nodes(data.get("@graph"))
    return found


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "svg", "form"]):
        tag.decompose()
    lines = (ln.strip() for ln in soup.get_text("\n").splitlines())
    return "\n".join(ln for ln in lines if ln)


def _jobposting_to_text(job: dict) -> str:
    org = job.get("hiringOrganization")
    company = org.get("name") if isinstance(org, dict) else org
    loc = job.get("jobLocation")
    if isinstance(loc, list) and loc:
        loc = loc[0]
    addr = loc.get("address", {}) if isinstance(loc, dict) else {}
    place = ", ".join(
        str(addr[k]) for k in ("addressLocality", "addressRegion", "addressCountry")
        if isinstance(addr, dict) and addr.get(k)
    )
    parts = [
        f"Title: {job.get('title')}" if job.get("title") else "",
        f"Company: {company}" if company else "",
        f"Location: {place}" if place else "",
        f"Remote: {job.get('jobLocationType')}" if job.get("jobLocationType") else "",
        f"Employment type: {job.get('employmentType')}" if job.get("employmentType") else "",
        f"Salary: {json.dumps(job['baseSalary'], ensure_ascii=False)}" if job.get("baseSalary") else "",
        f"Experience: {job.get('experienceRequirements')}" if job.get("experienceRequirements") else "",
        f"Skills: {job.get('skills')}" if job.get("skills") else "",
    ]
    desc = job.get("description") or ""
    parts.append(_html_to_text(desc) if "<" in desc else desc)
    return "\n".join(p for p in parts if p)


def html_to_job_text(html: str) -> str:
    """Prefiere los datos estructurados JobPosting; si no hay, usa el texto visible."""
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        jobs = _jobposting_nodes(data)
        jobs = [j for j in jobs if j.get("description")]
        if jobs:
            return _jobposting_to_text(jobs[0])[:MAX_TEXT_CHARS]
    text = _html_to_text(html)
    return text[:MAX_TEXT_CHARS]


def fetch_job_text(url: str) -> str:
    text = html_to_job_text(fetch_html(url.strip()))
    if len(text) < MIN_TEXT_CHARS:
        raise FetchError(
            "No pude leer el contenido de la oferta en esa página (probablemente se carga "
            "con JavaScript o pide login). Copiá el texto y pegalo en la otra pestaña.",
            422,
        )
    return text
