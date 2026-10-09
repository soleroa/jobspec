from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# Valores de relleno que un modelo puede escribir para "llenar" un campo obligatorio.
PLACEHOLDERS = {
    "desconocida", "desconocido", "no especificada", "no especificado", "no indicada",
    "no indicado", "no informada", "no informado", "sin especificar", "confidencial",
    "n/a", "na", "none", "null", "unknown", "not specified", "unspecified", "tbd", "-", "?",
}


class WorkMode(str, Enum):
    remote = "remote"
    hybrid = "hybrid"
    onsite = "onsite"


class Seniority(str, Enum):
    intern = "intern"
    junior = "junior"
    semi_senior = "semi_senior"
    senior = "senior"
    lead = "lead"


class SalaryPeriod(str, Enum):
    hour = "hour"
    month = "month"
    year = "year"


class Salary(BaseModel):
    min: Optional[float] = Field(None, ge=0, description="Salario mínimo, solo el número")
    max: Optional[float] = Field(None, ge=0, description="Salario máximo, solo el número")
    currency: Optional[str] = Field(None, description="Código ISO 4217, ej: USD, ARS, EUR")
    period: Optional[SalaryPeriod] = Field(None, description="Período al que corresponde el salario")

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().upper()
        if len(v) != 3 or not v.isalpha():
            raise ValueError("currency debe ser un código ISO de 3 letras")
        return v

    @model_validator(mode="after")
    def check_range(self) -> "Salary":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("salary.min no puede ser mayor que salary.max")
        return self


class Location(BaseModel):
    city: Optional[str] = None
    country: Optional[str] = None


class JobOffer(BaseModel):
    """Oferta de trabajo extraída de texto libre.

    Obligatorios: title y company. Todo lo demás es opcional: si el texto no
    lo menciona, el modelo debe devolver null (o lista vacía), nunca inventarlo.
    """

    title: str = Field(..., min_length=1, description="Título del puesto")
    company: str = Field(..., min_length=1, description="Nombre de la empresa")
    seniority: Optional[Seniority] = None
    work_mode: Optional[WorkMode] = None
    location: Optional[Location] = None
    salary: Optional[Salary] = None
    required_skills: List[str] = Field(
        default_factory=list, description="Skills o tecnologías obligatorias"
    )
    nice_to_have_skills: List[str] = Field(
        default_factory=list, description="Skills o tecnologías deseables, no obligatorias"
    )
    years_experience: Optional[int] = Field(
        None, ge=0, le=50, description="Años de experiencia mínimos pedidos"
    )

    @field_validator("title", "company")
    @classmethod
    def strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("no puede estar vacío")
        if v.lower().strip(".") in PLACEHOLDERS:
            raise ValueError(f"'{v}' es un valor de relleno: el dato no aparece en el texto")
        return v

    @field_validator("required_skills", "nice_to_have_skills")
    @classmethod
    def clean_skills(cls, v: List[str]) -> List[str]:
        # Saca espacios, vacíos y duplicados (sin distinguir mayúsculas), conservando el orden.
        seen = set()
        out = []
        for skill in v:
            s = skill.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out
