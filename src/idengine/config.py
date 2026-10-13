"""Configuration management.

Settings are resolved in three layers, later layers overriding earlier ones:

1. Built-in defaults (the pydantic models below).
2. An optional TOML config file (``--config`` flag or ``IDE_CONFIG_FILE``).
3. Environment variables named ``IDE_<SECTION>__<FIELD>`` (for example
   ``IDE_SEARCH__RRF_K=30``). A handful of common settings also have short
   aliases such as ``IDE_DATA_DIR`` and ``IDE_API_KEYS``.

All values are validated by pydantic; invalid configuration raises
:class:`~idengine.errors.ConfigurationError` with a readable message.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from idengine.errors import ConfigurationError

ENV_PREFIX = "IDE_"

# Short aliases for the most frequently overridden settings.
ENV_ALIASES: dict[str, tuple[str, str]] = {
    "IDE_DATA_DIR": ("storage", "data_dir"),
    "IDE_API_KEYS": ("api", "api_keys"),
    "IDE_AUTH_ENABLED": ("api", "auth_enabled"),
    "IDE_LOG_LEVEL": ("logging", "level"),
    "IDE_LLM_PROVIDER": ("llm", "provider"),
    "IDE_LLM_BASE_URL": ("llm", "base_url"),
    "IDE_LLM_MODEL": ("llm", "model"),
    "IDE_LLM_API_KEY": ("llm", "api_key"),
}


class Role(StrEnum):
    """API roles, ordered by privilege."""

    READER = "reader"
    WRITER = "writer"
    ADMIN = "admin"

    @property
    def rank(self) -> int:
        return {"reader": 0, "writer": 1, "admin": 2}[self.value]

    def allows(self, required: Role) -> bool:
        return self.rank >= required.rank


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StorageSettings(_Section):
    data_dir: Path = Path("var")

    @property
    def database_path(self) -> Path:
        return self.data_dir / "engine.sqlite3"

    @property
    def blob_dir(self) -> Path:
        return self.data_dir / "blobs"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"


class IngestSettings(_Section):
    max_file_bytes: int = Field(default=20 * 1024 * 1024, ge=1, le=512 * 1024 * 1024)
    allowed_types: list[str] = Field(
        default_factory=lambda: ["pdf", "docx", "txt", "markdown", "html"]
    )
    max_pdf_pages: int = Field(default=500, ge=1)
    max_docx_uncompressed_bytes: int = Field(default=200 * 1024 * 1024, ge=1024)
    max_docx_compression_ratio: float = Field(default=150.0, gt=1.0)
    near_duplicate_distance: int = Field(default=3, ge=0, le=64)
    auto_classify: bool = True

    @field_validator("allowed_types")
    @classmethod
    def _known_types(cls, value: list[str]) -> list[str]:
        known = {"pdf", "docx", "txt", "markdown", "html"}
        cleaned = [v.strip().lower() for v in value if v.strip()]
        unknown = sorted(set(cleaned) - known)
        if unknown:
            raise ValueError(f"unknown document types: {', '.join(unknown)}")
        if not cleaned:
            raise ValueError("at least one document type must be allowed")
        return cleaned


class ChunkingSettings(_Section):
    max_tokens: int = Field(default=200, ge=16, le=4096)
    overlap_tokens: int = Field(default=40, ge=0)
    min_tokens: int = Field(default=12, ge=1)
    respect_headings: bool = True

    @model_validator(mode="after")
    def _check_overlap(self) -> ChunkingSettings:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must not exceed max_tokens")
        return self


class EmbeddingSettings(_Section):
    provider: Literal["hashing", "lsa", "sentence-transformers"] = "hashing"
    dim: int = Field(default=256, ge=8, le=4096)
    seed: int = 13
    bigrams: bool = True
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class IndexSettings(_Section):
    kind: Literal["flat", "ivf", "hnsw"] = "hnsw"
    ivf_nlist: int = Field(default=16, ge=1, le=65536)
    ivf_nprobe: int = Field(default=4, ge=1)
    ivf_train_threshold: int = Field(default=256, ge=2)
    hnsw_m: int = Field(default=16, ge=2, le=128)
    hnsw_ef_construction: int = Field(default=100, ge=4)
    hnsw_ef_search: int = Field(default=64, ge=1)

    @model_validator(mode="after")
    def _check_probe(self) -> IndexSettings:
        if self.ivf_nprobe > self.ivf_nlist:
            raise ValueError("ivf_nprobe must not exceed ivf_nlist")
        return self


class SearchSettings(_Section):
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    rrf_k: int = Field(default=60, ge=1)
    default_top_k: int = Field(default=10, ge=1)
    max_top_k: int = Field(default=100, ge=1, le=1000)
    candidate_multiplier: int = Field(default=4, ge=1, le=50)
    max_query_chars: int = Field(default=2000, ge=1)


class RagSettings(_Section):
    retrieve_k: int = Field(default=20, ge=1, le=200)
    rerank_k: int = Field(default=5, ge=1, le=50)
    context_token_budget: int = Field(default=900, ge=50)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    min_relevance: float = Field(default=0.05, ge=0.0, le=1.0)


class LLMSettings(_Section):
    provider: Literal["extractive", "ollama", "openai"] = "extractive"
    base_url: str | None = None
    model: str = "llama3.1"
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=10)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=1, le=32768)
    answer_sentences: int = Field(default=3, ge=1, le=20)

    @model_validator(mode="after")
    def _check_remote(self) -> LLMSettings:
        if self.provider != "extractive" and not self.base_url:
            raise ValueError(f"llm.base_url is required when llm.provider is '{self.provider}'")
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ValueError("llm.base_url must start with http:// or https://")
        return self


class JobSettings(_Section):
    workers: int = Field(default=2, ge=0, le=64)
    max_attempts: int = Field(default=3, ge=1, le=20)
    backoff_base_seconds: float = Field(default=1.0, ge=0.0)
    backoff_max_seconds: float = Field(default=60.0, ge=0.0)
    poll_interval_seconds: float = Field(default=0.5, gt=0.0)
    lease_seconds: float = Field(default=300.0, gt=0.0)


class ClassifierSettings(_Section):
    algorithm: Literal["naive_bayes", "logistic"] = "naive_bayes"
    alpha: float = Field(default=0.5, gt=0.0)
    l2: float = Field(default=1e-3, ge=0.0)
    epochs: int = Field(default=300, ge=1, le=100_000)
    learning_rate: float = Field(default=0.5, gt=0.0)
    max_features: int = Field(default=20_000, ge=10)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ApiSettings(_Section):
    auth_enabled: bool = True
    api_keys: list[str] = Field(default_factory=list)
    max_request_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)
    rate_limit_per_minute: int = Field(default=600, ge=0)
    rate_limit_burst: int = Field(default=60, ge=1)
    metrics_public: bool = False

    @field_validator("api_keys")
    @classmethod
    def _parse_keys(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for raw in value:
            raw = raw.strip()
            if not raw:
                continue
            key, _, role = raw.partition(":")
            if len(key) < 8:
                raise ValueError("API keys must be at least 8 characters long")
            if role and role not in {r.value for r in Role}:
                raise ValueError(f"unknown role '{role}' (expected reader, writer or admin)")
            cleaned.append(raw)
        return cleaned

    def key_roles(self) -> dict[str, Role]:
        """Map each configured key to its role (default: admin)."""
        roles: dict[str, Role] = {}
        for raw in self.api_keys:
            key, _, role = raw.partition(":")
            roles[key] = Role(role) if role else Role.ADMIN
        return roles


class LoggingSettings(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = True

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value


class Settings(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    storage: StorageSettings = Field(default_factory=StorageSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    jobs: JobSettings = Field(default_factory=JobSettings)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    def redacted(self) -> dict[str, Any]:
        """Return a JSON-safe dump with secrets masked (for diagnostics)."""
        data = self.model_dump(mode="json")
        data["api"]["api_keys"] = [
            f"***:{k.partition(':')[2] or 'admin'}" for k in self.api.api_keys
        ]
        if data["llm"].get("api_key"):
            data["llm"]["api_key"] = "***"
        return data


def _section_fields() -> dict[str, type[BaseModel]]:
    sections: dict[str, type[BaseModel]] = {}
    for name, field in Settings.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            sections[name] = annotation
    return sections


def _coerce_env_value(section: type[BaseModel], field_name: str, raw: str) -> Any:
    field = section.model_fields.get(field_name)
    if field is None:
        return raw
    origin = getattr(field.annotation, "__origin__", None)
    if origin is list:
        return [part for part in (p.strip() for p in raw.split(",")) if part]
    return raw


def env_overrides(environ: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Translate ``IDE_*`` environment variables into a nested override dict."""
    sections = _section_fields()
    overrides: dict[str, dict[str, Any]] = {}

    def put(section_name: str, field_name: str, raw: str, source: str) -> None:
        section = sections.get(section_name)
        if section is None:
            raise ConfigurationError(f"{source}: unknown configuration section '{section_name}'")
        if field_name not in section.model_fields:
            raise ConfigurationError(
                f"{source}: unknown setting '{field_name}' in section '{section_name}'"
            )
        overrides.setdefault(section_name, {})[field_name] = _coerce_env_value(
            section, field_name, raw
        )

    for alias, (section_name, field_name) in ENV_ALIASES.items():
        if alias in environ:
            put(section_name, field_name, environ[alias], alias)

    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX) or "__" not in key:
            continue
        section_name, _, field_name = key[len(ENV_PREFIX) :].partition("__")
        put(section_name.lower(), field_name.lower(), raw, key)
    return overrides


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(
    config_file: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Build validated :class:`Settings` from file, environment and explicit overrides."""
    env = dict(os.environ if environ is None else environ)
    data: dict[str, Any] = {}

    path = config_file or env.get("IDE_CONFIG_FILE")
    if path:
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigurationError(f"config file not found: {config_path}")
        try:
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigurationError(f"invalid TOML in {config_path}: {exc}") from exc

    data = _deep_merge(data, env_overrides(env))
    if overrides:
        data = _deep_merge(data, overrides)

    try:
        return Settings.model_validate(data)
    except PydanticValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigurationError(f"invalid configuration: {problems}") from exc
