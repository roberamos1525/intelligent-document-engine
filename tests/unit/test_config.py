from __future__ import annotations

from pathlib import Path

import pytest

from idengine.config import Role, Settings, env_overrides, load_settings
from idengine.errors import ConfigurationError


def test_defaults_are_valid_and_safe() -> None:
    settings = load_settings(environ={})
    assert settings.api.auth_enabled is True
    assert settings.llm.provider == "extractive"
    assert settings.embeddings.provider == "hashing"
    assert settings.chunking.overlap_tokens < settings.chunking.max_tokens
    assert settings.storage.database_path.name == "engine.sqlite3"


def test_env_overrides_nested_and_aliases() -> None:
    env = {
        "IDE_SEARCH__RRF_K": "30",
        "IDE_INDEX__KIND": "ivf",
        "IDE_DATA_DIR": "/tmp/somewhere",
        "IDE_API_KEYS": "abcdefgh:reader, ijklmnop:admin",
        "IDE_LOG_LEVEL": "debug",
        "UNRELATED": "x",
    }
    settings = load_settings(environ=env)
    assert settings.search.rrf_k == 30
    assert settings.index.kind == "ivf"
    assert settings.storage.data_dir == Path("/tmp/somewhere")
    assert settings.api.key_roles() == {"abcdefgh": Role.READER, "ijklmnop": Role.ADMIN}
    assert settings.logging.level == "DEBUG"


def test_list_values_from_env_are_split() -> None:
    overrides = env_overrides({"IDE_INGEST__ALLOWED_TYPES": "pdf, txt ,"})
    assert overrides == {"ingest": {"allowed_types": ["pdf", "txt"]}}


def test_toml_file_then_env_precedence(tmp_path: Path) -> None:
    config = tmp_path / "engine.toml"
    config.write_text(
        "[chunking]\nmax_tokens = 300\noverlap_tokens = 30\n\n[search]\nrrf_k = 10\n",
        encoding="utf-8",
    )
    settings = load_settings(config, environ={"IDE_SEARCH__RRF_K": "20"})
    assert settings.chunking.max_tokens == 300
    assert settings.search.rrf_k == 20  # env wins over file


def test_config_file_from_environment(tmp_path: Path) -> None:
    config = tmp_path / "c.toml"
    config.write_text("[rag]\nrerank_k = 7\n", encoding="utf-8")
    assert load_settings(environ={"IDE_CONFIG_FILE": str(config)}).rag.rerank_k == 7


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        ({"IDE_CHUNKING__OVERLAP_TOKENS": "500"}, "overlap_tokens"),
        ({"IDE_NOPE__X": "1"}, "unknown configuration section"),
        ({"IDE_SEARCH__NOPE": "1"}, "unknown setting"),
        ({"IDE_INGEST__ALLOWED_TYPES": "pdf,exe"}, "unknown document types"),
        ({"IDE_LLM_PROVIDER": "ollama"}, "base_url is required"),
        ({"IDE_LLM_PROVIDER": "openai", "IDE_LLM_BASE_URL": "ftp://x"}, "http://"),
        ({"IDE_API_KEYS": "short"}, "at least 8 characters"),
        ({"IDE_API_KEYS": "longenough:superuser"}, "unknown role"),
        ({"IDE_INDEX__IVF_NPROBE": "64", "IDE_INDEX__IVF_NLIST": "8"}, "nprobe"),
        ({"IDE_SEARCH__BM25_B": "2"}, "bm25_b"),
    ],
)
def test_invalid_configuration_is_rejected(env: dict[str, str], fragment: str) -> None:
    with pytest.raises(ConfigurationError) as info:
        load_settings(environ=env)
    assert fragment in info.value.message


def test_missing_or_invalid_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_settings(tmp_path / "missing.toml", environ={})
    bad = tmp_path / "bad.toml"
    bad.write_text("[search\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid TOML"):
        load_settings(bad, environ={})


def test_redacted_hides_secrets() -> None:
    settings = load_settings(
        environ={
            "IDE_API_KEYS": "supersecret1:writer",
            "IDE_LLM_PROVIDER": "openai",
            "IDE_LLM_BASE_URL": "http://localhost:9999",
            "IDE_LLM_API_KEY": "sk-test",
        }
    )
    redacted = settings.redacted()
    assert redacted["api"]["api_keys"] == ["***:writer"]
    assert redacted["llm"]["api_key"] == "***"
    assert "supersecret1" not in str(redacted)


def test_role_ordering() -> None:
    assert Role.ADMIN.allows(Role.READER)
    assert Role.WRITER.allows(Role.WRITER)
    assert not Role.READER.allows(Role.WRITER)


def test_settings_reject_unknown_fields() -> None:
    with pytest.raises(ValueError, match="extra"):
        Settings.model_validate({"search": {"bogus": 1}})
