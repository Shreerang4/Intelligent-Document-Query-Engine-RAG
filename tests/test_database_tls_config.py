from __future__ import annotations

from pathlib import Path

import pytest

import persistence.db as database_config


def _configure_ca(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    ca_path = tmp_path / "local-mysql-ca.pem"
    ca_path.write_text("test-only-ca", encoding="utf-8")
    monkeypatch.setattr(database_config, "DB_CA_CERT", str(ca_path))
    return ca_path


def test_mysql_tls_verifies_hostname_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ca_path = _configure_ca(monkeypatch, tmp_path)
    monkeypatch.delenv(database_config.LOCAL_CERT_HOSTNAME_OVERRIDE_ENV, raising=False)

    assert database_config._build_connect_args(
        "mysql+pymysql://user:password@127.0.0.1/idqe_e2e_test"
    ) == {"ssl": {"ca": str(ca_path)}}


def test_local_disposable_mysql_can_disable_only_hostname_matching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ca_path = _configure_ca(monkeypatch, tmp_path)
    monkeypatch.setenv(database_config.LOCAL_CERT_HOSTNAME_OVERRIDE_ENV, "true")

    assert database_config._build_connect_args(
        "mysql+pymysql://user:password@127.0.0.1/idqe_e2e_test"
    ) == {"ssl": {"ca": str(ca_path), "check_hostname": False}}


@pytest.mark.parametrize(
    "database_url",
    [
        "mysql+pymysql://user:password@db.example.com/idqe_e2e_test",
        "mysql+pymysql://user:password@127.0.0.1/idqe_production",
    ],
)
def test_hostname_override_rejects_nonlocal_or_nondisposable_mysql(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    database_url: str,
) -> None:
    _configure_ca(monkeypatch, tmp_path)
    monkeypatch.setenv(database_config.LOCAL_CERT_HOSTNAME_OVERRIDE_ENV, "true")

    with pytest.raises(RuntimeError, match="restricted to loopback disposable/test databases"):
        database_config._build_connect_args(database_url)
