from pathlib import Path

import yaml


def test_compose_separates_api_and_worker_with_shared_local_object_volume() -> None:
    compose_path = Path(__file__).resolve().parents[1] / "compose.yaml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = compose["services"]

    assert set(services) == {"rabbitmq", "api", "worker"}
    assert services["api"]["image"] == services["worker"]["image"]
    assert services["api"]["build"] == services["worker"]["build"]
    assert services["api"]["command"] == ["python", "start.py"]
    assert services["worker"]["command"] == [
        "celery",
        "-A",
        "backend.app.celery_app",
        "worker",
        "--loglevel=INFO",
        "--concurrency=1",
    ]
    assert services["api"]["volumes"] == services["worker"]["volumes"]
    assert services["api"]["volumes"] == [
        "document-objects:/var/lib/idqe/object-storage"
    ]
    assert services["api"]["environment"]["OBJECT_STORAGE_LOCAL_ROOT"] == (
        "/var/lib/idqe/object-storage"
    )
    assert services["worker"]["environment"]["OBJECT_STORAGE_LOCAL_ROOT"] == (
        "/var/lib/idqe/object-storage"
    )
    assert services["api"]["environment"]["CELERY_BROKER_URL"] == (
        services["worker"]["environment"]["CELERY_BROKER_URL"]
    )
