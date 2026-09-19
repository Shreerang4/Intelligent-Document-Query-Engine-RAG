from __future__ import annotations

import signal
import sys
from pathlib import Path

import production_start


class FakeProcess:
    def __init__(self, return_code: int | None = None) -> None:
        self.return_code = return_code
        self.signals: list[int] = []
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return self.return_code

    def send_signal(self, signal_number: int) -> None:
        self.signals.append(signal_number)
        self.return_code = -signal_number

    def kill(self) -> None:
        self.killed = True
        self.return_code = -signal.SIGKILL

    def wait(self) -> int:
        self.waited = True
        return int(self.return_code or 0)


def test_launcher_commands_use_existing_api_and_celery_configuration() -> None:
    assert production_start.api_command("python-test") == ["python-test", "start.py"]
    assert production_start.worker_command("python-test") == [
        "python-test",
        "-m",
        "celery",
        "-A",
        "backend.app.celery_app",
        "worker",
        "--loglevel=INFO",
        "--concurrency=1",
    ]
    assert "--broker" not in production_start.worker_command()


def test_unexpected_api_exit_terminates_worker_and_returns_failure() -> None:
    api = FakeProcess(return_code=0)
    worker = FakeProcess()
    commands: list[list[str]] = []
    processes = iter((api, worker))

    result = production_start.run_supervisor(
        popen=lambda command: commands.append(command) or next(processes),
        sleep=lambda _seconds: None,
        install_signal_handlers=False,
    )

    assert commands == [
        [sys.executable, "start.py"],
        production_start.worker_command(),
    ]
    assert result == 1
    assert worker.signals == [signal.SIGTERM]
    assert worker.waited is True


def test_sigterm_is_forwarded_to_both_children(monkeypatch) -> None:
    api = FakeProcess()
    worker = FakeProcess()
    processes = iter((api, worker))
    handlers = {}

    monkeypatch.setattr(production_start.signal, "getsignal", lambda _number: signal.SIG_DFL)
    monkeypatch.setattr(
        production_start.signal,
        "signal",
        lambda number, handler: handlers.__setitem__(number, handler),
    )

    sent = False

    def request_shutdown(_seconds: float) -> None:
        nonlocal sent
        if not sent:
            sent = True
            handlers[signal.SIGTERM](signal.SIGTERM, None)

    result = production_start.run_supervisor(
        popen=lambda _command: next(processes),
        sleep=request_shutdown,
    )

    assert result == 128 + signal.SIGTERM
    assert api.signals == worker.signals == [signal.SIGTERM]
    assert api.waited is worker.waited is True


def test_docker_default_uses_launcher_while_compose_keeps_split_commands() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    compose = (root / "compose.yaml").read_text(encoding="utf-8")

    assert 'CMD ["python", "production_start.py"]' in dockerfile
    assert 'command: ["python", "start.py"]' in compose
    assert "- --concurrency=1" in compose
