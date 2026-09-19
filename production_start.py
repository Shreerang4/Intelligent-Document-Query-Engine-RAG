#!/usr/bin/env python3
"""Run the API and one Celery worker in a single production container."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any


POLL_INTERVAL_SECONDS = 0.2
SHUTDOWN_TIMEOUT_SECONDS = 10.0


def api_command(python_executable: str = sys.executable) -> list[str]:
    return [python_executable, "start.py"]


def worker_command(python_executable: str = sys.executable) -> list[str]:
    return [
        python_executable,
        "-m",
        "celery",
        "-A",
        "backend.app.celery_app",
        "worker",
        "--loglevel=INFO",
        "--concurrency=1",
    ]


def _signal_running(processes: Sequence[Any], signal_number: int) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal_number)


def _wait_then_kill(
    processes: Sequence[Any],
    *,
    timeout: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    deadline = monotonic() + timeout
    while any(process.poll() is None for process in processes) and monotonic() < deadline:
        sleep(POLL_INTERVAL_SECONDS)
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        process.wait()


def run_supervisor(
    *,
    popen: Callable[[list[str]], Any] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    install_signal_handlers: bool = True,
    shutdown_timeout: float = SHUTDOWN_TIMEOUT_SECONDS,
) -> int:
    """Run both services until signalled or until either service fails."""
    processes: list[Any] = []
    requested_signal: int | None = None
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signal_number: int, _frame: Any) -> None:
        nonlocal requested_signal
        if requested_signal is None:
            requested_signal = signal_number
            _signal_running(processes, signal_number)

    try:
        if install_signal_handlers:
            for signal_number in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, handle_signal)

        for command in (api_command(), worker_command()):
            processes.append(popen(command))

        while True:
            if requested_signal is not None:
                _wait_then_kill(
                    processes,
                    timeout=shutdown_timeout,
                    monotonic=monotonic,
                    sleep=sleep,
                )
                return 128 + requested_signal

            for process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    remaining = [candidate for candidate in processes if candidate is not process]
                    _signal_running(remaining, signal.SIGTERM)
                    _wait_then_kill(
                        remaining,
                        timeout=shutdown_timeout,
                        monotonic=monotonic,
                        sleep=sleep,
                    )
                    return exit_code if exit_code > 0 else 1
            sleep(POLL_INTERVAL_SECONDS)
    except BaseException:
        _signal_running(processes, signal.SIGTERM)
        _wait_then_kill(
            processes,
            timeout=shutdown_timeout,
            monotonic=monotonic,
            sleep=sleep,
        )
        raise
    finally:
        for signal_number, previous_handler in previous_handlers.items():
            signal.signal(signal_number, previous_handler)


if __name__ == "__main__":
    raise SystemExit(run_supervisor())
