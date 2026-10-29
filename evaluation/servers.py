"""Runtime management: exactly one model in memory at a time.

    with serve(settings):      # settings: run.ModelSettings
        ... send requests ...

Ollama models: every other loaded Ollama model is unloaded first, any llama-server on :8080 is stopped, the tag
must already be pulled (never pulled automatically), and the model is unloaded again afterwards.
llama-server models: all Ollama models are unloaded, a fresh llama-server is started (log: runs/llama_server.log),
and it is stopped afterwards, also on Ctrl-C or an exception.
"""

import contextlib
import os
import shlex
import signal
import subprocess
import time

import httpx

from . import ROOT

OLLAMA = "http://127.0.0.1:11434"
LLAMA_HOST, LLAMA_PORT = "127.0.0.1", 8080
LLAMA_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
LLAMA_LOG = ROOT / "runs" / "llama_server.log"
HEALTH_TIMEOUT = 300  # seconds

SARVAM_ALIAS = "sarvam-30b-q3"
SARVAM_CMD = [
    "llama-server", "-m", "models/sarvam-30b/sarvam-30b-Q3_K_M.gguf", "--alias", SARVAM_ALIAS,
    "-ngl", "99", "-c", "4096", "--jinja", "--reasoning-budget", "0",
    "--host", LLAMA_HOST, "--port", str(LLAMA_PORT),
]


class ServerError(RuntimeError):
    pass


# ---------------------------------------------------------------- Ollama

def ollama_loaded() -> list[str]:
    """Names of models currently in memory. Empty if Ollama is not running."""
    try:
        r = httpx.get(f"{OLLAMA}/api/ps", timeout=10)
        r.raise_for_status()
    except httpx.HTTPError:
        return []
    return [m.get("name") or m["model"] for m in r.json().get("models", [])]


def ollama_unload(names: list[str] | None = None) -> None:
    for name in ollama_loaded() if names is None else names:
        httpx.post(f"{OLLAMA}/api/generate", json={"model": name, "keep_alive": 0}, timeout=60)
    for _ in range(30):  # unloading is asynchronous
        if not [n for n in ollama_loaded() if names is None or n in names]:
            return
        time.sleep(1)
    raise ServerError("Ollama did not unload its models within 30 s")


def ollama_pulled(tag: str) -> bool:
    try:
        r = httpx.get(f"{OLLAMA}/api/tags", timeout=10)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise ServerError(f"Ollama is not reachable at {OLLAMA} ({e}). Start it with `ollama serve`.") from e
    have = {m["name"] for m in r.json().get("models", [])}
    return tag in have or (":" not in tag and f"{tag}:latest" in have)


# ---------------------------------------------------------------- llama-server

def _listener_pids(port: int) -> list[int]:
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def _command(pid: int) -> str:
    return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def llama_stop_foreign() -> None:
    """Stop any llama-server listening on :8080. Refuses to touch other programs."""
    for pid in _listener_pids(LLAMA_PORT):
        cmd = _command(pid)
        if "llama-server" not in cmd:
            raise ServerError(f"port {LLAMA_PORT} is used by another program (pid {pid}: {cmd[:80]})")
        _terminate_pid(pid)


def _terminate_pid(pid: int, grace: float = 30) -> None:
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10)):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, sig)
        deadline = time.time() + wait
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.5)
    raise ServerError(f"could not stop pid {pid}")


def _reusable_llama(alias: str) -> bool:
    """A llama-server already on :8080 is kept only if it has our alias and thinking disabled."""
    for pid in _listener_pids(LLAMA_PORT):
        cmd = _command(pid)
        if "llama-server" in cmd and f"--alias {alias}" in cmd and "--reasoning-budget 0" in cmd:
            return True
    return False


def wait_healthy(proc: subprocess.Popen | None, timeout: float = HEALTH_TIMEOUT) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise ServerError(f"llama-server exited with code {proc.returncode}; see {LLAMA_LOG}")
        try:
            if httpx.get(f"{LLAMA_URL}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise ServerError(f"llama-server not healthy after {timeout:.0f} s; see {LLAMA_LOG}")


def start_llama(cmd: list[str] = SARVAM_CMD) -> subprocess.Popen:
    LLAMA_LOG.parent.mkdir(exist_ok=True)
    log = open(LLAMA_LOG, "ab")
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(cmd)}\n".encode())
    log.flush()
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def stop_llama(proc: subprocess.Popen, grace: float = 30) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------- one model at a time

@contextlib.contextmanager
def serve(settings):
    """Make `settings` (run.ModelSettings) the only model in memory; clean up on exit."""
    if settings.runtime == "ollama":
        if not ollama_pulled(settings.tag):
            raise ServerError(f"Ollama tag {settings.tag!r} is not pulled. Run `ollama pull {settings.tag}` "
                              f"first (the harness never pulls automatically).")
        llama_stop_foreign()
        ollama_unload()
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                ollama_unload()
    elif settings.runtime == "llama-server":
        ollama_unload()
        proc = None
        if not _reusable_llama(settings.tag):
            llama_stop_foreign()
            if not (ROOT / SARVAM_CMD[2]).exists():
                raise ServerError(f"model file missing: {SARVAM_CMD[2]}")
            proc = start_llama()
        try:
            wait_healthy(proc)
            yield
        finally:
            if proc is not None:
                stop_llama(proc)
    else:
        raise ServerError(f"unknown runtime {settings.runtime!r}")
