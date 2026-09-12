"""Repeatable live validation through the daemon's public HTTP interface.

The runner owns a fresh scratch directory and its child processes. Reports,
credentials, indexes, and logs stay together outside Git; no live corpus is used.
"""

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time

import httpx

from mainframe import __version__

DOCUMENTS = {
    "recovery.md": "# Model recovery\n\nA device failure releases the old model before a replacement loads. "
                   "The daemon retries once and records repeated failures for the operator.\n",
    "ownership.md": "# Index ownership\n\nOne writer owns the index. Shutdown waits for active writes. "
                    "A restart reuses unchanged documents without embedding them again.\n",
}
QUERY = "How does model recovery release the old model before retrying?"
RESCAN_SECONDS = 180
CHANGE_SECONDS = 300


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def wait_for(check, timeout, *, alive=lambda: True):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require(alive(), "validation daemon exited before completing the check")
        result = check()
        if result:
            return result
        time.sleep(0.1)
    raise TimeoutError("validation condition did not complete within its timeout")


@contextlib.contextmanager
def daemon_session(config, log_path, timeout):
    """Own one child; always stop it, including when a validation check fails."""
    from filelock import FileLock
    from mainframe.service.discovery import read_discovery

    env = {k: v for k, v in os.environ.items() if not k.startswith("MAINFRAME_")}
    env.update(MAINFRAME_CONFIG=str(log_path.parent / "config.json"), PYTHONUNBUFFERED="1",
               PYTHONIOENCODING="utf-8")
    client = None
    owned = False
    failed = False
    nonce = secrets.token_hex(16)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "mainframe.adapters.validation", nonce],
                                   cwd=log_path.parent, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            def ready():
                nonlocal client, owned
                info = read_discovery(config)
                # Windows venv launchers can run Python in a second process.
                # Only this launch knows the nonce; Popen.pid need not match.
                if not info or info.get("nonce") != nonce:
                    return False
                if client is None:
                    client = httpx.Client(base_url=f"http://127.0.0.1:{info['port']}",
                                          headers={"Authorization": "Bearer " + config["service"]["token"]},
                                          timeout=min(timeout, 600), trust_env=False)
                try:
                    response = client.get("/healthz", timeout=3)
                    owned = response.status_code == 200 and response.json().get("nonce") == nonce
                    return owned
                except (httpx.HTTPError, ValueError):
                    return False

            wait_for(ready, min(timeout, 120), alive=lambda: process.poll() is None)
            yield client
        except BaseException:
            failed = True
            raise
        finally:
            try:
                if process.poll() is None:
                    require(owned, "daemon did not prove ownership of its endpoint")
                    response = client.post("/shutdown", timeout=10)
                    response.raise_for_status()
                    require(response.json()["stopping"], "daemon did not accept shutdown")
                    process.wait(timeout=float(config["service"]["shutdown_timeout_seconds"]) + 10)
                require(process.returncode == 0, "validation daemon exited unsuccessfully")
                state = Path(config["paths"]["mainframe_dir"])
                require(not (state / "daemon.json").exists(), "shutdown left a discovery record")
                with FileLock(str(state / ".daemon.lock")).acquire(timeout=0):
                    pass
            except Exception:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                if not failed:
                    raise
            finally:
                if client is not None:
                    client.close()


class ValidationRun:
    def __init__(self, output, config, duration=0, timeout=2400, session_factory=daemon_session):
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and nonnegative")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.output = Path(output).expanduser().resolve()
        self.state = self.output / "state"
        self.project = self.output / "repos" / "validation"
        self.config = json.loads(json.dumps(config))
        self.duration, self.timeout = duration, timeout
        self.session_factory = session_factory
        self.client = None
        self.health_times, self.health_errors = [], []
        self.report = {"schema_version": 1, "status": "running", "version": __version__,
                       "python": sys.version.split()[0], "started_at": time.time(),
                       "duration_requested_seconds": duration, "checks": [], "searches": [],
                       "timings": {}, "snapshots": [], "mutation_cycles": 0}

    def save(self):
        samples = sorted(self.health_times)
        self.report["health"] = {"requests": len(samples), "failures": list(self.health_errors),
                                 "max_seconds": max(samples, default=0),
                                 "p95_seconds": samples[math.ceil(len(samples) * 0.95) - 1] if samples else 0}
        pending = self.output / "report.json.tmp"
        pending.write_text(json.dumps(self.report, indent=2), encoding="utf-8")
        pending.replace(self.output / "report.json")

    def mark(self, name, **details):
        entry = {"name": name, "at": time.time(), **details}
        self.report["checks"].append(entry)
        self.save()
        print(json.dumps(entry), flush=True)

    def prepare(self):
        if any((p / ".git").exists() for p in (self.output, *self.output.parents)):
            raise ValueError("validation output must be outside Git worktrees")
        self.output.mkdir(parents=True, exist_ok=False)
        (self.project / "docs").mkdir(parents=True)
        for name, body in DOCUMENTS.items():
            (self.project / "docs" / name).write_text(body, encoding="utf-8")
        self.hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (self.project / "docs").iterdir()}
        (self.state / ".lancedb").mkdir(parents=True)
        (self.state / ".lancedb" / "sentinel").write_text("v1 isolation", encoding="utf-8")
        self.config["paths"] = {"mainframe_dir": str(self.state), "repos_dir": str(self.project.parent),
                                "model_cache": self.config["paths"]["model_cache"],
                                "include_projects": ["validation"], "exclude_projects": []}
        self.config["service"].update(port=0, token=secrets.token_urlsafe(32))
        self.config["index"]["rescan_hours"] = RESCAN_SECONDS / 3600
        for name in ("contextual", "consolidator", "nli"):
            self.config[name]["enabled"] = False
        (self.output / "config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        self.report["models"] = {k: self.config[k]["model"] for k in ("embedder", "reranker")}
        self.mark("isolated synthetic corpus prepared")

    @contextlib.contextmanager
    def session(self, label):
        started = time.monotonic()
        with self.session_factory(self.config, self.output / f"{label}.log", self.timeout) as client:
            self.client = client
            self.report["timings"][label + "_ready"] = time.monotonic() - started
            stop = threading.Event()

            def probe():
                while True:
                    before = time.monotonic()
                    try:
                        client.get("/healthz", timeout=3).raise_for_status()
                        self.health_times.append(time.monotonic() - before)
                    except Exception as exc:
                        self.health_errors.append(str(exc))
                    if stop.wait(2):
                        return

            thread = threading.Thread(target=probe, name="validation-health", daemon=True)
            thread.start()
            try:
                yield
            finally:
                stop.set()
                thread.join(5)
                require(not thread.is_alive(), "health probe did not stop")
        self.mark(label + " shutdown")

    def request(self, method, path, **kwargs):
        response = self.client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def indexed(self):
        status = self.request("GET", "/status")
        require(not (any(m["state"] == "failed" for m in status["models"].values()) or
                     any(e["kind"] == "model.failed" for e in status["recent_events"])),
                "model load failed; see the daemon log and model status")
        require(not any(e["kind"] == "job.failed" for e in status["recent_events"]),
                "daemon recorded a failed job; see the daemon log")
        require(not status["index"]["rebuild_required"], "index settings drifted")
        last = status["index"]["last_run"]
        if last:
            require(not last["failed_count"], "indexing reported failed files")
        return status if last and status["daemon"]["queue_depth"] == 0 else None

    def search(self, query=QUERY, **options):
        started = time.monotonic()
        result = self.request("POST", "/search", json={"query": query, "response_format": "full", **options})
        require(result["results"], "synthetic search returned no results")
        for hit in result["results"]:
            path = Path(hit["file"]).resolve()
            require(path.is_relative_to(self.project) or path.is_relative_to(self.state),
                    "search returned a file outside validation scope")
        self.report["searches"].append({"seconds": time.monotonic() - started, "results": len(result["results"])})
        return result["results"]

    def mutation_cycle(self):
        number = self.report["mutation_cycles"]
        nonce = secrets.token_hex(8)
        path = self.project / "docs" / f"probe-{number}.md"
        before = wait_for(self.indexed, self.timeout)["index"]["docs"]
        body = f"# Receipt {nonce}\n\n" + (f"Receipt {nonce} records a violet diagnostic checkpoint. " * 12)
        path.write_text(body, encoding="utf-8")

        def visible(name, color):
            return any(Path(h["file"]).name == name and color in h["text"] and "line_start" in h
                       for h in self.search(f"{color} diagnostic checkpoint {nonce}"))

        wait_for(lambda: visible(path.name, "violet"), self.timeout)
        path.write_text(body.replace("violet", "amber"), encoding="utf-8")
        wait_for(lambda: visible(path.name, "amber"), self.timeout)
        renamed = path.with_name(f"renamed-{number}.md")
        path.rename(renamed)
        wait_for(lambda: visible(renamed.name, "amber"), self.timeout)
        renamed.unlink()
        wait_for(lambda: (s := self.indexed()) and s["index"]["docs"] == before, self.timeout)
        require(not any(Path(h["file"]).name in (path.name, renamed.name)
                        for h in self.search(f"amber diagnostic checkpoint {nonce}")), "deleted probe is searchable")
        capture = self.request("POST", "/capture", json={"content": body, "title": "Validation receipt",
                                                         "project": "validation"})
        captured = Path(capture["file"]).resolve()
        require(captured.is_relative_to(self.state) and captured.is_file(), "capture escaped validation state")
        wait_for(lambda: any(Path(h["file"]).resolve() == captured
                             for h in self.search(f"violet diagnostic checkpoint {nonce}", include_sessions=True)),
                 self.timeout)
        require(not any(Path(h["file"]).resolve() == captured
                        for h in self.search(f"violet diagnostic checkpoint {nonce}")), "capture leaked into default search")
        self.report["mutation_cycles"] += 1
        self.mark("watcher and capture cycle", cycle=number + 1)

    def observe(self, deadline, next_change):
        while time.monotonic() < deadline:
            started = time.monotonic()
            self.search()
            if time.monotonic() < deadline and time.monotonic() >= next_change:
                self.mutation_cycle()
                next_change = time.monotonic() + CHANGE_SECONDS
            status = wait_for(self.indexed, self.timeout)
            self.report["snapshots"].append({"at": time.time(), "rows": status["index"]["rows"],
                                              "queue_depth": status["daemon"]["queue_depth"], "vram": status["vram"]})
            self.mark("observation", searches=len(self.report["searches"]), health_failures=len(self.health_errors))
            time.sleep(max(0, min(deadline - time.monotonic(), 30 - (time.monotonic() - started))))
        return next_change

    def run(self):
        self.prepare()  # Refusal must never overwrite an existing report.
        try:
            with self.session("startup"):
                started = time.monotonic()
                wait_for(self.indexed, self.timeout)
                self.report["timings"]["initial_index"] = time.monotonic() - started
                require(self.client.get("/healthz", headers={"Authorization": ""}).status_code == 401,
                        "authentication boundary failed")
                for path in ("/capture", "/jobs/rescan", "/shutdown"):
                    require(self.client.post(path, headers={"Origin": "https://example.com"}, content="{}").status_code == 403,
                            "browser origin boundary failed")
                self.mark("authentication and origin boundaries")
                for label in ("first_search", "warm_search"):
                    self.search()
                    self.report["timings"][label] = self.report["searches"][-1]["seconds"]
                start = time.monotonic()
                self.mutation_cycle()
                next_change = self.observe(start + self.duration / 2, start + CHANGE_SECONDS)
            with self.session("restart"):
                state = wait_for(self.indexed, self.timeout)
                upserted = state["index"]["last_run"]["upserted_rows"]
                require(upserted == 0, "restart re-embedded unchanged documents")
                self.report["restart_upserted_rows"] = upserted
                self.mark("restart reused unchanged documents")
                self.search()
                self.observe(start + self.duration, next_change)
                wait_for(self.indexed, self.timeout)
            self.report["observed_seconds"] = time.monotonic() - start
            from mainframe.service.events import EventStore
            events = EventStore(self.state / "events.db")
            self.report["failed_jobs"] = events.count("job.failed")
            self.report["rescan_count"] = events.count("index.rescan")
            bound = 2 + math.ceil((time.time() - self.report["started_at"]) / RESCAN_SECONDS) + 6 * self.report["mutation_cycles"]
            require(self.report["rescan_count"] <= bound, "rescan count exceeded the validation bound")
            require(self.report["failed_jobs"] == 0, "daemon recorded failed jobs")
            for name, digest in self.hashes.items():
                require(hashlib.sha256((self.project / "docs" / name).read_bytes()).hexdigest() == digest,
                        "validation changed an original fixture")
            require((self.state / ".lancedb" / "sentinel").read_text() == "v1 isolation", "v1 state was modified")
            require(not self.health_errors, "health probes failed; see report.json")
            self.report["status"] = "passed"
            self.mark("validation passed")
        except BaseException as exc:
            self.report.update(status="failed", error=str(exc))
            raise
        finally:
            self.report["finished_at"] = time.time()
            self.save()
        return self.report


if __name__ == "__main__":
    from mainframe.config import load_config
    from mainframe.service.daemon import Daemon

    daemon = Daemon(load_config())
    daemon.nonce = sys.argv[1]
    daemon.serve()
