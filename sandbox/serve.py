#!/usr/bin/env python3
"""Base44 sandbox preview console for Niyah.Engine.

Niyah.Engine is a native CLI toolchain: it serves no HTTP surface of its own.
This file exists only so the Base44 sandbox preview (host port 3000) can drive
the real executables built from the working tree and show their actual output.

It performs no engine work itself. Every action shells out to the compiled
`niyah`, `niyah-train` and `niyah_probe` binaries and streams their stdout.
Nothing is written inside the repository: run artifacts go to /tmp.

Only the standard library is used, so no extra dependency is added to the
project's build.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = (ROOT / os.environ.get("NIYAH_BUILD_DIR", "build/base44")).resolve()
TESTS_BUILD_DIR = ROOT / "build" / "base44-tests"
CONSOLE_DIR = Path("/tmp/niyah-console")
RUNS_DIR = CONSOLE_DIR / "runs"
STATE_FILE = CONSOLE_DIR / "state.json"
CORPUS = ROOT / "sandbox" / "corpus.txt"
HELDOUT = ROOT / "sandbox" / "heldout.txt"
INDEX_HTML = ROOT / "sandbox" / "index.html"

BINARIES = ("niyah", "niyah-train", "niyah_probe")
MAX_JOB_SECONDS = 1800
MAX_LOG_LINES = 6000

# Demo shapes. "quick" finishes in seconds; "bigger" trains a wider model.
PRESETS = {
    "quick": {
        "label": "Quick demo",
        "vocab": 512,
        "sequence_length": 64,
        "updates": 150,
        "resume_updates": 50,
        "batch_size": 16,
        "accumulation_steps": 1,
        "context_length": 128,
        "embedding_dim": 64,
        "layers": 2,
        "heads": 2,
        "kv_heads": 1,
        "ffn_hidden_dim": 128,
        "learning_rate": 0.001,
    },
    "bigger": {
        "label": "Wider model",
        "vocab": 512,
        "sequence_length": 64,
        "updates": 300,
        "resume_updates": 50,
        "batch_size": 16,
        "accumulation_steps": 1,
        "context_length": 128,
        "embedding_dim": 128,
        "layers": 3,
        "heads": 4,
        "kv_heads": 2,
        "ffn_hidden_dim": 256,
        "learning_rate": 0.0008,
    },
}

SUGGESTED_PROMPTS = [
    "User: what does a checkpoint do?\nAssistant:",
    "User: is a shard bound to identity?\nAssistant:",
    "User: how is decoding done?\nAssistant:",
]


def child_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{BUILD_DIR}:{env.get('PATH', '')}"
    return env


def first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def probe_toolchain() -> dict:
    info = {
        "root": str(ROOT),
        "build_dir": str(BUILD_DIR),
        "preview_mode": os.environ.get("BASE44_PREVIEW_MODE", ""),
        "binaries": {},
        "toolchain": {},
        "fixtures": {},
    }
    for name in BINARIES:
        path = BUILD_DIR / name
        info["binaries"][name] = {
            "path": str(path),
            "built": path.is_file(),
            "built_at": int(path.stat().st_mtime) if path.is_file() else None,
        }
    for label, args in (
        ("c_compiler", ["cc", "--version"]),
        ("cmake", ["cmake", "--version"]),
        ("git", ["git", "rev-parse", "--short", "HEAD"]),
    ):
        try:
            done = subprocess.run(
                args, cwd=str(ROOT), capture_output=True, text=True, timeout=10
            )
            info["toolchain"][label] = first_line(done.stdout) if done.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            info["toolchain"][label] = ""
    for label, path in (("corpus", CORPUS), ("heldout", HELDOUT)):
        info["fixtures"][label] = {
            "path": str(path),
            "present": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
        }
    return info


ENGINE = probe_toolchain()


class Console:
    """Runs at most one engine job at a time and keeps its streamed log."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.job: dict | None = None
        self.counter = 0
        self.state: dict = {}
        if STATE_FILE.is_file():
            try:
                self.state = json.loads(STATE_FILE.read_text())
            except (OSError, ValueError):
                self.state = {}

    # -- job plumbing ----------------------------------------------------
    def _emit(self, job: dict, text: str, kind: str = "out") -> None:
        for line in (text.splitlines() or [""]):
            job["log"].append({"kind": kind, "text": line})
        if len(job["log"]) > MAX_LOG_LINES:
            del job["log"][: len(job["log"]) - MAX_LOG_LINES]

    def _run(self, job: dict, args: list[str], cwd: Path | None = None) -> tuple[int, str]:
        self._emit(job, "$ " + " ".join(args), "cmd")
        started = time.time()
        try:
            proc = subprocess.Popen(
                args,
                cwd=str(cwd or ROOT),
                env=child_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._emit(job, f"failed to start {args[0]}: {exc}", "err")
            return 127, ""
        lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            self._emit(job, line)
            if time.time() - started > MAX_JOB_SECONDS:
                proc.kill()
                lines.append("")
                lines.append(f"job exceeded {MAX_JOB_SECONDS}s and was stopped")
                self._emit(job, lines[-1], "err")
                break
        proc.wait()
        self._emit(job, f"exit={proc.returncode} in {time.time() - started:.2f}s", "meta")
        return proc.returncode, "\n".join(lines)

    def _step(self, job: dict, title: str) -> float:
        self._emit(job, f"### {title}", "step")
        return time.time()

    def _finish(self, job: dict, started: float, ok: bool) -> None:
        job["exit_code"] = 0 if ok else 1
        job["state"] = "done" if ok else "failed"
        job["finished"] = time.time()
        job["elapsed_ms"] = int((job["finished"] - started) * 1000)

    def start(self, kind: str, params: dict) -> dict:
        with self.lock:
            if self.job is not None and self.job["state"] == "running":
                raise BusyError()
            self.counter += 1
            job = {
                "id": self.counter,
                "kind": kind,
                "params": params,
                "state": "running",
                "exit_code": None,
                "log": [],
                "result": {},
                "started": time.time(),
                "finished": None,
                "elapsed_ms": 0,
            }
            self.job = job
        target = {
            "pipeline": self._pipeline,
            "generate": self._generate,
            "tests": self._tests,
            "build": self._build,
        }[kind]
        threading.Thread(target=self._guard, args=(job, target), daemon=True).start()
        return job

    def _guard(self, job: dict, target) -> None:
        started = job["started"]
        try:
            ok = target(job)
        except Exception as exc:  # noqa: BLE001 - surfaced to the console
            self._emit(job, f"console error: {exc!r}", "err")
            ok = False
        self._finish(job, started, ok)

    # -- engine invocations ---------------------------------------------
    def _pipeline(self, job: dict) -> bool:
        preset_name = job["params"].get("preset", "quick")
        preset = PRESETS[preset_name]
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_dir = RUNS_DIR / time.strftime("run-%Y%m%d-%H%M%S")
        suffix = 1
        while run_dir.exists():
            run_dir = RUNS_DIR / time.strftime(f"run-%Y%m%d-%H%M%S-{suffix}")
            suffix += 1
        run_dir.mkdir(parents=True)

        tokenizer = run_dir / "tokenizer.bin"
        shard = run_dir / "shard.bin"
        total_updates = preset["updates"] + preset["resume_updates"]
        checkpoint = run_dir / f"model-{preset['updates']:04d}.ckpt"
        cursor = run_dir / f"cursor-{preset['updates']:04d}.bin"
        checkpoint_resumed = run_dir / f"model-{total_updates:04d}.ckpt"
        cursor_resumed = run_dir / f"cursor-{total_updates:04d}.bin"
        steps = []
        job["result"] = {"preset": preset_name, "run_dir": str(run_dir), "steps": steps}

        def begin(title: str) -> float:
            return self._step(job, title)

        # 1: corpus -> tokenizer + dataset shard
        t0 = begin("prepare: build the native tokenizer and a supervised dataset shard")
        code, _ = self._run(
            job,
            [
                "niyah", "prepare",
                "--corpus", str(CORPUS),
                "--tokenizer-out", str(tokenizer),
                "--shard-out", str(shard),
                "--target-vocab", str(preset["vocab"]),
                "--min-pair-frequency", "2",
                "--sequence-length", str(preset["sequence_length"]),
                "--record-mode", "blank-line",
                "--response-delimiter", "Assistant:",
            ],
        )
        steps.append({"name": "prepare", "ok": code == 0, "seconds": round(time.time() - t0, 2)})
        if code != 0:
            return False

        # 2: fresh training
        t0 = begin(f"train: {preset['updates']} updates from scratch")
        code, _ = self._run(job, self._train_args(preset, tokenizer, shard, checkpoint, cursor))
        steps.append({"name": "train", "ok": code == 0, "seconds": round(time.time() - t0, 2)})
        if code != 0:
            return False

        # 3: resume from the persisted checkpoint + dataset cursor pair
        t0 = begin(f"resume: {preset['resume_updates']} more updates from the checkpoint")
        code, _ = self._run(
            job,
            [
                "niyah-train", "resume",
                "--tokenizer", str(tokenizer),
                "--shard", str(shard),
                "--checkpoint-in", str(checkpoint),
                "--cursor-in", str(cursor),
                "--checkpoint-out", str(checkpoint_resumed),
                "--cursor-out", str(cursor_resumed),
                "--updates", str(preset["resume_updates"]),
                "--batch-size", str(preset["batch_size"]),
                "--accumulation-steps", str(preset["accumulation_steps"]),
            ],
        )
        steps.append({"name": "resume", "ok": code == 0, "seconds": round(time.time() - t0, 2)})
        if code != 0:
            return False

        # 4: held-out evaluation against the same tokenizer
        t0 = begin("eval: held-out loss, perplexity and the add-1 bigram baseline")
        code, output = self._run(
            job,
            [
                "niyah", "eval",
                "--tokenizer", str(tokenizer),
                "--checkpoint", str(checkpoint_resumed),
                "--heldout", str(HELDOUT),
                "--format", "text",
                "--sequence-length", str(preset["sequence_length"]),
            ],
        )
        steps.append({"name": "eval", "ok": code == 0, "seconds": round(time.time() - t0, 2)})
        if code == 0:
            metrics = {}
            for line in output.splitlines():
                if "=" in line and not line.startswith(" "):
                    key, _, value = line.partition("=")
                    if key in {
                        "mean_loss", "perplexity", "bits_per_token", "bits_per_byte",
                        "baseline_mean_loss", "baseline_bits_per_token",
                        "baseline_bits_per_byte", "sample_count", "token_count",
                    }:
                        metrics[key] = value
            job["result"]["eval"] = metrics
        else:
            return False

        # 5: greedy generation from the resumed checkpoint
        prompt = SUGGESTED_PROMPTS[0]
        t0 = begin("run: autoregressive generation through the KV-cache decode path")
        code, output = self._run(
            job,
            [
                "niyah", "run",
                "--tokenizer", str(tokenizer),
                "--checkpoint", str(checkpoint_resumed),
                "--prompt", prompt,
                "--max-new-tokens", "48",
                "--temperature", "0",
                "--seed", "42",
                "--backend", "cpu",
            ],
        )
        steps.append({"name": "run", "ok": code == 0, "seconds": round(time.time() - t0, 2)})
        if code == 0:
            job["result"]["generated"] = {
                "prompt": prompt,
                "text": output.strip("\n"),
            }
        else:
            return False

        # 6: diagnostic probe
        t0 = begin("probe: shard statistics and next-token logits")
        code, _ = self._run(
            job,
            [
                "niyah_probe",
                "--tokenizer", str(tokenizer),
                "--shard", str(shard),
                "--checkpoint", str(checkpoint_resumed),
                "--prompt", "User: what is scored in eval?",
                "--topk", "5",
                "--trace-steps", "4",
            ],
        )
        steps.append({"name": "probe", "ok": code == 0, "seconds": round(time.time() - t0, 2)})

        self.state = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "tokenizer": str(tokenizer),
            "checkpoint": str(checkpoint_resumed),
            "preset": preset_name,
            "finished_at": time.time(),
        }
        job["result"]["checkpoint"] = str(checkpoint_resumed)
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(json.dumps(self.state, indent=2))
        except OSError as exc:
            self._emit(job, f"could not persist console state: {exc}", "err")
        return True

    @staticmethod
    def _train_args(preset: dict, tokenizer, shard, checkpoint, cursor) -> list[str]:
        return [
            "niyah-train", "new",
            "--tokenizer", str(tokenizer),
            "--shard", str(shard),
            "--checkpoint-out", str(checkpoint),
            "--cursor-out", str(cursor),
            "--updates", str(preset["updates"]),
            "--batch-size", str(preset["batch_size"]),
            "--accumulation-steps", str(preset["accumulation_steps"]),
            "--model-seed", "42",
            "--data-seed", "42",
            "--context-length", str(preset["context_length"]),
            "--embedding-dim", str(preset["embedding_dim"]),
            "--layers", str(preset["layers"]),
            "--heads", str(preset["heads"]),
            "--kv-heads", str(preset["kv_heads"]),
            "--ffn-hidden-dim", str(preset["ffn_hidden_dim"]),
            "--rms-norm-eps", "1e-5",
            "--tie-word-embeddings", "1",
            "--learning-rate", str(preset["learning_rate"]),
            "--beta1", "0.9",
            "--beta2", "0.999",
            "--epsilon", "1e-8",
            "--weight-decay", "0.01",
            "--max-grad-norm", "1.0",
        ]

    def _generate(self, job: dict) -> bool:
        params = job["params"]
        checkpoint = self.state.get("checkpoint")
        tokenizer = self.state.get("tokenizer")
        if not checkpoint or not Path(checkpoint).is_file() or not Path(tokenizer).is_file():
            self._emit(
                job,
                "no trained checkpoint yet - run the end-to-end pipeline first",
                "err",
            )
            return False
        prompt = str(params["prompt"])
        max_new = int(params["max_new_tokens"])
        temperature = float(params["temperature"])
        self._step(job, f"run: generate from {Path(checkpoint).name}")
        code, output = self._run(
            job,
            [
                "niyah", "run",
                "--tokenizer", tokenizer,
                "--checkpoint", checkpoint,
                "--prompt", prompt,
                "--max-new-tokens", str(max_new),
                "--temperature", str(temperature),
                "--seed", "42",
                "--backend", "cpu",
            ],
        )
        if code != 0:
            return False
        job["result"]["prompt"] = prompt
        job["result"]["text"] = output.strip("\n")
        job["result"]["checkpoint"] = checkpoint
        return True

    def _tests(self, job: dict) -> bool:
        t0 = self._step(job, "configure: tests build tree")
        code, _ = self._run(
            job,
            ["cmake", "-S", ".", "-B", str(TESTS_BUILD_DIR),
             "-DCMAKE_BUILD_TYPE=Release", "-DNIYAH_BUILD_TESTS=ON"],
        )
        job["result"].setdefault("steps", []).append(
            {"name": "configure", "ok": code == 0, "seconds": round(time.time() - t0, 2)}
        )
        if code != 0:
            return False

        t0 = self._step(job, "build: compile the regression targets")
        code, _ = self._run(
            job,
            ["cmake", "--build", str(TESTS_BUILD_DIR), "--parallel", "2"],
        )
        job["result"]["steps"].append(
            {"name": "build", "ok": code == 0, "seconds": round(time.time() - t0, 2)}
        )
        if code != 0:
            return False

        t0 = self._step(job, "ctest: repository verification gate")
        code, output = self._run(
            job,
            ["ctest", "--test-dir", str(TESTS_BUILD_DIR), "-C", "Release",
             "--output-on-failure"],
        )
        job["result"]["steps"].append(
            {"name": "ctest", "ok": code == 0, "seconds": round(time.time() - t0, 2)}
        )
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("tests passed") or "tests failed out of" in stripped:
                job["result"]["ctest_summary"] = stripped
        return code == 0

    def _build(self, job: dict) -> bool:
        t0 = self._step(job, "configure: CLI build tree")
        code, _ = self._run(
            job,
            ["cmake", "-S", ".", "-B", str(BUILD_DIR),
             "-DCMAKE_BUILD_TYPE=Release", "-DNIYAH_BUILD_TESTS=OFF"],
        )
        job["result"].setdefault("steps", []).append(
            {"name": "configure", "ok": code == 0, "seconds": round(time.time() - t0, 2)}
        )
        if code != 0:
            return False

        t0 = self._step(job, "build: recompile the tools from the working tree")
        code, _ = self._run(
            job, ["cmake", "--build", str(BUILD_DIR), "--parallel", "2"]
        )
        job["result"]["steps"].append(
            {"name": "build", "ok": code == 0, "seconds": round(time.time() - t0, 2)}
        )
        return code == 0

    # -- views -----------------------------------------------------------
    def snapshot(self) -> dict:
        job = self.job
        if job is not None and job["state"] == "running":
            job = dict(job)
            job["elapsed_ms"] = int((time.time() - job["started"]) * 1000)
        return {
            "job": job,
            "engine": ENGINE,
            "presets": {
                key: {"label": value["label"], "updates": value["updates"],
                      "embedding_dim": value["embedding_dim"], "layers": value["layers"]}
                for key, value in PRESETS.items()
            },
            "suggested_prompts": SUGGESTED_PROMPTS,
            "trained_run": self.state or None,
        }


class BusyError(RuntimeError):
    pass


CONSOLE = Console()


class Handler(BaseHTTPRequestHandler):
    server_version = "niyah-base44-console"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 - quieter access log
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length))
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @property
    def route(self) -> str:
        """Request path without the query string."""
        return self.path.split("?", 1)[0]

    def do_GET(self) -> None:  # noqa: N802
        if self.route in ("/", "/index.html"):
            try:
                body = INDEX_HTML.read_bytes()
            except OSError:
                self._send(500, b"console page missing", "text/plain")
                return
            self._send(200, body, "text/html; charset=utf-8")
            return
        if self.route == "/healthz":
            missing = [n for n in BINARIES if not (BUILD_DIR / n).is_file()]
            if missing:
                self._json(503, {"status": "building", "missing_binaries": missing})
            else:
                self._json(200, {"status": "ok", "binaries": list(BINARIES)})
            return
        if self.route == "/api/state":
            self._json(200, CONSOLE.snapshot())
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_json()
        route = self.route
        try:
            if route == "/api/pipeline":
                preset = payload.get("preset", "quick")
                if preset not in PRESETS:
                    preset = "quick"
                job = CONSOLE.start("pipeline", {"preset": preset})
            elif route == "/api/generate":
                prompt = str(payload.get("prompt", "")).strip("\n")
                if not prompt or len(prompt) > 2000:
                    self._json(400, {"error": "prompt must be 1-2000 characters"})
                    return
                try:
                    max_new = int(payload.get("max_new_tokens", 48))
                    temperature = float(payload.get("temperature", 0.0))
                except (TypeError, ValueError):
                    self._json(400, {"error": "max_new_tokens and temperature must be numeric"})
                    return
                if not 1 <= max_new <= 256:
                    self._json(400, {"error": "max_new_tokens must be 1-256"})
                    return
                if not 0.0 <= temperature <= 5.0:
                    self._json(400, {"error": "temperature must be 0-5"})
                    return
                job = CONSOLE.start("generate", {
                    "prompt": prompt, "max_new_tokens": max_new, "temperature": temperature,
                })
            elif route == "/api/tests":
                job = CONSOLE.start("tests", {})
            elif route == "/api/build":
                job = CONSOLE.start("build", {})
            else:
                self._json(404, {"error": "not found"})
                return
        except BusyError:
            self._json(409, {"error": "an engine job is already running"})
            return
        self._json(202, {"job_id": job["id"], "kind": job["kind"]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Base44 preview console for Niyah.Engine")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    missing = [n for n in BINARIES if not (BUILD_DIR / n).is_file()]
    print(
        f"niyah console on http://{args.host}:{args.port} "
        f"(build_dir={BUILD_DIR}, missing={missing or 'none'})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
