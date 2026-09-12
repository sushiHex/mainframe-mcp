"""Check a built wheel in a fresh CPU-only environment, outside the checkout."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


PROBE = '''
import importlib
from importlib import metadata
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = json.loads((root / "modules.json").read_text())
prefix = Path(sys.prefix).resolve()
for name in expected:
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(prefix):
        raise RuntimeError(f"{name} was imported outside the installed environment")

distribution = metadata.distribution("mainframe-mcp")
for name in ("mainframe", "mainframe_mcp"):
    if importlib.import_module(name).__version__ != distribution.version:
        raise RuntimeError(f"{name} runtime version differs from installed metadata")
entrypoints = {e.name: e for e in distribution.entry_points if e.group == "console_scripts"}
for name in ("mainframe", "mainframe-mcp"):
    if not callable(entrypoints[name].load()):
        raise RuntimeError(f"{name} entry point is not callable")
if entrypoints["mainframe-mcp"].value != "mainframe.adapters.mcp_stdio_shim:main":
    raise RuntimeError("default MCP entry point must share the daemon's Harrier stack")

from mainframe import config as v2
from mainframe_mcp import config as v1
default = v2.load_config(root / "missing.json")
if default["embedder"]["model"] != "microsoft/harrier-oss-v1-0.6b" or default["embedder"]["encoding"] != "native":
    raise RuntimeError("installed default does not select native Harrier")
for module in (v1, v2):
    reranker = module.load_config(root / "missing.json")["reranker"]
    if (reranker["model"] != "Qwen/Qwen3-Reranker-0.6B"
            or reranker["revision"] != "e61197ed45024b0ed8a2d74b80b4d909f1255473"
            or reranker["quantize"]):
        raise RuntimeError("installed default does not select the pinned BF16 compact reranker")
    config = module.load_config(root / "settings.json")
    if Path(config["paths"]["mainframe_dir"]).resolve() != root / "state":
        raise RuntimeError("installed configuration did not honor its state directory")
    if config["embedder"]["model"] != "synthetic-embedding-model":
        raise RuntimeError("installed configuration did not honor its model setting")

import torch
if torch.cuda.is_initialized():
    raise RuntimeError("package imports initialized CUDA")
print(json.dumps({"version": distribution.version, "modules": len(expected),
                  "entrypoints": sorted(entrypoints), "configuration": "passed"}))
'''


def check(wheel_dir):
    wheels = list(Path(wheel_dir).resolve().glob("mainframe_mcp-*.whl"))
    if len(wheels) != 1:
        raise ValueError("wheel directory must contain exactly one mainframe-mcp wheel")
    # Derive the expected inventory from source, then import it from an isolated
    # interpreter. Pytest's conftest prepends src/, hiding missing wheel modules.
    source = Path(__file__).resolve().parents[1] / "src"
    modules = []
    for path in sorted(source.rglob("*.py")):
        parts = list(path.relative_to(source).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules.append(".".join(parts))
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("MAINFRAME_") and k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1")
    with tempfile.TemporaryDirectory(prefix="mainframe-install-") as temporary:
        root = Path(temporary).resolve()
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")

        def run(*args, expected=0):
            result = subprocess.run([str(x) for x in args], cwd=root, env=env, stdin=subprocess.DEVNULL)
            if result.returncode != expected:
                raise RuntimeError(f"installed-package check exited {result.returncode}; expected {expected}")

        # Older Python releases seed an old pip that rejects normalized package
        # names in current index metadata. Upgrade only this disposable venv.
        run(python, "-m", "pip", "install", "--upgrade", "pip>=25.3")
        run(python, "-m", "pip", "install", "--no-compile", "torch",
            "--index-url", "https://download.pytorch.org/whl/cpu")
        run(python, "-m", "pip", "install", "--no-compile", wheels[0])
        (root / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
        (root / "settings.json").write_text(json.dumps({
            "paths": {"mainframe_dir": str(root / "state"), "repos_dir": str(root / "repos"),
                      "include_projects": ["example-project"]},
            "embedder": {"model": "synthetic-embedding-model"}}), encoding="utf-8")
        env["MAINFRAME_CONFIG"] = str(root / "settings.json")
        (root / "probe.py").write_text(PROBE, encoding="utf-8")
        run(python, "-I", root / "probe.py", root)
        cli = scripts / ("mainframe.exe" if os.name == "nt" else "mainframe")
        run(cli, "--help")
        run(cli, "validate", "--help")
        run(cli, "status", expected=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", required=True, type=Path)
    check(parser.parse_args().wheel_dir)
