import json
from pathlib import Path

from mainframe_mcp.config import load_config


def test_shared_configuration_preserves_project_filters_while_expanding_paths(tmp_path, monkeypatch):
    for name in ("MAINFRAME_CONFIG", "MAINFRAME_PRESET", "MAINFRAME_DIR", "MAINFRAME_REPOS_DIR"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"paths": {"mainframe_dir": "~/mainframe-example",
                                           "include_projects": ["example-*"],
                                           "exclude_projects": ["example-old"]}}), encoding="utf-8")
    config = load_config(path)
    assert config["paths"]["include_projects"] == ["example-*"]
    assert config["paths"]["exclude_projects"] == ["example-old"]
    assert Path(config["paths"]["mainframe_dir"]) == Path.home() / "mainframe-example"
