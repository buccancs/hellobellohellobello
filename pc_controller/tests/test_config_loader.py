from __future__ import annotations

import json

from pc_controller.src import config as cfg


def test_config_loader_reads_env_path_and_get(tmp_path, monkeypatch):
    # Prepare temp config
    d = tmp_path
    p = d / "cfg.json"
    data = {
        "server_ip": "127.0.0.1",
        "timesync_port": 9999,
        "custom": "x",
    }
    p.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setenv("PC_CONFIG_PATH", str(p))
    cfg.reload_config()

    loaded = cfg.get_config()
    assert loaded["server_ip"] == "127.0.0.1"
    assert loaded["timesync_port"] == 9999
    assert cfg.get("missing", 42) == 42


def test_config_loader_missing_file_returns_empty(tmp_path, monkeypatch):
    missing = tmp_path / "nope.json"
    monkeypatch.setenv("PC_CONFIG_PATH", str(missing))
    cfg.reload_config()
    loaded = cfg.get_config()
    assert isinstance(loaded, dict)
    assert loaded == {}
