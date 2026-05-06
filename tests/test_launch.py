import pytest

import nlpsych_app.launch as launch


def test_launch_main_exits_with_clear_message_when_streamlit_missing(monkeypatch, capsys):
    monkeypatch.setattr(launch, "find_spec", lambda name: None)

    with pytest.raises(SystemExit) as excinfo:
        launch.main()

    assert excinfo.value.code == 1
    stderr = capsys.readouterr().err
    assert "Streamlit is not installed." in stderr
    assert "pip install nlpsych" in stderr


def test_launch_main_sets_default_config_and_invokes_streamlit(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(launch, "find_spec", lambda name: object())
    monkeypatch.setattr(launch, "files", lambda package: tmp_path)
    monkeypatch.setattr(launch.sys, "argv", ["nlpsych-app", "--server.port", "9999"])
    monkeypatch.setattr(launch.sys, "executable", "/fake/python")
    monkeypatch.delenv("STREAMLIT_CONFIG_DIR", raising=False)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.delenv(key, raising=False)

    def fake_run(cmd, check):
        calls.append({"cmd": cmd, "check": check})

    monkeypatch.setattr(launch.subprocess, "run", fake_run)

    launch.main()

    assert launch.os.environ["STREAMLIT_CONFIG_DIR"] == str(tmp_path / ".streamlit")
    assert launch.os.environ["OMP_NUM_THREADS"] == "1"
    assert launch.os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert calls == [
        {
            "cmd": [
                "/fake/python",
                "-m",
                "streamlit",
                "run",
                str(tmp_path / "app.py"),
                "--server.port",
                "9999",
            ],
            "check": True,
        }
    ]


def test_launch_main_respects_existing_streamlit_config_dir(monkeypatch, tmp_path):
    calls = []
    existing_cfg = str(tmp_path / "existing-config")

    monkeypatch.setattr(launch, "find_spec", lambda name: object())
    monkeypatch.setattr(launch, "files", lambda package: tmp_path / "pkg-root")
    monkeypatch.setattr(launch.sys, "argv", ["nlpsych-app"])
    monkeypatch.setenv("STREAMLIT_CONFIG_DIR", existing_cfg)
    monkeypatch.setattr(launch.subprocess, "run", lambda cmd, check: calls.append((cmd, check)))

    launch.main()

    assert launch.os.environ["STREAMLIT_CONFIG_DIR"] == existing_cfg
    assert calls[0][0][4] == str((tmp_path / "pkg-root") / "app.py")
