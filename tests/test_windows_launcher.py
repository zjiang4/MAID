from pathlib import Path


LAUNCHER = Path(__file__).parents[1] / "start_maid_windows.bat"


def test_windows_launcher_is_self_contained_and_uses_official_pypi():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'set "PYTHON_CMD=python"' in source
    assert "https://pypi.org/simple" in source
    assert '"%PYTHON_CMD%" -m pip install -r requirements.txt' in source
    assert "multi_agent_dev.py" in source


def test_windows_launcher_starts_hidden_waits_and_opens_browser():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "Start-Process" in source
    assert "-WindowStyle Hidden" in source
    assert "maid_server.log" in source
    assert "127.0.0.1:8080" in source
    assert "Start-Process 'http://127.0.0.1:8080'" in source

