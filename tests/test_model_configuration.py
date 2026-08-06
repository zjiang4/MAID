from pathlib import Path


SOURCE = Path(__file__).parents[1] / "multi_agent_dev.py"


def test_save_config_callback_passes_model_id_input():
    source = SOURCE.read_text(encoding="utf-8")

    assert 'model_id_in = ui.input(label="Model ID")' in source
    assert (
        "save_model_config(name_in, ak_in, bu_in, mt_in, model_id_in, provider_in)"
        in source
    )


def test_server_listens_outside_localhost_for_reviewer_run_command():
    source = SOURCE.read_text(encoding="utf-8")

    assert 'host="0.0.0.0"' in source


def test_demo_button_uses_preconfigured_key_as_fallback():
    source = SOURCE.read_text(encoding="utf-8")

    assert "os.environ.get('NVIDIA_API_KEY', DEFAULT_NVIDIA_DEMO_API_KEY)" in source
