from prompt_settings import load_prompt_settings, save_prompt_settings


def test_prompt_settings_round_trip(tmp_path):
    path = tmp_path / "prompt_settings.json"
    prompts = {"writer_system": "Custom writer", "reviewer_system": "Custom reviewer"}

    save_prompt_settings(prompts, path)

    assert load_prompt_settings(path) == prompts


def test_missing_prompt_settings_returns_empty_mapping(tmp_path):
    assert load_prompt_settings(tmp_path / "missing.json") == {}
