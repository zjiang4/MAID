from pathlib import Path


SOURCE = Path(__file__).parents[1] / "multi_agent_dev.py"


def test_mcq_system_delegates_model_calls():
    source = SOURCE.read_text(encoding="utf-8")

    assert "async def call_ai_model(self, model: AIModel" in source
    assert "return await model.call_ai_model(system_prompt, user_prompt, json_mode)" in source
    assert "def call_ai_model_stream(self, model: AIModel" in source
    assert "return model.call_ai_model_stream(system_prompt, user_prompt)" in source


def test_pipeline_returns_the_system_history_instead_of_empty_local_history():
    source = SOURCE.read_text(encoding="utf-8")

    assert "mcq_system.clear_history()" in source
    assert "return list(mcq_system.history), usage_stats" in source
    assert "history = []\n    usage_stats" not in source


def test_generation_ui_has_overall_and_agent_progress():
    source = SOURCE.read_text(encoding="utf-8")

    assert 'ui_refs["overall_progress"]' in source
    assert 'ui_refs["generation_phase_label"]' in source
    assert "Generation in progress" in source
