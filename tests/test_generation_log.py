import multi_agent_dev


def test_generation_log_contains_every_agent_response():
    formatter = getattr(multi_agent_dev, "format_generation_history_log", None)
    assert formatter is not None, "generation history formatter is missing"

    history = [
        {
            "timestamp": "2026-08-07 09:00:00",
            "role": "Q1_Writer (model-a)",
            "version": 1,
            "content": "Writer draft response",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        },
        {
            "timestamp": "2026-08-07 09:01:00",
            "role": "Q1_Reviewer 1 (model-b)",
            "version": 2,
            "content": "Reviewer response",
        },
    ]

    result = formatter(1, "Innate immune responses", history)

    assert "Question 1: Innate immune responses" in result
    assert "Q1_Writer (model-a)" in result
    assert "Writer draft response" in result
    assert "prompt_tokens: 10" in result
    assert "Q1_Reviewer 1 (model-b)" in result
    assert "Reviewer response" in result

