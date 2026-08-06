# NVIDIA Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair model configuration and add health-checked NVIDIA demo models to MAID.

**Architecture:** Add a small provider module that presents a common async interface for OpenAI and ChatNVIDIA. Keep NiceGUI responsible for state and notifications, while pure helper functions handle demo filtering and assignments for straightforward tests.

**Tech Stack:** Python 3.11, NiceGUI, OpenAI Python SDK, langchain-nvidia-ai-endpoints, pytest.

---

### Task 1: Regression Test the Configuration Form

**Files:**
- Create: `tests/test_model_configuration.py`
- Modify: `multi_agent_dev.py`

- [ ] Write a source-level regression test that verifies the Save Config callback passes display name, key, URL, type, and model ID inputs.
- [ ] Run `pytest tests/test_model_configuration.py -v` and confirm the test fails against the reviewer-reported four-argument callback.
- [ ] Add the missing model ID input and pass it to `save_model_config`.
- [ ] Run the focused test and confirm it passes.

### Task 2: Provider Adapter

**Files:**
- Create: `model_providers.py`
- Create: `tests/test_model_providers.py`
- Modify: `multi_agent_dev.py`

- [ ] Write failing tests for OpenAI default-provider compatibility and normalized NVIDIA invoke/stream responses.
- [ ] Run `pytest tests/test_model_providers.py -v` and confirm failures identify the missing adapter.
- [ ] Implement OpenAI and NVIDIA clients behind a common `AIModel` interface, including reasoning and usage normalization.
- [ ] Run provider tests and confirm they pass.

### Task 3: Demo Health Checks and Assignment

**Files:**
- Modify: `model_providers.py`
- Modify: `multi_agent_dev.py`
- Modify: `tests/test_model_providers.py`

- [ ] Write failing tests proving all demo models are checked, failures are isolated, and only healthy models are assigned.
- [ ] Run focused tests and confirm expected failures.
- [ ] Add the NVIDIA demo catalog, bounded concurrent health checks, and pure healthy-model assignment helper.
- [ ] Add the Fill with Demo LLMs button and progress/result notifications.
- [ ] Run focused tests and confirm they pass.

### Task 4: Dependencies and Documentation

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`
- Create: `.env.example`

- [ ] Add `langchain-nvidia-ai-endpoints`, `pytest`, setup instructions, provider configuration guidance, and the `NVIDIA_API_KEY` example.
- [ ] Verify no live API or GitHub token appears in tracked files.

### Task 5: Full Verification and Publication

**Files:**
- Modify as required by verification findings.

- [ ] Run the complete pytest suite.
- [ ] Compile all Python modules.
- [ ] Start the NiceGUI application and verify the page responds locally.
- [ ] Run live NVIDIA health checks with the supplied key and document which models are currently healthy.
- [ ] Initialize/restore Git metadata, review the diff, commit, and push to `zjiang4/MAID` only after every verification step succeeds.

