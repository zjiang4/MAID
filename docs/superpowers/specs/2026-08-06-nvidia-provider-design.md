# NVIDIA Provider Design

## Goal

Fix the model configuration callback crash and add NVIDIA-hosted language models without changing the existing OpenAI-compatible workflow.

## Architecture

Model configurations gain a `provider` field whose supported values are `openai` and `nvidia`. A focused provider module constructs clients, normalizes responses, performs health checks, and keeps NVIDIA-specific behavior out of the NiceGUI page. Existing configurations without a provider continue to default to OpenAI.

The NVIDIA demo action reads `NVIDIA_API_KEY` from the process environment. It tests every configured demo model with a short prompt, records individual failures, adds only healthy models to application state, and randomly assigns those healthy models to all text roles. No usable credential is committed to the repository.

## UI Behavior

- The manual configuration form contains display name, provider, model ID, API key, base URL, and model type.
- Save Config passes all fields to the handler and preserves backward compatibility.
- Fill with Demo LLMs shows progress, tests all NVIDIA demo candidates, reports failures, and assigns only successful models.
- If no demo model is healthy, existing role assignments remain unchanged.

## Error Handling

Provider calls return the existing `(content, usage)` shape. NVIDIA reasoning content is retained in usage metadata when present. Health checks use bounded timeouts and return structured status rather than allowing one failing endpoint to abort the batch.

## Testing

Unit tests cover the reviewer callback regression, provider defaults, NVIDIA response normalization, health-check failure isolation, healthy-model filtering, and role assignment. Live smoke testing uses the supplied key only through the local environment.

