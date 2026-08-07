# Response to Reviewer 1

We sincerely thank the reviewer for downloading, deploying, and testing MAID.
The reported deployment and runtime issue helped us improve both the
reliability and accessibility of the software. We have revised the repository
and verified the updated workflow. The principal changes are summarized below.

- We fixed the model-configuration callback error that previously caused
  `save_model_config() missing 1 required positional argument` when the user
  clicked **Save Config**.
- We repaired the MCQ generation pipeline error in which
  `MCQDevelopmentSystem` did not expose `call_ai_model`, and restored the full
  Writer, Reviewer, Editor, revision, and final-decision workflow.
- We added native NVIDIA model support through
  `langchain_nvidia_ai_endpoints.ChatNVIDIA`, while retaining compatibility
  with standard OpenAI-compatible endpoints.
- We added **Fill with Demo LLMs - No API Key Required**. Before assigning
  models to agent roles, MAID tests every bundled NVIDIA model and excludes
  models that are temporarily unavailable.
- We added a prominent waiting dialog with per-model health-check results and
  progress indicators for both Demo LLM testing and item generation.
- We preserved the original Chinese medical examination syllabus and added a
  separate English **USMLE Step 1** syllabus for English-speaking reviewers.
- We corrected syllabus-tree handling so leaf-level topics from both built-in
  syllabuses can be selected and used as item-generation targets.
- We added a dedicated **Prompt DIY** tab where users can customize and persist
  the system prompts for the Writer, Reviewers, Editor, and Final Decision
  Editor.
- We added a Windows one-click launcher, `start_maid_windows.bat`, which checks
  Python and pip dependencies, installs missing packages from the official
  PyPI source, starts MAID in the background, and opens the user interface.
- We added `deploymentInstruction.txt`, which provides complete instructions
  for Windows one-click deployment, manual deployment, model configuration,
  syllabus selection, item generation, and troubleshooting.
- We expanded the downloadable generation log so it records the complete
  response from every agent and LLM for every generated question, including
  role, model, timestamp, version, and token-usage information.
- We added a short end-to-end demonstration video,
  **[`docs/MAID_demo.mp4`](docs/MAID_demo.mp4)**. The video shows the basic
  workflow from syllabus selection and Demo LLM loading to item generation.
- We verified the revised application on multiple computers and tested the
  actual end-to-end generation workflow using syllabus-derived content points.
  Automated regression tests were also added for the reported failures and the
  newly introduced functionality.

Users can now try the complete item-generation workflow without entering or
configuring any API key. They can launch MAID with the Windows one-click script,
follow the included demonstration video, select either the English USMLE Step 1
or original Chinese syllabus, click **Fill with Demo LLMs - No API Key
Required**, and generate demonstration items. This substantially lowers the
technical barrier for reviewers who wish to evaluate MAID before configuring
their own model credentials.

