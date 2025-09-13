# Copilot Project Instructions: AgentClinic

Purpose: Multimodal clinical simulation benchmark. Three interacting LLM agents (Doctor, Patient, Measurement) plus a Moderator judge. Supports multiple datasets (MedQA, MedQA_Ext, NEJM, NEJM_Ext, MIMICIV) and bias injection. This file orients AI coding assistants for productive, safe changes.

## Architecture & Data Flow
1. Entry point: `agentclinic.py:main()` loads two configs: `agentclinic.config.json` (run + dataset paths) and `llm.config.json` (OpenAI + runtime). Merged config is persisted per run in `runs/<DATASET>_<TS>/config.used.json`.
2. Scenario layer: one loader + scenario class per dataset (`ScenarioLoader*` + `Scenario*`). Each exposes `patient_information`, `examiner_information`, `exam_information`, `diagnosis_information` used by agents.
3. Agents:
   - `DoctorAgent`: drives conversation; must output either a question, `REQUEST TEST: <test>` or `DIAGNOSIS READY: <diagnosis>` (capitalization critical for control flow). Optional `REQUEST IMAGES` (NEJM only when `doctor_image_request` true) triggers image inclusion.
   - `PatientAgent`: answers, never reveals diagnosis directly; may include bias prompt segment.
   - `MeasurementAgent`: returns test results; responds with `RESULTS:` prefix (prompt enforces).
   - `compare_results` uses `moderator.json` prompt; expects pure `Yes`/`No`.
4. Loop control: For each scenario up to `total_inferences`, doctor turn -> branching:
   - Contains `DIAGNOSIS READY` -> evaluate & record.
   - Contains `REQUEST TEST` -> call MeasurementAgent, append to histories.
   - Else normal patient reply (human or LLM). Histories appended verbatim; simple string search controls logic.
5. Parallelism: Optional multi-scenario threading (`--workers` >1 and not human modes). Thread-safe file appends guarded by `lock`.
6. Outputs: Per-scenario record appended to `scenarios.jsonl`; final `summary.json` with accuracy + timing.

## Config & Runtime
- `agentclinic.config.json` keys under `run`: dataset selection, model names, bias flags, counts. Change dataset paths via `datasets` mapping without editing code.
- `llm.config.json` supplies OpenAI credentials & unified Responses API kwargs; sanitized in `set_llm_config` to avoid overriding reserved keys.
- NEVER hardcode or commit real API keys (current file includes a key; treat as placeholder—remove / env var in contributions).
- Runtime overrides: global `_llm_runtime_cfg` supports retries, timeout (sleep between retries), prompt clipping.

## Prompts & Conventions
- Prompt JSON files in `prompts/` with keys: `system_base`, optional suffixes, `user_prompt_template`, and `biases` map (doctor & patient). Bias lookup returns string or prints warning; unsupported value silently ignored.
- Control tokens (exact strings): `REQUEST TEST:`, `DIAGNOSIS READY:`, `REQUEST IMAGES`. Chat loop depends on simple substring tests; preserve casing and colon.
- New agent behaviors: extend by adding new prompt file + new Agent class with `inference_<role>` method mirroring pattern (history accumulation + `query_model`).

## Adding a Dataset
1. Place processed JSONL in `processed_data/` with one JSON object per line conforming to fields consumed by a new Scenario class.
2. Implement `Scenario<Dataset>` & `ScenarioLoader<Dataset>` (mirror existing). Required methods: `patient_information`, `examiner_information`, `exam_information`, `diagnosis_information`.
3. Register dataset key + path in `agentclinic.config.json` under `datasets` and extend loader dispatch in `main`.

## Generation Scripts
- Located in `generate_cases/`; e.g. `gen_mimic_tutorial.py` streams raw MIMIC-IV CSVs -> builds patient context -> prompts model for OSCE JSON structure (see embedded `examples` template). Uses `ThreadPoolExecutor` and validates JSON before append.
- `gen.config.json` holds per-source generation params (`limit`, `output`, `base_dataset_dir`). Override via CLI flags.

## LLM Abstraction
- Single `query_model` using OpenAI Responses API for text & optional image (NEJM). Multimodal call constructed when `image_requested` true and scenario has `image_url`.
- Global config merges `llm.llm.responses` into request except reserved fields (`model`, `input`) and GPT-5-only keys for non-GPT-5 models.

## Extending / Modifying Safely
- When altering loop logic, adjust substring checks consistently; consider compiling regex if patterns grow.
- Maintain JSONL append atomicity (use existing lock if adding writers).
- If adding new output fields, update both per-scenario record and doc consumers (summary unaffected unless needed).
- For performance: adjust `--workers` but ensure OpenAI rate limits & `timeout` interplay (current retry sleeps `timeout` seconds).

## Quick Start Examples
Evaluate 10 MedQA scenarios with biases:
`python agentclinic.py --config agentclinic.config.json --workers 4`
Generate MIMIC cases (limit 15 workers 30):
`python generate_cases/gen_mimic_tutorial.py --workers 30 --limit 15`

## Common Pitfalls
- Forgetting `DIAGNOSIS READY` prevents evaluation (loop ends with status `max_infs_reached`).
- Changing prompt phrasing may break control flow—ensure required tokens remain.
- Large prompts: enable clipping in `llm.config.json` (`clip_prompt: true`) if hitting length errors.
- Parallel runs on Windows: avoid interactive `human_*` modes with `--workers >1`.

## Style & Dependencies
- Minimal deps (`requirements.txt`); keep new libs justified. Avoid heavy frameworks.
- Use same history accumulation pattern (`agent_hist += ... + "\n\n"`).

Feedback: Provide clarifications if adding agents, datasets, or evaluation metrics so we can refine these instructions.
