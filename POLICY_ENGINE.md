# Doctor Test Policy Engine

This module turns the prompt-only test policy into a structured planning loop when `doctor_test_policy_enabled` is set to `true`.

## Workflow
1. On every doctor turn the engine recreates the case context (examiner objective, baseline patient information, transcript so far, and the latest patient/test reply).
2. `DifferentialDiagnosisEngine` asks the configured doctor model to produce a weighted differential. Each hypothesis must list the configured minimum number of high-yield findings (weights 1-5) and tag whether every finding is present, absent, or still unknown based on the dialogue.
3. The engine converts those findings into support (+weight), contradiction (-weight), and unknown (0). It normalises to a 0-1 confidence score and also tracks coverage so low-evidence hypotheses cannot trigger an early finish.
4. If the top hypothesis clears `finish_early_if_confident`, beats the competition by more than `confidence_close_threshold`, coverage is ≥0.6, and the doctor has already exchanged dialogue, the agent auto-emits `DIAGNOSIS READY`.
5. When multiple diagnoses remain within the confidence gap, a discriminator prompt asks for the most discriminating symptoms/tests. The answers appear inside `[Planning Notes]` to steer the next turn while keeping the dialogue output compliant.
6. If the LLM repeatedly fails to meet the minimum-key-symptom requirement, the engine relaxes the requirement one step at a time (never below three) and annotates the guidance so you know why the change occurred. Successful turns gradually restore the requested minimum.
7. When, even after relaxation, the planner cannot return valid JSON, a fallback decision is emitted: planning notes will state that the planner is temporarily unavailable and instruct the doctor to gather an open-ended history/reset context. The next turn re-attempts the structured plan.

## Configuration
Behaviour is controlled from `prompts/test_policy.json`:

- `constraints.finish_early_if_confident`: numeric threshold (default `0.7`).
- `differential_settings.min_symptoms_per_disease`: minimum findings per hypothesis (default `4`).
- `differential_settings.max_differentials`: maximum diagnoses retrieved (default `5`).
- `differential_settings.confidence_close_threshold`: confidence margin that triggers discriminator planning (default `0.12`).

## Notes
- The same doctor model handles both dialogue and planning, so reasoning stays aligned with its conversational style.
- `[Planning Notes]` are only injected when the policy is active; legacy runs stay unchanged.
- When the planner is recovering from errors or a requirement relaxation occurs, the notes include explicit messaging so you can diagnose issues quickly.
- Automatic diagnoses are suppressed on the very first doctor turn to ensure at least one history interaction before committing to a conclusion.
