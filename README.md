# MathTutorAI - Memory-Augmented MATHTUTOR v7

MathTutorAI is a Streamlit prototype for Tunisian Baccalaureate mathematics tutoring and exercise generation.

The main thesis contribution of this project is:

- LLM exercise generation
- memory adaptation
- curriculum alignment
- judge validation
- deterministic mathematical verification
- final student-facing display gate

Dataset fallback is not the normal success path. In v7, it is kept only as an explicit demo or offline mode.

## Main ideas

The app generates a mathematics exercise for a selected:

- section
- topic
- subtopic
- exercise type

Then it validates the result before any student sees it.

The student should only receive an exercise when it is:

- aligned with the official program
- structurally complete
- mathematically coherent
- correctly formatted
- locally rechecked when deterministic validation is possible

## v7 architecture

```text
Student or teacher request
    ->
Alignment precheck
    ->
Memory adaptation
    ->
OpenRouter LLM generation
    ->
Strict JSON extraction and schema normalization
    ->
Judge validation
    ->
Solution validation
    ->
Local deterministic validators
    ->
Student-facing format guard
    ->
Final presentation gate
    ->
Display + MongoDB persistence + audit logging
```

## LLM generation is the main path

When an OpenRouter API key is configured:

- the app first tries true LLM generation
- JSON mode is requested when supported
- invalid JSON is repaired or retried
- validator errors are injected into later retries
- dataset fallback is not silently presented as if it were LLM output

Expected successful LLM records look like:

- `record_kind = "final_presented"`
- `generation_backend = "openrouter-llm"` or `openrouter-llm-repaired-json`
- `is_true_llm_generation = true`
- `display_source_category = "llm_generated"`
- `fallback_used = false`

## Dataset fallback policy

Dataset fallback is allowed only when:

- OpenRouter is not configured, or
- the user explicitly activates demo mode, or
- a developer or test flow explicitly requests it

In normal mode, fallback must not hide LLM failure.

The UI now distinguishes:

- `Generation LLM validee`
- `Bloquee apres validation`
- `Mode demonstration dataset`

## Structured exercise schema v7

Each generated exercise is normalized to a strict structured schema:

```json
{
  "title": "string",
  "context": "string",
  "questions": ["string", "string"],
  "instruction": "string",
  "solution": "string",
  "expected_answer": "string",
  "answer_kind": "text | expression | numeric | table",
  "solution_steps": ["string"],
  "learning_objective": "string",
  "estimated_time": "string",
  "table_data": null,
  "chart_data": null,
  "graph_data": null,
  "generation_metadata": {
    "target_section": "string",
    "target_topic": "string",
    "target_subtopic": "string",
    "exercise_family": "string",
    "requires_symbolic_check": true,
    "requires_numeric_check": false
  }
}
```

Rules:

- `context` describes the situation
- `questions` must contain explicit student tasks
- `instruction` must combine the context and the numbered questions
- empty, placeholder-only, or context-only outputs are rejected

## Memory adaptation layers

v7 adds a more explicit memory stack.

### 1. Dataset semantic memory

The generator retrieves nearby dataset cases using a hybrid score:

- 60% semantic similarity
- 25% metadata match
- 10% exercise type and difficulty match
- 5% freshness or success prior

If sentence-transformers is not available, the code falls back to a lightweight TF-IDF similarity approach.

### 2. Generation outcome memory

Each generation attempt stores outcome signals such as:

- prompt signature
- retrieved case ids
- backend
- validation result
- judge issues
- local validation issues
- final display decision
- format issues

### 3. Positive memory

True LLM generations that passed the final gate are rewarded and reused as structural inspiration.

### 4. Negative memory

Repeated failure patterns are penalized, for example:

- invalid_json
- context_only
- probability_inconsistent
- schema_invalid
- format_invalid
- expected_answer_mismatch
- too_similar_to_source_case

### 5. Anti-copy protection

If a generated instruction is too similar to a retrieved source case, it is rejected and retried.

## Retry controller

v7 replaces a naive retry loop with a reason-aware controller.

The controller can switch between strategies such as:

- `normal_memory_adapted_generation`
- `strict_schema_generation`
- `simple_exercise_generation`
- `topic_template_guided_generation`
- `deterministic_arithmetic_repair`
- `final_fail_no_fallback`

Examples:

- repeated context-only failures force extra numbered questions
- repeated invalid JSON failures switch to a simpler schema prompt
- recognized probability arithmetic failures can trigger deterministic repair

## Validation pipeline

The validation pipeline is intentionally fail-closed.

### Judge validation

The judge checks:

- alignment with the official curriculum
- exercise completeness
- consistency between instruction, solution, and expected answer

### Solution validation

The solution validator checks:

- consistency between `solution` and `expected_answer`
- symbolic or numeric coherence when applicable

### Local deterministic validators

The project includes deterministic validators for selected topics and failure modes, including:

- derivatives
- limits
- integrals and areas
- recurrence sequences
- differential equations
- conics
- complex transformations
- regression and correlation
- probability consistency
- visual support requirements
- pedagogical completeness

### Student-facing format guard

Malformed math text is blocked before display. Examples:

- `frace`
- `fracpi`
- `mathbb R`
- `in fty`
- `e^{0U}_0`
- malformed integrals
- unresolved placeholders

### Final display gate

An exercise is shown to the student only if all core conditions pass, including:

- judge approved
- aligned with the program
- solution validation approved
- local validation approved
- student-facing format approved
- explicit questions detected
- symbolic checks passed when required
- no unsupported fallback shown in non-demo mode

## Deterministic probability checker

v7 adds a deterministic probability solver in:

- `frontend/utils/validators/probability_solver.py`

It currently supports common finite urn exercises such as:

- simultaneous draw of 2 balls without replacement
- random variable defined as the sum of drawn values
- recomputation of the law of `X`
- recomputation of `E(X)`
- recomputation of `V(X)`

If the exercise pattern is recognized:

- wrong probability arithmetic is rejected
- inconsistent `expected_answer` is rejected
- deterministic repair can rewrite the arithmetic while keeping the LLM-generated context and questions when they are valid

This keeps the thesis focus on LLM generation while using code to secure arithmetic correctness.

## Audit logging

The audit log keeps generation traces separate from student display success.

Important fields include:

- `generation_backend`
- `is_true_llm_generation`
- `llm_json_parse_status`
- `llm_generation_attempts_count`
- `fallback_used`
- `fallback_reason`
- `display_source_category`
- `judge_validation_flag`
- `solution_validation_flag`
- `local_validation_flag`
- `student_facing_format_flag`
- `final_display_decision`
- `final_display_blocking_reasons`
- `retry_strategy`
- `failure_categories`
- `memory_positive_cases_used`
- `memory_negative_patterns_avoided`

## Main folders

```text
app.py
README.md
requirements.txt
.streamlit/
data/
frontend/
tests/
```

Important files:

- `frontend/utils/api_client.py`
- `frontend/utils/exercise_agent.py`
- `frontend/utils/openrouter_client.py`
- `frontend/utils/exercise_presentation_gate.py`
- `frontend/utils/exercise_solution_validator.py`
- `frontend/utils/validators/local_math_validators.py`
- `frontend/utils/validators/probability_solver.py`
- `frontend/utils/memory_adaptation.py`
- `frontend/utils/generation_retry_controller.py`

## Local setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Secrets configuration

Copy the example file:

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Minimal OpenRouter and MongoDB example:

```toml
[openrouter]
api_key = "your-openrouter-api-key"
base_url = "https://openrouter.ai/api/v1"
exercise_model_primary = "qwen/qwen-2.5-7b-instruct"
exercise_model_fallback = "qwen/qwen-2.5-7b-instruct"
judge_model = "qwen/qwen-2.5-7b-instruct"
validator_model = "qwen/qwen-2.5-7b-instruct"
tutor_model = "qwen/qwen-2.5-7b-instruct"
openrouter_response_mode = "auto"
site_url = "http://localhost:8501"
app_name = "MathTutorAI"
allow_dataset_demo = false

[mongo]
uri = "mongodb://localhost:27017"
db_name = "mathtutorai"
```

## Run the app

```bash
streamlit run app.py
```

## Run the validation suite

```bash
python tests/run_validation_tests.py
python -m compileall .
```

## v8 OpenRouter diagnostics

The generation pipeline now keeps OpenRouter failures separate from JSON parsing failures. Each attempt records the model, response format mode, HTTP status when available, error type, raw response preview, JSON extraction method, prompt size, retry strategy, and injected memory cases.

Useful local diagnostics:

```bash
python scripts/check_openrouter_generation.py
python scripts/check_memory_retrieval.py --section "Sciences expérimentales" --topic "Statistiques" --subtopic "séries à deux caractères, régression et corrélation"
```

Response format fallback order is `json_schema`, then `json_object`, then prompt-only JSON when `openrouter_response_mode = "auto"`. Dataset fallback remains an explicit demo/offline path and is not silently counted as a valid LLM generation.

## v9 validation hardening

The v9 layer adds deterministic domain checks before final display:

- LaTeX repair/guard for corrupted commands such as `extit{}`, `rac{}`, `hickapprox`, `extasciitilde`, and repeated backslashes before commands.
- Regression/statistics recalculation from the displayed data, including transformed values such as `y=ln(x)`, correlation, and regression-line coefficients.
- Bayes/conditional-probability validation and deterministic repair for total probability and posterior probability exercises.
- Exponential-law validation for simple probability queries such as `P(a<X<b)`, `P(X>a)`, and `P(X<=a)`.
- Stricter memory retrieval filtering so Bayes/probability prompts do not receive geometry cases when same-topic cases exist.
- Judge and solution-validator calls now use the same structured-output fallback path as the generator.

## Notes for packaging

Before zipping the project:

- remove `__pycache__`
- remove `.pyc` files
- keep runtime audit logs out of versioned deliverables when not needed

## Current scope

This is still a Streamlit research prototype, not a production backend.

The app keeps:

- student and teacher pages
- MongoDB persistence
- audit logs
- tutoring chat
- assignment workflows
- deterministic validators
- LLM judge and validation chain

while making true LLM generation the primary success path.
