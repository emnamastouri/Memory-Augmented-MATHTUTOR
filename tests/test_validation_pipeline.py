from __future__ import annotations

from fractions import Fraction

import streamlit as st

from frontend.utils.exercise_presentation_gate import can_present_exercise, validate_student_facing_math_text
from frontend.utils.math_format_guard import repair_corrupted_latex_commands, repair_math_text_locally
from frontend.utils.exercise_schema import has_explicit_questions, normalize_exercise_schema
from frontend.utils.generation_retry_controller import GenerationRetryController, classify_failure_categories
from frontend.utils.openrouter_client import (
    OpenRouterCallResult,
    call_openrouter_chat,
    extract_json_object,
    parse_json_object_detailed,
    validate_generated_exercise_schema,
)
from frontend.utils import memory_adaptation as memory_module
from frontend.utils.validators.local_math_validators import (
    _validate_differential_equation,
    _validate_regression_numeric,
    validate_exercise_locally,
)
from frontend.utils.validators.probability_solver import (
    compute_expectation,
    compute_two_draw_sum_distribution,
    compute_variance,
    repair_probability_exercise_with_deterministic_solution,
    validate_probability_distribution,
    validate_probability_exercise,
)
from frontend.utils.validators.bayes_solver import (
    compute_bayes_values,
    parse_bayes_context,
    repair_bayes_solution,
    validate_bayes_exercise,
)
from frontend.utils.validators.complex_number_solver import (
    is_nth_root_of_unity,
    parse_complex_expression,
    validate_complex_number_exercise,
    verify_polynomial_root,
)
from frontend.utils.validators.domain_router import get_domain_validator_key
from frontend.utils.validators.exponential_law_solver import (
    compute_exponential_probability,
    parse_exponential_context,
    validate_exponential_law_exercise,
)
from frontend.utils.validators.graph_support_validator import validate_graph_support
from frontend.utils.validators.linear_system_solver import (
    parse_linear_system,
    repair_latex_cases_environment,
    solve_linear_system,
    validate_linear_system_exercise,
)
from frontend.utils.validators.question_coverage import validate_question_answer_coverage
from frontend.utils.validators.regression_solver import (
    compute_linear_regression,
    compute_transformed_y,
    parse_numeric_series_from_text,
    repair_regression_solution,
    validate_regression_exercise,
)


def _base_record() -> dict:
    context = "On considere la fonction \\(f(x)=x-\\ln(x)\\) definie sur \\(]0,+\\infty[\\)."
    questions = [
        "Calculer \\(f'(x)\\).",
        "Determiner le sens de variation de \\(f\\).",
    ]
    instruction = (
        f"{context} "
        "1) Calculer \\(f'(x)\\). "
        "2) Determiner le sens de variation de \\(f\\)."
    )
    return {
        "id": "LLM-TEST",
        "title": "Exercice test",
        "context": context,
        "questions": questions,
        "instruction": instruction,
        "prompt": instruction,
        "display_answer": "\\(f'(x)=1-1/x\\) et \\(f\\) est croissante sur \\([1,+\\infty[\\).",
        "hidden_solution": (
            "On derive \\(f\\) et on obtient \\(f'(x)=1-1/x\\). "
            "Le signe de \\(f'(x)\\) permet de conclure que \\(f\\) decroit sur \\(]0,1]\\) "
            "puis croit sur \\([1,+\\infty[\\)."
        ),
        "answer_kind": "expression",
        "solution_steps": [
            "Deriver \\(x\\) puis \\(\\ln(x)\\).",
            "Etudier le signe de \\(1-1/x\\).",
        ],
        "learning_objective": "Etudier une fonction logarithmique.",
        "estimated_time": "10 a 15 min",
        "topic": "Analyse",
        "subtopic": "fonction logarithme",
        "section": "Mathematiques",
        "exercise_type": "Exercice probleme",
        "judge_validation_flag": "approved",
        "judge_alignment_status": "aligned",
        "judge_status": "Valide",
        "solution_validation_flag": "approved",
        "local_validation_flag": "approved",
        "symbolic_checks_required": False,
        "symbolic_checks_ran": False,
        "symbolic_checks_passed": None,
        "blocked_before_display": False,
        "support_ready": False,
        "generation_backend": "openrouter-llm",
        "pedagogical_completeness_flag": "approved",
        "pedagogical_completeness_issues": [],
        "student_facing_format_flag": "approved",
        "student_facing_format_issues": [],
        "display_source_category": "llm_generated",
        "is_true_llm_generation": True,
        "demo_mode_used": False,
        "judge_issues": [],
        "local_validation_issues": [],
        "local_validation_checks": {},
        "table_data": None,
        "chart_data": None,
        "graph_data": None,
        "corrected_fields_applied": False,
    }


def _base_schema_payload() -> dict:
    return {
        "title": "Loi de probabilite d'une variable aleatoire",
        "context": (
            "Une urne contient quatre boules portant le nombre -1, trois boules portant le nombre 0 "
            "et trois boules portant le nombre 1. On tire simultanement deux boules."
        ),
        "questions": [
            "Determiner la loi de probabilite de la variable aleatoire X, somme des deux nombres obtenus.",
            "Calculer E(X).",
            "Calculer V(X).",
        ],
        "instruction": (
            "Une urne contient quatre boules portant le nombre -1, trois boules portant le nombre 0 "
            "et trois boules portant le nombre 1. On tire simultanement deux boules. "
            "1) Determiner la loi de probabilite de la variable aleatoire X, somme des deux nombres obtenus. "
            "2) Calculer E(X). 3) Calculer V(X)."
        ),
        "solution": (
            "La loi est donnee par \\(P(X=-2)=\\frac{2}{15}\\), \\(P(X=-1)=\\frac{4}{15}\\), "
            "\\(P(X=0)=\\frac{1}{3}\\), \\(P(X=1)=\\frac{1}{5}\\), \\(P(X=2)=\\frac{1}{15}\\). "
            "On trouve ensuite \\(E(X)=\\frac{-1}{5}\\) puis \\(V(X)=\\frac{92}{75}\\)."
        ),
        "expected_answer": (
            "Loi : \\(P(X=-2)=\\frac{2}{15}\\), \\(P(X=-1)=\\frac{4}{15}\\), \\(P(X=0)=\\frac{1}{3}\\), "
            "\\(P(X=1)=\\frac{1}{5}\\), \\(P(X=2)=\\frac{1}{15}\\) ; "
            "\\(E(X)=\\frac{-1}{5}\\) ; \\(V(X)=\\frac{92}{75}\\)."
        ),
        "answer_kind": "text",
        "solution_steps": [
            "Compter toutes les paires possibles.",
            "Construire la loi de probabilite de X.",
            "Calculer l'esperance et la variance.",
        ],
        "learning_objective": "Maitriser la loi, l'esperance et la variance d'une variable aleatoire finie.",
        "estimated_time": "15 a 20 min",
        "table_data": None,
        "chart_data": None,
        "graph_data": None,
        "generation_metadata": {
            "target_section": "Sciences experimentales",
            "target_topic": "Probabilites",
            "target_subtopic": "variables aleatoires, esperance et variance",
            "exercise_family": "Exercice probleme",
            "requires_symbolic_check": False,
            "requires_numeric_check": True,
        },
    }


def _base_probability_exercise() -> dict:
    payload = _base_schema_payload()
    return {
        "title": payload["title"],
        "context": payload["context"],
        "questions": payload["questions"],
        "prompt": payload["instruction"],
        "instruction": payload["instruction"],
        "hidden_solution": payload["solution"],
        "display_answer": payload["expected_answer"],
        "learning_objective": payload["learning_objective"],
    }


def _reset_memory_state() -> None:
    for key in ("generation_outcome_memory", "generation_memory_bank"):
        if key in st.session_state:
            del st.session_state[key]


def test_extract_json_object_from_pure_json() -> None:
    parsed = extract_json_object(
        '{"title":"A","context":"Contexte","questions":["Calculer x.","Donner y."],"instruction":"Contexte 1) Calculer x. 2) Donner y.","solution":"S","expected_answer":"R","answer_kind":"text","solution_steps":[],"learning_objective":"E","estimated_time":"10 min","table_data":null,"chart_data":null,"graph_data":null,"generation_metadata":{"target_section":"Bac","target_topic":"Analyse","target_subtopic":"fonctions","exercise_family":"Exercice","requires_symbolic_check":false,"requires_numeric_check":false}}'
    )
    assert parsed is not None
    assert parsed["title"] == "A"


def test_extract_json_object_from_fenced_json() -> None:
    parsed = extract_json_object(
        """```json
{"title":"A","context":"Contexte","questions":["Calculer x.","Donner y."],"instruction":"Contexte 1) Calculer x. 2) Donner y.","solution":"S","expected_answer":"R","answer_kind":"text","solution_steps":[],"learning_objective":"E","estimated_time":"10 min","table_data":null,"chart_data":null,"graph_data":null,"generation_metadata":{"target_section":"Bac","target_topic":"Analyse","target_subtopic":"fonctions","exercise_family":"Exercice","requires_symbolic_check":false,"requires_numeric_check":false}}
```"""
    )
    assert parsed is not None
    assert parsed["context"] == "Contexte"


def test_extract_json_object_from_prose_wrapped_json() -> None:
    parsed = extract_json_object(
        'Voici le JSON demande : {"title":"A","context":"Contexte","questions":["Calculer x.","Donner y."],"instruction":"Contexte 1) Calculer x. 2) Donner y.","solution":"S","expected_answer":"R","answer_kind":"text","solution_steps":[],"learning_objective":"E","estimated_time":"10 min","table_data":null,"chart_data":null,"graph_data":null,"generation_metadata":{"target_section":"Bac","target_topic":"Analyse","target_subtopic":"fonctions","exercise_family":"Exercice","requires_symbolic_check":false,"requires_numeric_check":false}} merci.'
    )
    assert parsed is not None
    assert parsed["solution"] == "S"


def test_extract_json_object_rejects_invalid_json() -> None:
    assert extract_json_object("{title: A,}") is None


def test_schema_context_only_fails() -> None:
    payload = _base_schema_payload()
    payload["questions"] = []
    payload["instruction"] = payload["context"]
    is_valid, issues = validate_generated_exercise_schema(payload)
    assert not is_valid
    assert any("questions" in issue.lower() or "question" in issue.lower() for issue in issues)


def test_schema_context_plus_three_questions_passes() -> None:
    is_valid, issues = validate_generated_exercise_schema(_base_schema_payload())
    assert is_valid, issues


def test_generated_exercise_schema_rejects_missing_expected_answer() -> None:
    payload = _base_schema_payload()
    payload["expected_answer"] = ""
    is_valid, issues = validate_generated_exercise_schema(payload)
    assert not is_valid
    assert any("expected_answer" in issue.lower() for issue in issues)


def test_normalize_exercise_schema_splits_legacy_instruction() -> None:
    payload = {
        "title": "Legacy",
        "instruction": "Une urne contient deux boules rouges. 1) Calculer p. 2) Determiner E(X).",
        "solution": "S",
        "expected_answer": "R",
        "answer_kind": "text",
        "solution_steps": [],
        "learning_objective": "Objectif",
        "estimated_time": "10 min",
    }
    normalized = normalize_exercise_schema(payload)
    assert normalized["context"]
    assert len(normalized["questions"]) >= 2


def test_structural_guard_accepts_probability_command() -> None:
    assert has_explicit_questions(
        "Une urne contient ... On tire simultanement deux boules. Determinez la loi de probabilite de la variable aleatoire X."
    )


def test_structural_guard_rejects_pure_urn_context() -> None:
    assert not has_explicit_questions("Une urne contient ... On tire simultanement deux boules de l'urne.")


def test_structural_guard_accepts_expectation_question() -> None:
    assert has_explicit_questions("Calculer E(X).")


def test_student_facing_format_guard_rejects_frace() -> None:
    ok, issues = validate_student_facing_math_text(
        {"prompt": "On a frace^x", "display_answer": "1", "hidden_solution": "1", "solution_steps": []}
    )
    assert not ok
    assert any("frace" in issue.lower() for issue in issues)


def test_student_facing_format_guard_rejects_corrupted_suite_notation() -> None:
    ok, issues = validate_student_facing_math_text(
        {"prompt": "U_1=e^{0U}_0=1", "display_answer": "1", "hidden_solution": "1", "solution_steps": []}
    )
    assert not ok
    assert issues


def test_student_facing_format_guard_rejects_malformed_integral() -> None:
    ok, issues = validate_student_facing_math_text(
        {"prompt": r"A=\\int_\alpha^{{{0 f}}}(x)\,dx", "display_answer": "1", "hidden_solution": "1", "solution_steps": []}
    )
    assert not ok
    assert issues


def test_latex_repair_restores_common_commands() -> None:
    assert repair_corrupted_latex_commands("extit{t}") == "\\textit{t}"
    assert repair_corrupted_latex_commands("rac{1}{2}") == "\\frac{1}{2}"
    assert repair_corrupted_latex_commands("hickapprox") == "\\approx"
    assert repair_corrupted_latex_commands("extasciitilde") == "\\sim"
    assert repair_corrupted_latex_commands("\\\\\\ln(x)") == "\\ln(x)"
    assert repair_corrupted_latex_commands("extbf{A}") == "\\textbf{A}"
    assert repair_corrupted_latex_commands("egin{cases} x=1 end{cases}") == "\\begin{cases} x=1 \\end{cases}"


def test_domain_router_strict_keys() -> None:
    assert get_domain_validator_key("Analyse", "nombres complexes") == "complex_numbers"
    assert get_domain_validator_key("Probabilités", "loi exponentielle") == "exponential_law"
    assert get_domain_validator_key("Graphes", "graphes") == "graphs"
    assert get_domain_validator_key("Algèbre", "matrices, déterminants et systèmes linéaires") == "linear_systems"
    assert get_domain_validator_key("Statistiques", "séries à deux caractères, régression et corrélation") == "regression"
    assert get_domain_validator_key("Probabilités", "conditionnement, probabilités totales et Bayes") == "bayes"


def test_complex_number_validator_core_cases() -> None:
    z = parse_complex_expression("1/2+i*sqrt(3)/2")
    assert z is not None
    assert not is_nth_root_of_unity(z, 5)
    assert verify_polynomial_root("z^2-2*z+1", "1")
    assert not verify_polynomial_root("z^2-2*z+1", "1+i")
    ok, issues, _metadata = validate_complex_number_exercise(
        {**_base_record(), "subtopic": "nombres complexes", "prompt": "Soit z=1/2+i sqrt(3)/2. Dire si z est une cinquieme racine de l'unite.", "hidden_solution": "z est une cinquieme racine de l'unite.", "display_answer": "Vrai"}
    )
    assert not ok and issues


def test_linear_system_validator_solves_and_rejects_wrong_solution() -> None:
    system = repair_latex_cases_environment("egin{cases} 2x+y=10 ; x+2y=12 end{cases}")
    parsed = parse_linear_system(system)
    assert parsed
    solution = solve_linear_system(*parsed)
    assert solution and str(solution["x"]) == "8/3" and str(solution["y"]) == "14/3"
    ok, issues, _metadata = validate_linear_system_exercise(
        {**_base_record(), "subtopic": "matrices, déterminants et systèmes linéaires", "prompt": system, "hidden_solution": "x=1, y=2", "display_answer": "x=1, y=2"}
    )
    assert not ok and issues


def test_question_coverage_and_graph_support() -> None:
    record = {**_base_record(), "questions": ["Calculer x.", "Estimer y."], "hidden_solution": "1) x=2.", "display_answer": "x=2"}
    ok, issues, metadata = validate_question_answer_coverage(record)
    assert not ok and metadata["unanswered_question_indices"]
    graph_record = {**_base_record(), "topic": "Graphes", "subtopic": "graphes", "prompt": "On considere le graphe ci-dessous. Determiner une chaine eulerienne."}
    ok, issues, _metadata = validate_graph_support(graph_record)
    assert not ok and issues
    graph_record["graph_data"] = {"vertices": ["A", "B"], "edges": [["A", "B"]], "directed": False}
    ok, issues, _metadata = validate_graph_support(graph_record)
    assert ok


def test_student_guard_blocks_remaining_corrupted_commands() -> None:
    ok, issues = validate_student_facing_math_text(
        {**_base_record(), "prompt": "On obtient extit{t} puis rac{1}{2}."}
    )
    assert not ok
    assert issues


def test_regression_solver_recomputes_ln_values_and_detects_old_copy() -> None:
    instruction = (
        "On donne t = 0, 1, 2, 3, 4, 5, 6 et x = 9, 11, 14, 18, 22, 28, 36. "
        "On pose y=ln(x). Calculer r et la droite de regression."
    )
    parsed = parse_numeric_series_from_text(instruction)
    assert parsed
    y_values = compute_transformed_y(parsed["t"], parsed["x"], parsed["y_formula"])
    assert y_values and abs(y_values[4] - 3.091) < 0.01
    record = {
        **_base_record(),
        "topic": "Statistiques",
        "subtopic": "séries à deux caractères, régression et corrélation",
        "prompt": instruction,
        "context": instruction,
        "hidden_solution": "On obtient ln(22.8)=3.127 et la droite y=0.2t+2.",
        "display_answer": "r≈0.99, y=0.2t+2",
    }
    ok, issues, metadata = validate_regression_exercise(record)
    assert metadata["applicable"]
    assert not ok
    assert any("y_" in issue for issue in issues)


def test_regression_repair_rewrites_solution() -> None:
    record = {
        **_base_record(),
        "topic": "Statistiques",
        "subtopic": "régression",
        "prompt": "On donne t = 0, 1, 2, 3 et x = 9, 11, 14, 18. On pose y=ln(x). Calculer la droite de regression.",
        "context": "On donne t = 0, 1, 2, 3 et x = 9, 11, 14, 18. On pose y=ln(x).",
    }
    repaired = repair_regression_solution(record)
    assert repaired.get("corrected_fields_applied") is True
    assert "droite" in repaired["display_answer"].lower()


def test_bayes_solver_computes_and_repairs() -> None:
    text = "10% des composants sont defectueux. Parmi ceux-ci, 95% sont detectes. 2% des composants non defectueux sont detectes."
    params = parse_bayes_context(text)
    assert params
    values = compute_bayes_values(params)
    assert abs(float(values["p_h"]) - 0.113) < 1e-9
    assert abs(float(values["p_d_given_h"]) - 0.8407) < 1e-3
    record = {**_base_record(), "topic": "Probabilités", "subtopic": "conditionnement, probabilités totales et Bayes", "prompt": text, "hidden_solution": "P(H)=0.2", "display_answer": "P(H)=0.2"}
    ok, issues, _metadata = validate_bayes_exercise(record)
    assert not ok and issues
    repaired = repair_bayes_solution(record)
    assert repaired.get("corrected_fields_applied") is True
    assert "0.113" in repaired["hidden_solution"]


def test_exponential_law_solver() -> None:
    parsed = parse_exponential_context("X suit une loi exponentielle de paramètre lambda=0,2. Calculer P(3<X<5).")
    assert parsed and parsed["query"][0] == "between"
    expected = compute_exponential_probability(parsed["lambda"], parsed["query"])
    assert abs(expected - (2.718281828459045 ** -0.6 - 2.718281828459045 ** -1.0)) < 1e-3
    ok, issues, _metadata = validate_exponential_law_exercise(
        {**_base_record(), "subtopic": "loi exponentielle", "prompt": "X suit une loi exponentielle de paramètre lambda=0,2. Calculer P(3<X<5).", "hidden_solution": "0.1", "display_answer": "0.1"}
    )
    assert not ok and issues


def test_student_facing_format_guard_accepts_valid_latex() -> None:
    ok, issues = validate_student_facing_math_text(
        {
            "prompt": r"Calculer \( \int_\alpha^0 f(x)\,dx \). 1) Justifier le signe. 2) Donner l'aire.",
            "display_answer": r"\(\frac{1}{2}\)",
            "hidden_solution": r"Reponse finale : \(\frac{1}{2}\).",
            "solution_steps": [],
        }
    )
    assert ok
    assert not issues


def test_probability_solver_distribution_for_urn_case() -> None:
    distribution = compute_two_draw_sum_distribution({-1: 4, 0: 3, 1: 3})
    assert distribution == {
        -2: Fraction(2, 15),
        -1: Fraction(4, 15),
        0: Fraction(1, 3),
        1: Fraction(1, 5),
        2: Fraction(1, 15),
    }
    assert compute_expectation(distribution) == Fraction(-1, 5)
    assert compute_variance(distribution) == Fraction(92, 75)


def test_probability_distribution_sum_to_one() -> None:
    ok, issues = validate_probability_distribution(
        {-2: Fraction(2, 15), -1: Fraction(4, 15), 0: Fraction(1, 3), 1: Fraction(1, 5), 2: Fraction(1, 15)}
    )
    assert ok, issues


def test_probability_distribution_rejects_value_above_one() -> None:
    ok, issues = validate_probability_distribution({0: Fraction(5, 4)})
    assert not ok
    assert issues


def test_probability_exercise_rejects_wrong_expected_answer() -> None:
    exercise = _base_probability_exercise()
    exercise["display_answer"] = "Loi : \\(P(X=-2)=\\frac{1}{15}\\)."
    ok, issues, _metadata = validate_probability_exercise(exercise)
    assert not ok
    assert any("expected_answer" in issue.lower() or "display_answer" in issue.lower() for issue in issues)


def test_probability_deterministic_repair_corrects_expected_answer() -> None:
    exercise = _base_probability_exercise()
    exercise["display_answer"] = "Loi : \\(P(X=-2)=\\frac{1}{15}\\)."
    exercise["hidden_solution"] = "Solution fausse."
    repaired = repair_probability_exercise_with_deterministic_solution(exercise)
    assert repaired["corrected_fields_applied"] is True
    assert "\\frac{2}{15}" in repaired["display_answer"]
    ok, issues, metadata = validate_probability_exercise(repaired)
    assert ok, issues
    assert metadata["expectation"] == "\\frac{-1}{5}"


def test_valid_numeric_sequence_passes() -> None:
    record = {
        "title": "Suites numeriques",
        "topic": "Analyse",
        "subtopic": "suites numeriques",
        "prompt": (
            "On considere la suite \\(U_n\\) definie par \\(U_0=1\\) et \\(U_{n+1}=e^{-n}U_n\\). "
            "On pose \\(V_n=\\ln(U_n)\\). 1) Calculer \\(U_1\\) et \\(U_2\\). "
            "2) Montrer que \\(V_{n+1}-V_n=-n\\). "
            "3) En deduire \\(V_n=-\\frac{n(n-1)}{2}\\) puis \\(U_n=e^{-n(n-1)/2}\\). "
            "4) Determiner la limite de \\(U_n\\)."
        ),
        "hidden_solution": (
            "U_1=e^0U_0=1. U_2=e^{-1}. V_{n+1}-V_n=-n. "
            "V_n=-n(n-1)/2. U_n=e^{-n(n-1)/2}. La limite est 0."
        ),
        "display_answer": r"\(U_n=e^{-n(n-1)/2}\) et \(\lim U_n=0\)",
        "answer_kind": "text",
        "solution_steps": [],
    }
    outcome = validate_exercise_locally(record)
    assert outcome["local_validation_flag"] == "approved", outcome


def test_wrong_sequence_u2_fails() -> None:
    record = {
        "title": "Suites numeriques",
        "topic": "Analyse",
        "subtopic": "suites numeriques",
        "prompt": "On considere la suite \\(U_n\\) definie par \\(U_0=1\\) et \\(U_{n+1}=e^{-n}U_n\\).",
        "hidden_solution": "U_2=e^{-2}.",
        "display_answer": "Voir solution",
        "answer_kind": "text",
        "solution_steps": [],
    }
    outcome = validate_exercise_locally(record)
    assert outcome["local_validation_flag"] == "wrong"


def test_wrong_sequence_closed_form_fails() -> None:
    record = {
        "title": "Suites numeriques",
        "topic": "Analyse",
        "subtopic": "suites numeriques",
        "prompt": "On considere la suite \\(U_n\\) definie par \\(U_0=1\\) et \\(U_{n+1}=e^{-n}U_n\\).",
        "hidden_solution": "U_n=e^{-n(n+1)/2}.",
        "display_answer": "Voir solution",
        "answer_kind": "text",
        "solution_steps": [],
    }
    outcome = validate_exercise_locally(record)
    assert outcome["local_validation_flag"] == "wrong"


def test_missing_sequence_recurrence_fails() -> None:
    record = {
        "title": "Suites numeriques",
        "topic": "Analyse",
        "subtopic": "suites numeriques",
        "prompt": "On considere une suite numerique. Determiner sa limite.",
        "hidden_solution": "La limite est 0.",
        "display_answer": "0",
        "answer_kind": "text",
        "solution_steps": [],
    }
    outcome = validate_exercise_locally(record)
    assert outcome["local_validation_flag"] == "wrong"


def test_regression_numeric_valid_table_passes() -> None:
    record = {
        "title": "Regression lineaire",
        "topic": "Statistiques",
        "subtopic": "regression",
        "prompt": (
            "Le tableau suivant donne une serie a deux caracteres. "
            "1) Calculer le coefficient de correlation. 2) Determiner la droite de regression."
        ),
        "hidden_solution": "On obtient râ‰ˆ1. La droite de regression est P=1x+1.",
        "display_answer": "râ‰ˆ1 et P=1x+1",
        "answer_kind": "text",
        "table_data": {"caption": "Serie", "headers": ["x", "y"], "rows": [[1, 2], [2, 3], [3, 4], [4, 5]]},
    }
    outcome = _validate_regression_numeric(record)
    assert outcome["status"] == "passed", outcome


def test_regression_numeric_changed_table_fails() -> None:
    record = {
        "title": "Regression lineaire",
        "topic": "Statistiques",
        "subtopic": "regression",
        "prompt": (
            "Le tableau suivant donne une serie a deux caracteres. "
            "1) Calculer le coefficient de correlation. 2) Determiner la droite de regression."
        ),
        "hidden_solution": "On obtient râ‰ˆ1. La droite de regression est P=1x+1.",
        "display_answer": "râ‰ˆ1 et P=1x+1",
        "answer_kind": "text",
        "table_data": {"caption": "Serie", "headers": ["x", "y"], "rows": [[1, 10], [2, 9], [3, 8], [4, 7]]},
    }
    outcome = _validate_regression_numeric(record)
    assert outcome["status"] == "failed", outcome


def test_ode_general_solution_passes() -> None:
    record = {
        "title": "Equation differentielle",
        "topic": "Analyse",
        "subtopic": "equations differentielles",
        "prompt": "1) Resoudre l'equation differentielle y''+y=0. 2) Donner la solution generale.",
        "hidden_solution": "La solution generale est y=A sin(x)+B cos(x).",
        "display_answer": "y=A sin(x)+B cos(x)",
    }
    outcome = _validate_differential_equation(record)
    assert outcome["status"] == "passed", outcome


def test_ode_family_b_cos_x_passes_functional_condition() -> None:
    record = {
        "title": "Equation differentielle",
        "topic": "Analyse",
        "subtopic": "equations differentielles",
        "prompt": "1) Trouver les fonctions verifiant f'(x)+f(pi/2-x)=0. 2) Conclure sur la famille de solutions.",
        "hidden_solution": "On trouve f(x)=B cos(x).",
        "display_answer": "B cos(x)",
    }
    outcome = _validate_differential_equation(record)
    assert outcome["status"] == "passed", outcome


def test_memory_retrieval_prefers_matching_topic_and_subtopic() -> None:
    _reset_memory_state()
    original_loader = memory_module.load_dataset_exercise_cases
    normalized_section = memory_module.normalize_section_label("Mathematiques")
    memory_module.load_dataset_exercise_cases = lambda: [
        memory_module.DatasetExerciseCase(
            "case-1",
            normalized_section,
            "Analyse",
            "suites numeriques",
            "2023",
            "Suite numerique avec limite et forme explicite",
            "Sol",
            "Ans",
            "Exercice",
        ),
        memory_module.DatasetExerciseCase(
            "case-2",
            normalized_section,
            "Probabilites",
            "variables aleatoires",
            "2023",
            "Exercice de probabilites avec loi discrete",
            "Sol",
            "Ans",
            "Exercice",
        ),
    ]
    memory_module._idf_scores.cache_clear()
    memory_module._tfidf_vector.cache_clear()
    try:
        cases = memory_module.retrieve_dataset_cases(
            section="Mathematiques",
            topic="Analyse",
            subtopic="suites numeriques",
            top_k=2,
        )
        assert cases[0].case_id == "case-1"
    finally:
        memory_module.load_dataset_exercise_cases = original_loader
        memory_module._idf_scores.cache_clear()
        memory_module._tfidf_vector.cache_clear()


def test_memory_positive_prior_rewards_successful_case() -> None:
    _reset_memory_state()
    normalized_section = memory_module.normalize_section_label("Mathematiques")
    st.session_state.generation_outcome_memory = [
        {
            "section": normalized_section,
            "topic": "Analyse",
            "subtopic": "suites numeriques",
            "retrieved_case_ids": ["case-2"],
            "is_true_llm_generation": True,
            "final_display_decision": "presented",
        }
    ]
    original_loader = memory_module.load_dataset_exercise_cases
    memory_module.load_dataset_exercise_cases = lambda: [
        memory_module.DatasetExerciseCase(
            "case-1",
            normalized_section,
            "Analyse",
            "suites numeriques",
            "2023",
            "Suite numerique calculer limite",
            "Sol",
            "Ans",
            "Exercice",
        ),
        memory_module.DatasetExerciseCase(
            "case-2",
            normalized_section,
            "Analyse",
            "suites numeriques",
            "2023",
            "Suite numerique calculer limite",
            "Sol",
            "Ans",
            "Exercice",
        ),
    ]
    memory_module._idf_scores.cache_clear()
    memory_module._tfidf_vector.cache_clear()
    try:
        cases = memory_module.retrieve_dataset_cases(
            section="Mathematiques",
            topic="Analyse",
            subtopic="suites numeriques",
            top_k=2,
        )
        assert cases[0].case_id == "case-2"
    finally:
        memory_module.load_dataset_exercise_cases = original_loader
        memory_module._idf_scores.cache_clear()
        memory_module._tfidf_vector.cache_clear()
        _reset_memory_state()


def test_memory_negative_patterns_are_ranked() -> None:
    _reset_memory_state()
    st.session_state.generation_outcome_memory = [
        {
            "section": "Mathematiques",
            "topic": "Probabilites",
            "subtopic": "variables aleatoires",
            "failure_categories": ["probability_inconsistent", "context_only"],
        },
        {
            "section": "Mathematiques",
            "topic": "Probabilites",
            "subtopic": "variables aleatoires",
            "failure_categories": ["probability_inconsistent"],
        },
    ]
    patterns = memory_module.get_negative_memory_patterns("Mathematiques", "Probabilites", "variables aleatoires")
    assert patterns
    assert patterns[0]["pattern"] == "probability_inconsistent"
    _reset_memory_state()


def test_memory_similarity_guard_rejects_near_copy() -> None:
    cases = [
        memory_module.DatasetExerciseCase(
            "case-1",
            "Mathematiques",
            "Analyse",
            "suites numeriques",
            "2023",
            "On considere la suite U_n definie par U_{n+1}=2U_n. 1) Calculer U_1. 2) Determiner la limite.",
            "Sol",
            "Ans",
            "Exercice",
        )
    ]
    similar, case_id, score = memory_module.find_too_similar_source_case(
        "On considere la suite U_n definie par U_{n+1}=2U_n. 1) Calculer U_1. 2) Determiner la limite.",
        cases,
    )
    assert similar is True
    assert case_id == "case-1"
    assert score > 0.92


def test_retry_controller_context_only_triggers_strict_schema() -> None:
    controller = GenerationRetryController()
    controller.register_failure(1, ["L'enonce presente seulement un contexte sans question ni consigne explicite."])
    assert controller.next_strategy() == "strict_schema_generation"


def test_retry_controller_invalid_json_twice_triggers_simple_generation() -> None:
    controller = GenerationRetryController()
    controller.register_failure(1, ["Le modele n'a pas renvoye un JSON exploitable pour l'exercice."])
    controller.register_failure(2, ["Le JSON est invalide apres extraction."])
    assert controller.next_strategy() == "simple_exercise_generation"


def test_retry_controller_repeated_same_failure_switches_strategy() -> None:
    controller = GenerationRetryController()
    controller.register_failure(1, ["Erreur inconnue."])
    controller.register_failure(2, ["Erreur inconnue."])
    controller.register_failure(3, ["Erreur inconnue."])
    assert controller.next_strategy() == "topic_template_guided_generation"


def test_final_gate_blocks_malformed_format_even_if_judge_approved() -> None:
    record = _base_record()
    record["prompt"] = "On a frace^x"
    allowed, reasons = can_present_exercise(record)
    assert not allowed
    assert reasons


def test_final_gate_blocks_missing_expected_answer() -> None:
    record = _base_record()
    record["display_answer"] = ""
    allowed, reasons = can_present_exercise(record)
    assert not allowed
    assert any("reponse attendue" in reason.lower() for reason in reasons)


def test_final_gate_allows_valid_llm_exercise() -> None:
    allowed, reasons = can_present_exercise(_base_record())
    assert allowed, reasons


def test_final_gate_allows_valid_llm_probability_exercise() -> None:
    record = _base_record()
    probability = _base_probability_exercise()
    record.update(
        {
            "title": probability["title"],
            "context": probability["context"],
            "questions": probability["questions"],
            "prompt": probability["prompt"],
            "instruction": probability["instruction"],
            "hidden_solution": probability["hidden_solution"],
            "display_answer": probability["display_answer"],
            "topic": "Probabilites",
            "subtopic": "variables aleatoires, esperance et variance",
            "learning_objective": probability["learning_objective"],
            "local_validation_checks": {"probability": {"status": "passed"}},
        }
    )
    allowed, reasons = can_present_exercise(record)
    assert allowed, reasons


def test_final_gate_blocks_context_only_statement() -> None:
    record = _base_record()
    record["questions"] = []
    record["prompt"] = "Une urne contient trois boules rouges et deux boules bleues. On tire deux boules."
    allowed, reasons = can_present_exercise(record)
    assert not allowed
    assert any("question" in reason.lower() for reason in reasons)


def test_final_gate_blocks_dataset_fallback_without_demo_mode() -> None:
    record = _base_record()
    record["generation_backend"] = "trusted-dataset-demo"
    record["is_true_llm_generation"] = False
    record["display_source_category"] = "demo_dataset"
    record["demo_mode_used"] = False
    allowed, reasons = can_present_exercise(record)
    assert not allowed
    assert any("demonstration" in reason.lower() for reason in reasons)


def test_final_gate_blocks_expected_answer_mismatch() -> None:
    record = _base_record()
    record["local_validation_flag"] = "wrong"
    record["local_validation_issues"] = ["Expected_answer contradictoire avec la solution."]
    allowed, reasons = can_present_exercise(record)
    assert not allowed
    assert any("contradic" in reason.lower() for reason in reasons)


def test_json_parser_reports_direct_fenced_balanced_and_trailing_comma() -> None:
    direct = parse_json_object_detailed('{"ok": true}')
    assert direct.ok and direct.extraction_method == "direct"

    fenced = parse_json_object_detailed('```json\n{"ok": true}\n```')
    assert fenced.ok and fenced.extraction_method == "fenced"

    balanced = parse_json_object_detailed('Voici la sortie: {"ok": true, "message": "hello"} merci.')
    assert balanced.ok and balanced.extraction_method == "balanced_object"

    repaired = parse_json_object_detailed('{"ok": true,}')
    assert repaired.ok and repaired.extraction_method == "trailing_comma_repair"

    failed = parse_json_object_detailed("aucun json ici")
    assert not failed.ok and failed.error


def test_retry_categories_keep_connection_error_separate_from_invalid_json() -> None:
    categories = classify_failure_categories(["connection_error", "Connection error."])
    assert "connection_error" in categories
    assert "invalid_json" not in categories


def test_memory_retrieval_returns_known_topic_cases() -> None:
    matches = memory_module.retrieve_dataset_case_matches(
        section="Sciences expérimentales",
        topic="Statistiques",
        subtopic="séries à deux caractères, régression et corrélation",
        top_k=3,
    )
    assert matches
    assert matches[0][0] >= 0
    assert matches[0][1].instruction


def test_openrouter_response_format_fallback_simulated() -> None:
    from frontend.utils import openrouter_client as client_module

    original_settings = client_module.get_openrouter_settings
    original_call_once = client_module._call_openrouter_once

    class FakeSettings:
        api_key = "x"
        base_url = "https://example.test"
        exercise_model = "primary"
        exercise_model_primary = "primary"
        exercise_model_fallback = ""
        judge_model = "judge"
        validator_model = "validator"
        tutor_model = "tutor"
        site_url = "http://localhost"
        app_name = "MathTutorAI"
        allow_dataset_demo = False
        response_mode = "auto"

    calls: list[str] = []

    def fake_call_once(**kwargs):
        mode = kwargs["response_format_mode"]
        calls.append(mode)
        if mode == "json_schema":
            return OpenRouterCallResult(
                ok=False,
                model=kwargs["model"],
                response_format_mode=mode,
                error_type="unsupported_response_format",
                error_message="unsupported",
            )
        return OpenRouterCallResult(
            ok=True,
            content='{"ok": true}',
            raw_response_preview='{"ok": true}',
            model=kwargs["model"],
            response_format_mode=mode,
        )

    try:
        client_module.get_openrouter_settings = lambda: FakeSettings()
        client_module._call_openrouter_once = fake_call_once
        result = call_openrouter_chat(
            messages=[{"role": "user", "content": "json"}],
            model="primary",
            temperature=0,
            top_p=0.1,
            max_tokens=20,
            purpose="health",
        )
    finally:
        client_module.get_openrouter_settings = original_settings
        client_module._call_openrouter_once = original_call_once

    assert result.ok
    assert calls[:2] == ["json_schema", "json_object"]
