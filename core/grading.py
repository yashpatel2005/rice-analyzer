"""
Phase 10 – Final Verdict

A configurable rule-based decision engine that combines all computed
measurements and statistical analyses to produce a transparent,
explainable final grade.

Grades: Premium → Grade A → Grade B → Grade C → Reject

The engine checks each grade's thresholds in order and returns the
first match.  Every criterion is logged so the verdict is fully
explainable.
"""

from typing import Dict, Any, Optional

import config


class GradingEngine:
    """Transparent, rule-based rice quality grading."""

    GRADE_ORDER = ["premium", "grade_a", "grade_b", "grade_c", "reject"]
    GRADE_LABELS = {
        "premium": "Premium",
        "grade_a": "Grade A",
        "grade_b": "Grade B",
        "grade_c": "Grade C",
        "reject": "Reject",
    }

    def __init__(self, rules: Optional[dict] = None):
        self.rules = rules or config.GRADING_RULES

    # ------------------------------------------------------------------
    # Evaluate one grade's rules
    # ------------------------------------------------------------------
    def _check_grade(self, grade: str, quality: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
        """Check whether the sample meets all thresholds for *grade*."""
        rule = self.rules[grade]
        checks = []

        # Broken grain percentage
        broken = quality.get("broken_pct", 100)
        checks.append({
            "criterion": "broken_pct",
            "value": broken,
            "threshold": rule["max_broken_pct"],
            "operator": "<=",
            "passed": broken <= rule["max_broken_pct"],
        })

        # Uniformity
        uniformity = quality.get("uniformity_index", 0)
        checks.append({
            "criterion": "uniformity_index",
            "value": uniformity,
            "threshold": rule["min_uniformity"],
            "operator": ">=",
            "passed": uniformity >= rule["min_uniformity"],
        })

        # Abnormal percentage
        abnormal = quality.get("abnormal_pct", 100)
        checks.append({
            "criterion": "abnormal_pct",
            "value": abnormal,
            "threshold": rule["max_abnormal_pct"],
            "operator": "<=",
            "passed": abnormal <= rule["max_abnormal_pct"],
        })

        # CV of length
        cv_len = quality.get("cv_length", 100)
        checks.append({
            "criterion": "cv_length",
            "value": cv_len,
            "threshold": rule["max_cv_length"],
            "operator": "<=",
            "passed": cv_len <= rule["max_cv_length"],
        })

        all_passed = all(c["passed"] for c in checks)
        return {
            "grade": grade,
            "label": self.GRADE_LABELS[grade],
            "all_passed": all_passed,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # Main grading
    # ------------------------------------------------------------------
    def grade(
        self, quality: Dict[str, Any], stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate all grades in order and return the first match plus
        the full evaluation log for explainability.
        """
        all_evaluations = []
        selected_grade = "reject"  # default worst case
        selected = None

        for grade in self.GRADE_ORDER:
            evaluation = self._check_grade(grade, quality, stats)
            all_evaluations.append(evaluation)
            if evaluation["all_passed"] and selected is None:
                selected = evaluation
                selected_grade = grade

        # Build explanation
        explanation = self._build_explanation(selected, quality)

        return {
            "grade": self.GRADE_LABELS[selected_grade],
            "grade_key": selected_grade,
            "explanation": explanation,
            "criteria_results": selected["checks"] if selected else [],
            "all_evaluations": all_evaluations,
        }

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------
    @staticmethod
    def _build_explanation(selected: Optional[Dict[str, Any]], quality: Dict[str, Any]) -> str:
        if selected is None:
            return "Sample does not meet minimum quality criteria."
        grade = selected["label"]
        parts = [f"Sample classified as **{grade}**."]

        for check in selected["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            parts.append(
                f"  • {check['criterion']}: {check['value']:.2f} "
                f"{check['operator']} {check['threshold']:.2f} → {status}"
            )

        parts.append(f"  • Total grains: {quality.get('total_grains', 0)}")
        parts.append(f"  • Broken: {quality.get('broken_pct', 0):.2f}%")
        parts.append(f"  • Uniformity: {quality.get('uniformity_index', 0):.2f}%")
        parts.append(f"  • Abnormal: {quality.get('abnormal_pct', 0):.2f}%")

        return "\n".join(parts)
