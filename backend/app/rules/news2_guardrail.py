"""
NEWS2-Style Rule-Based Safety Guardrail
========================================
Rules ALWAYS run first and ALWAYS override ML output when triggered.
Clinical priority: SAFETY over probability.

Reference: Royal College of Physicians NEWS2 (2017) + NHM India PHC guidelines.
"""

from typing import List, Tuple
from dataclasses import dataclass, field
from app.schemas.intake import VitalsInput, VulnerabilityFlags, SymptomInput, RiskLevel
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    triggered: bool = False
    risk_level: RiskLevel = RiskLevel.low
    reasons: List[str] = field(default_factory=list)
    override_ml: bool = False

    @property
    def is_critical(self) -> bool:
        return self.risk_level == RiskLevel.critical


class NEWS2Guardrail:
    """
    Deterministic clinical safety guardrail.
    Any CRITICAL trigger immediately sets risk = CRITICAL and bypasses ML.
    """

    # ─── CRITICAL Thresholds (Immediate Override) ─────────────────────────────

    CRITICAL_RULES = [
        {
            "check": lambda v: v.spo2 < settings.SPO2_CRITICAL,
            "reason": lambda v: f"Critical oxygen desaturation: SpO₂ = {v.spo2}% (threshold < {settings.SPO2_CRITICAL}%)",
            "flag": "SPO2_CRITICAL",
        },
        {
            "check": lambda v: v.systolic_bp < settings.SBP_CRITICAL,
            "reason": lambda v: f"Severe hypotension/shock risk: SBP = {v.systolic_bp} mmHg (threshold < {settings.SBP_CRITICAL} mmHg)",
            "flag": "SBP_CRITICAL",
        },
        {
            "check": lambda v: v.respiratory_rate > settings.RR_CRITICAL,
            "reason": lambda v: f"Severe respiratory distress: RR = {v.respiratory_rate}/min (threshold > {settings.RR_CRITICAL}/min)",
            "flag": "RR_CRITICAL",
        },
        {
            "check": lambda v: v.temperature > settings.TEMP_CRITICAL,
            "reason": lambda v: f"Hyperpyrexia: Temp = {v.temperature}°C (threshold > {settings.TEMP_CRITICAL}°C)",
            "flag": "TEMP_CRITICAL",
        },
        {
            "check": lambda v: v.temperature < 35.0,
            "reason": lambda v: f"Hypothermia: Temp = {v.temperature}°C",
            "flag": "TEMP_HYPOTHERMIA",
        },
        {
            "check": lambda v: v.systolic_bp >= 180,
            "reason": lambda v: f"Hypertensive crisis: BP = {v.systolic_bp}/{v.diastolic_bp} mmHg",
            "flag": "HTN_CRISIS",
        },
        {
            "check": lambda v: v.blood_glucose_mgdl is not None and v.blood_glucose_mgdl < 54,
            "reason": lambda v: f"Severe hypoglycaemia: BG = {v.blood_glucose_mgdl} mg/dL",
            "flag": "HYPOGLYCEMIA_SEVERE",
        },
        {
            "check": lambda v: v.gcs_score is not None and v.gcs_score <= 8,
            "reason": lambda v: f"Severely altered consciousness: GCS = {v.gcs_score}",
            "flag": "GCS_CRITICAL",
        },
    ]

    # ─── HIGH Warning Thresholds ──────────────────────────────────────────────

    HIGH_RULES = [
        {
            "check": lambda v: v.spo2 < 94,
            "reason": lambda v: f"Low oxygen saturation: SpO₂ = {v.spo2}%",
        },
        {
            "check": lambda v: v.systolic_bp < 100,
            "reason": lambda v: f"Low systolic BP: {v.systolic_bp} mmHg",
        },
        {
            "check": lambda v: v.heart_rate > 120,
            "reason": lambda v: f"Significant tachycardia: HR = {v.heart_rate} bpm",
        },
        {
            "check": lambda v: v.heart_rate < 45,
            "reason": lambda v: f"Significant bradycardia: HR = {v.heart_rate} bpm",
        },
        {
            "check": lambda v: v.respiratory_rate > 24,
            "reason": lambda v: f"Tachypnoea: RR = {v.respiratory_rate}/min",
        },
        {
            "check": lambda v: v.temperature >= 39.0,
            "reason": lambda v: f"High fever: Temp = {v.temperature}°C",
        },
        {
            "check": lambda v: v.blood_glucose_mgdl is not None and v.blood_glucose_mgdl > 400,
            "reason": lambda v: f"Severe hyperglycaemia: BG = {v.blood_glucose_mgdl} mg/dL",
        },
        {
            "check": lambda v: v.shock_index > 1.0,
            "reason": lambda v: f"Elevated shock index: {v.shock_index:.2f} (HR/SBP)",
        },
    ]

    # ─── Symptom-based Critical Overrides ────────────────────────────────────

    CRITICAL_SYMPTOM_KEYWORDS = [
        "chest pain", "unconscious", "seizure", "convulsion",
        "stroke", "paralysis", "sudden vision loss", "coughing blood",
        "vomiting blood", "stopped breathing", "cardiac arrest",
        "severe abdominal pain", "stiff neck"
    ]

    # ─── Pregnancy Danger Signs ───────────────────────────────────────────────

    OBSTETRIC_CRITICAL_KEYWORDS = [
        "bleeding", "vaginal bleeding", "antepartum hemorrhage",
        "severe headache", "visual disturbance", "blurred vision",
        "epigastric pain", "fits", "convulsion", "eclampsia",
        "reduced fetal movement", "leaking", "cord prolapse"
    ]

    def evaluate(
        self,
        vitals: VitalsInput,
        flags: VulnerabilityFlags,
        symptoms: List[SymptomInput],
    ) -> RuleResult:
        result = RuleResult()

        # 1. Check CRITICAL vital thresholds
        for rule in self.CRITICAL_RULES:
            if rule["check"](vitals):
                result.triggered = True
                result.override_ml = True
                result.risk_level = RiskLevel.critical
                result.reasons.append(rule["reason"](vitals))
                logger.warning(f"NEWS2 CRITICAL triggered: {rule['flag']}")

        # 2. Check HIGH thresholds (only if not already CRITICAL)
        if not result.triggered:
            high_reasons = []
            for rule in self.HIGH_RULES:
                if rule["check"](vitals):
                    high_reasons.append(rule["reason"](vitals))

            if len(high_reasons) >= 2:
                # Two or more HIGH flags → escalate to HIGH
                result.triggered = True
                result.risk_level = RiskLevel.high
                result.reasons.extend(high_reasons)
            elif len(high_reasons) == 1:
                result.triggered = True
                result.risk_level = RiskLevel.moderate
                result.reasons.extend(high_reasons)

        # 3. Symptom red flags
        sym_names = [s.symptom_name.lower() for s in symptoms]
        for keyword in self.CRITICAL_SYMPTOM_KEYWORDS:
            if any(keyword in s for s in sym_names):
                result.triggered = True
                result.override_ml = True
                result.risk_level = RiskLevel.critical
                result.reasons.append(f"Critical symptom reported: '{keyword}'")

        # 4. Obstetric danger signs
        if flags.pregnant:
            for keyword in self.OBSTETRIC_CRITICAL_KEYWORDS:
                if any(keyword in s for s in sym_names):
                    result.triggered = True
                    result.override_ml = True
                    result.risk_level = RiskLevel.critical
                    result.reasons.append(f"Obstetric danger sign: '{keyword}'")

            # Pregnancy + hypertension check
            if vitals.systolic_bp >= 140 or vitals.diastolic_bp >= 90:
                result.triggered = True
                result.risk_level = RiskLevel.critical
                result.override_ml = True
                result.reasons.append(
                    f"Pregnancy hypertension (possible preeclampsia): "
                    f"BP {vitals.systolic_bp}/{vitals.diastolic_bp} mmHg"
                )

        # 5. Immunocompromised + fever → HIGH concern
        if flags.immunocompromised and vitals.temperature >= 38.0:
            if result.risk_level not in (RiskLevel.critical, RiskLevel.high):
                result.triggered = True
                result.risk_level = RiskLevel.high
                result.reasons.append(
                    f"Immunocompromised patient with fever: {vitals.temperature}°C — "
                    "sepsis must be excluded"
                )

        return result


# Singleton
news2_guardrail = NEWS2Guardrail()
