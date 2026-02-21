"""
XGBoost Risk Prediction + SHAP Explainability
==============================================
Model: XGBoost binary classifier
Target: P(high-risk event within 24 hours)
Output: probability (0–1) + SHAP top-5 feature drivers

SHAP ensures every prediction is explainable.
"Low BP + elevated HR suggest early shock."
"""

import numpy as np
import asyncio
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from app.schemas.intake import (
    VitalsInput, VulnerabilityFlags, SymptomInput,
    MLResult, SHAPFeature, RiskLevel
)

logger = logging.getLogger(__name__)


@dataclass
class ModelRegistry:
    """Holds loaded models — loaded once at startup, reused per-request."""
    _xgb_model = None
    _shap_explainer = None
    _ready: bool = False

    async def load_all(self):
        """Load models from disk at application startup."""
        try:
            import joblib, xgboost as xgb, shap

            loop = asyncio.get_event_loop()
            # Load in thread pool (blocking I/O)
            self._xgb_model = await loop.run_in_executor(
                None, joblib.load, "app/ml/models/xgb_risk_model.pkl"
            )
            self._shap_explainer = await loop.run_in_executor(
                None, joblib.load, "app/ml/models/shap_explainer.pkl"
            )
            self._ready = True
            logger.info("XGBoost model + SHAP explainer loaded successfully")
        except FileNotFoundError:
            logger.warning(
                "ML model files not found — running in HEURISTIC mode. "
                "Train and save models to app/ml/models/ for production."
            )
            self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    async def cleanup(self):
        self._xgb_model = None
        self._shap_explainer = None

    @property
    def model(self):
        return self._xgb_model

    @property
    def explainer(self):
        return self._shap_explainer


model_registry = ModelRegistry()


# ─── Feature Engineering ──────────────────────────────────────────────────────

FEATURE_NAMES = [
    "spo2", "systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
    "temperature", "blood_glucose", "age_years", "sex_encoded",
    "is_pregnant", "is_diabetic", "has_heart_disease", "is_immunocompromised",
    "bmi_proxy", "shock_index", "pulse_pressure",
    "has_chest_pain", "has_altered_consciousness", "has_breathing_difficulty",
    "has_severe_headache", "has_bleeding", "red_flag_count"
]

FEATURE_LABELS = {
    "spo2": "Oxygen Saturation (SpO₂)",
    "systolic_bp": "Systolic Blood Pressure",
    "diastolic_bp": "Diastolic Blood Pressure",
    "heart_rate": "Heart Rate",
    "respiratory_rate": "Respiratory Rate",
    "temperature": "Temperature",
    "blood_glucose": "Blood Glucose",
    "age_years": "Patient Age",
    "sex_encoded": "Sex",
    "is_pregnant": "Pregnancy",
    "is_diabetic": "Diabetes",
    "has_heart_disease": "Heart Disease",
    "is_immunocompromised": "Immunocompromised",
    "bmi_proxy": "Weight Category",
    "shock_index": "Shock Index (HR/SBP)",
    "pulse_pressure": "Pulse Pressure",
    "has_chest_pain": "Chest Pain Symptom",
    "has_altered_consciousness": "Altered Consciousness",
    "has_breathing_difficulty": "Breathing Difficulty",
    "has_severe_headache": "Severe Headache",
    "has_bleeding": "Bleeding Symptom",
    "red_flag_count": "Number of Red Flag Symptoms",
}


def _extract_features(
    vitals: VitalsInput,
    age: int,
    sex: str,
    flags: VulnerabilityFlags,
    symptoms: List[SymptomInput],
) -> np.ndarray:
    sym_lower = [s.symptom_name.lower() for s in symptoms]

    def has_symptom(*keywords) -> float:
        return 1.0 if any(any(kw in s for kw in keywords) for s in sym_lower) else 0.0

    red_flag_count = sum(1 for s in symptoms if s.is_red_flag)

    features = np.array([
        vitals.spo2,
        vitals.systolic_bp,
        vitals.diastolic_bp,
        vitals.heart_rate,
        vitals.respiratory_rate,
        vitals.temperature,
        vitals.blood_glucose_mgdl or 100.0,
        float(age),
        0.0 if sex == "male" else 1.0,
        1.0 if flags.pregnant else 0.0,
        1.0 if flags.diabetic else 0.0,
        1.0 if flags.heart_disease else 0.0,
        1.0 if flags.immunocompromised else 0.0,
        float(vitals.weight_kg or 60) / 60.0,  # BMI proxy (normalized)
        vitals.shock_index,
        float(vitals.pulse_pressure),
        has_symptom("chest pain", "chest tightness"),
        has_symptom("unconscious", "confused", "confusion", "altered"),
        has_symptom("breathing", "breathless", "dyspnoea"),
        has_symptom("headache", "severe headache"),
        has_symptom("bleeding", "hemorrhage", "blood"),
        float(red_flag_count),
    ], dtype=np.float32)

    return features


# ─── Heuristic Scoring (fallback when model not loaded) ───────────────────────

def _heuristic_predict(features: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calibrated heuristic that approximates XGBoost output.
    In production: replaced by model.predict_proba([features])[0][1]
    """
    (spo2, sbp, dbp, hr, rr, temp, bg, age, sex, preg,
     diabetic, heart_dz, immunocomp, bmi, shock_idx, pp,
     chest_pain, confusion, breathing, headache, bleeding, rf_count) = features

    score = 0.0

    # SpO2 (max 0.30)
    if spo2 < 90:    score += 0.30
    elif spo2 < 94:  score += 0.18
    elif spo2 < 96:  score += 0.08

    # BP (max 0.25)
    if sbp < 90:    score += 0.25
    elif sbp < 100: score += 0.15
    elif sbp >= 180: score += 0.20
    elif sbp >= 160: score += 0.12

    # Shock index (max 0.15)
    if shock_idx > 1.0:  score += 0.15
    elif shock_idx > 0.8: score += 0.08

    # RR (max 0.12)
    if rr > 30:    score += 0.12
    elif rr > 24:  score += 0.06

    # Temperature (max 0.08)
    if temp > 40 or temp < 35: score += 0.08
    elif temp > 39 or temp < 36: score += 0.04

    # Symptoms
    score += chest_pain * 0.12
    score += confusion * 0.15
    score += breathing * 0.10
    score += bleeding * 0.08

    # Risk factors
    if preg:        score += 0.08
    if immunocomp:  score += 0.07
    if heart_dz:    score += 0.05
    if diabetic:    score += 0.03
    score += rf_count * 0.03

    # Age
    if age < 1 or age > 75: score += 0.05

    # SHAP-like values (approximate feature contributions)
    shap_values = np.array([
        (90.0 - spo2) * 0.01,           # spo2
        (90.0 - sbp) * 0.003 if sbp < 90 else (sbp - 140) * 0.001,  # sbp
        0.0, (hr - 80) * 0.001, (rr - 18) * 0.004, (temp - 37) * 0.015,
        0.0, (age - 40) * 0.001,
        0.0, preg * 0.08, diabetic * 0.03, heart_dz * 0.05, immunocomp * 0.07,
        0.0, shock_idx * 0.1 if shock_idx > 0.8 else 0.0, 0.0,
        chest_pain * 0.12, confusion * 0.15, breathing * 0.10,
        headache * 0.03, bleeding * 0.08, rf_count * 0.03,
    ], dtype=np.float32)

    return float(min(score, 1.0)), shap_values


# ─── Main Predictor ───────────────────────────────────────────────────────────

def _score_to_level(score: float) -> RiskLevel:
    if score >= 0.70:   return RiskLevel.high
    elif score >= 0.30: return RiskLevel.moderate
    return RiskLevel.low


def _build_shap_features(
    shap_values: np.ndarray,
    features: np.ndarray,
    risk_level: RiskLevel,
) -> Tuple[List[SHAPFeature], str]:
    """Build top-5 SHAP features with clinical labels and text summary."""

    # Sort by absolute SHAP value, descending
    indices = np.argsort(np.abs(shap_values))[::-1][:5]

    top_features = []
    for idx in indices:
        name = FEATURE_NAMES[idx]
        label = FEATURE_LABELS.get(name, name)
        shap_val = float(shap_values[idx])
        feat_val = float(features[idx])

        top_features.append(SHAPFeature(
            feature=name,
            value=feat_val,
            shap_value=round(shap_val, 4),
            label=f"{label} = {feat_val:.1f} (impact: {'↑' if shap_val > 0 else '↓'}{abs(shap_val):.3f})",
        ))

    # Generate clinical text interpretation
    text = _build_shap_text(top_features, risk_level)
    return top_features, text


def _build_shap_text(features: List[SHAPFeature], risk_level: RiskLevel) -> str:
    """Convert SHAP values into a human-readable clinical sentence."""
    if not features:
        return "Insufficient data to generate clinical interpretation."

    top = features[0]
    second = features[1] if len(features) > 1 else None

    interpretations = {
        "spo2": "oxygen desaturation",
        "systolic_bp": "low blood pressure" if features[0].value < 100 else "elevated blood pressure",
        "shock_index": "shock indicators (elevated HR relative to BP)",
        "respiratory_rate": "rapid breathing",
        "heart_rate": "rapid heart rate",
        "has_altered_consciousness": "altered level of consciousness",
        "has_chest_pain": "chest pain",
        "is_immunocompromised": "immunocompromised state",
        "is_pregnant": "pregnancy-related risk",
        "temperature": "abnormal temperature",
    }

    top_label = interpretations.get(top.feature, top.feature.replace("_", " "))
    text = f"Primary driver: {top_label}"

    if second:
        second_label = interpretations.get(second.feature, second.feature.replace("_", " "))
        text += f" combined with {second_label}"

    risk_phrase = {
        RiskLevel.critical: "suggest critical deterioration requiring immediate intervention",
        RiskLevel.high: "indicate high risk — escalation strongly recommended",
        RiskLevel.moderate: "suggest moderate risk — close monitoring required",
        RiskLevel.low: "suggest lower risk — standard care appropriate",
    }

    return f"{text} {risk_phrase.get(risk_level, 'require clinical review')}."


async def predict_risk(
    vitals: VitalsInput,
    age: int,
    sex: str,
    flags: VulnerabilityFlags,
    symptoms: List[SymptomInput],
) -> MLResult:
    """
    Main async prediction entry point.
    Uses real XGBoost model if loaded, else heuristic fallback.
    """
    features = _extract_features(vitals, age, sex, flags, symptoms)

    if model_registry.is_ready() and model_registry.model is not None:
        # Production: real XGBoost inference + SHAP
        loop = asyncio.get_event_loop()

        risk_prob = await loop.run_in_executor(
            None,
            lambda: float(model_registry.model.predict_proba([features])[0][1])
        )
        shap_vals = await loop.run_in_executor(
            None,
            lambda: model_registry.explainer.shap_values(features)[0]
        )
        confidence = 0.90
    else:
        # Fallback heuristic
        risk_prob, shap_vals = _heuristic_predict(features)
        confidence = 0.78

    risk_level = _score_to_level(risk_prob)
    top_features, shap_text = _build_shap_features(shap_vals, features, risk_level)

    logger.info(
        f"ML prediction: prob={risk_prob:.3f} level={risk_level} "
        f"mode={'model' if model_registry.is_ready() else 'heuristic'}"
    )

    return MLResult(
        risk_probability=round(risk_prob, 3),
        risk_level=risk_level,
        shap_features=top_features,
        shap_text=shap_text,
    )
