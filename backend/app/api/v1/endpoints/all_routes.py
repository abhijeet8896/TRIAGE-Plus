"""
API Endpoints — Auth, Patient Intake, Hybrid Analysis, Escalation, Specialist
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FILE: app/api/v1/endpoints/auth.py
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

auth_router = APIRouter(prefix="/auth")

# ── POST /auth/login ──────────────────────────────────────────────────────────

"""
POST /api/v1/auth/login

Request:
    { "username": "phw@phc.in", "password": "secure123" }

Response:
    {
        "access_token": "eyJhbGc...",
        "token_type": "bearer",
        "role": "phw",
        "full_name": "Priya Sharma"
    }
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FILE: app/api/v1/endpoints/analyze.py   ← CORE ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, require_phw
from app.schemas.intake import (
    PatientIntakeRequest, RiskAssessmentResponse,
    RuleEngineResult, MLResult, RiskLevel
)
from app.rules.news2_guardrail import news2_guardrail
from app.rules.medication_engine import medication_engine
from app.ml.risk_predictor import predict_risk

analyze_router = APIRouter(prefix="/analyze")


@analyze_router.post(
    "/risk",
    response_model=RiskAssessmentResponse,
    summary="Hybrid AI risk assessment",
    description=(
        "Runs the full hybrid decision pipeline: "
        "(1) NEWS2 rule guardrail → (2) XGBoost ML → (3) Medication engine → "
        "(4) Aggregate decision. Rule-based results always override ML."
    )
)
async def analyze_risk(
    payload: PatientIntakeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(require_phw),
):
    """
    ==========================================================================
    DECISION PIPELINE:

    1. Run NEWS2 rule engine + medication engine + ML in PARALLEL
    2. Aggregate:
        a. If rule engine CRITICAL → final = CRITICAL, skip ML tier
        b. If medication override → force escalation
        c. Else → use ML risk level
    3. Determine recommendation text
    4. Store assessment in DB
    5. Return full response

    Sample Request:
    {
        "patient_name": "Meena Devi",
        "age": 32,
        "sex": "female",
        "vulnerability_flags": { "pregnant": true },
        "vitals": {
            "systolic_bp": 155, "diastolic_bp": 100,
            "heart_rate": 98, "respiratory_rate": 20,
            "spo2": 97.0, "temperature": 37.2
        },
        "medications": [{"drug_name": "Iron supplement"}, {"drug_name": "Folic acid"}],
        "symptoms": [
            {"symptom_name": "severe headache", "is_red_flag": true},
            {"symptom_name": "blurred vision", "is_red_flag": true}
        ],
        "chief_complaint": "Severe headache and visual disturbance at 34 weeks pregnancy"
    }

    Sample Response:
    {
        "assessment_id": "uuid",
        "case_id": "uuid",
        "final_risk_level": "critical",
        "final_risk_score": 0.91,
        "rule_engine": {
            "triggered": true,
            "risk_level": "critical",
            "reasons": ["Pregnancy hypertension (possible preeclampsia): BP 155/100 mmHg"],
            "override_ml": true
        },
        "ml_result": {
            "risk_probability": 0.84,
            "risk_level": "high",
            "shap_features": [...],
            "shap_text": "Pregnancy + elevated BP + severe headache suggest preeclampsia — high risk."
        },
        "med_warnings": [],
        "recommendation": "IMMEDIATE ESCALATION REQUIRED. Signs consistent with preeclampsia. ...",
        "escalation_suggested": true,
        "assessed_at": "2025-01-15T10:30:00Z"
    }
    ==========================================================================
    """

    # ── Step 1: Parallel Evaluation ─────────────────────────────────────────

    rule_task = asyncio.create_task(
        asyncio.coroutine(lambda: news2_guardrail.evaluate(
            payload.vitals, payload.vulnerability_flags, payload.symptoms
        ))()
    )
    ml_task = asyncio.create_task(
        predict_risk(
            payload.vitals, payload.age, payload.sex,
            payload.vulnerability_flags, payload.symptoms
        )
    )
    med_task = asyncio.create_task(
        asyncio.coroutine(lambda: medication_engine(
            payload.medications, payload.symptoms, payload.vulnerability_flags
        ))()
    )

    rule_result, ml_result, (med_warnings, med_override) = await asyncio.gather(
        rule_task, ml_task, med_task
    )

    # ── Step 2: Aggregate Decision ───────────────────────────────────────────

    # Rule engine has absolute priority
    if rule_result.override_ml:
        final_risk = rule_result.risk_level
        final_score = 0.95 if final_risk == RiskLevel.critical else 0.78
    elif med_override:
        final_risk = RiskLevel.critical
        final_score = 0.90
    else:
        # Use ML result as primary
        final_risk = ml_result.risk_level
        final_score = ml_result.risk_probability
        # If rule engine flagged (but not override), take max
        if rule_result.triggered:
            rule_scores = {
                RiskLevel.critical: 4, RiskLevel.high: 3,
                RiskLevel.moderate: 2, RiskLevel.low: 1
            }
            if rule_scores.get(rule_result.risk_level, 0) > rule_scores.get(final_risk, 0):
                final_risk = rule_result.risk_level

    escalation_suggested = final_risk in (RiskLevel.critical, RiskLevel.high) or med_override

    # ── Step 3: Recommendation Text ──────────────────────────────────────────

    recommendation = _build_recommendation(
        final_risk, rule_result.reasons, med_warnings, ml_result.shap_text,
        med_override, payload.vulnerability_flags
    )

    # ── Step 4: Build Response ───────────────────────────────────────────────

    assessment_id = str(uuid.uuid4())
    case_id = str(uuid.uuid4())  # In real implementation: get or create case

    return RiskAssessmentResponse(
        assessment_id=assessment_id,
        case_id=case_id,
        final_risk_level=final_risk,
        final_risk_score=round(final_score, 3),
        rule_engine=RuleEngineResult(
            triggered=rule_result.triggered,
            risk_level=rule_result.risk_level,
            reasons=rule_result.reasons,
            override_ml=rule_result.override_ml,
        ),
        ml_result=ml_result,
        med_warnings=med_warnings,
        recommendation=recommendation,
        escalation_suggested=escalation_suggested,
        assessed_at=datetime.utcnow(),
    )


def _build_recommendation(
    risk_level, rule_reasons, med_warnings, shap_text,
    med_override, flags
) -> str:
    """Generate human-readable clinical recommendation."""
    lines = []

    if risk_level == RiskLevel.critical:
        lines.append("⚠️ IMMEDIATE ESCALATION REQUIRED.")
        if rule_reasons:
            lines.append(f"Critical finding: {rule_reasons[0]}")
    elif risk_level == RiskLevel.high:
        lines.append("URGENT: Escalation to specialist strongly recommended.")
    elif risk_level == RiskLevel.moderate:
        lines.append("CAUTION: Close monitoring required. Consider specialist consultation.")
    else:
        lines.append("LOW RISK: Can be managed at PHC level with standard protocols.")

    if shap_text:
        lines.append(f"AI interpretation: {shap_text}")

    severe_meds = [w for w in med_warnings if w.severity in ("severe", "contraindicated")]
    if severe_meds:
        lines.append(f"⚠️ Medication alert: {severe_meds[0].message}")

    if flags.pregnant and risk_level in (RiskLevel.critical, RiskLevel.high):
        lines.append("Maternal emergency protocol — ensure IV access, monitor fetal heart rate.")

    return " ".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE: app/api/v1/endpoints/escalate.py
# ═══════════════════════════════════════════════════════════════════════════════

escalate_router = APIRouter(prefix="/escalate")


@escalate_router.post(
    "",
    summary="Escalate case to specialist",
    description="Generates magic link, triggers SBAR generation, notifies specialist, broadcasts WebSocket."
)
async def escalate_case(
    case_id: str,
    escalation_reason: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(require_phw),
):
    """
    Sample Request:
    POST /api/v1/escalate
    {
        "case_id": "uuid",
        "escalation_reason": "Patient showing signs of preeclampsia — BP rising, severe headache"
    }

    Sample Response:
    {
        "case_id": "uuid",
        "specialist_magic_link": "https://cdss.phc.in/specialist/portal?token=eyJ...",
        "sbar": {
            "situation": "32-year-old pregnant female at 34 weeks with BP 155/100, severe headache...",
            "background": "No prior hypertension. On iron + folic acid. Escalated by PHW Priya Sharma...",
            "assessment": "AI classified CRITICAL (score 0.91). Rule engine triggered: pregnancy hypertension...",
            "recommendation": "Urgent specialist assessment required. Magnesium sulfate may be indicated..."
        },
        "escalated_at": "2025-01-15T10:32:00Z"
    }
    """
    from app.core.security import create_magic_link_token
    from app.services.sbar_service import generate_sbar
    from app.websocket.manager import ws_manager

    # Generate secure magic link
    magic_token = create_magic_link_token(case_id)
    magic_link = f"{settings.FRONTEND_URL}/specialist/portal?token={magic_token}"

    # Update case status in DB (omitted for brevity — uses SQLAlchemy async)
    # ...

    # Broadcast to room
    await ws_manager.broadcast_to_room(case_id, {
        "type": "STATUS_UPDATE",
        "status": "escalated",
        "case_id": case_id,
    })

    return {
        "case_id": case_id,
        "specialist_magic_link": magic_link,
        "escalated_at": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FILE: app/api/v1/endpoints/specialist.py
# ═══════════════════════════════════════════════════════════════════════════════

from app.core.config import settings

specialist_router = APIRouter(prefix="/specialist")


@specialist_router.get(
    "/portal/{token}",
    summary="Load specialist portal data via magic link"
)
async def specialist_portal(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns full case data for specialist review.
    Authenticated via magic link token (no login required for specialist).

    Sample Response:
    {
        "case_id": "uuid",
        "patient_summary": { "age": 32, "sex": "female", "vulnerability_flags": {...} },
        "vitals": { "systolic_bp": 155, ... },
        "risk_assessment": { "final_risk_level": "critical", "shap_features": [...], ... },
        "sbar": { "situation": "...", "background": "...", "assessment": "...", "recommendation": "..." },
        "phw_name": "Priya Sharma",
        "facility": "PHC Wardha Rural"
    }
    """
    from app.core.security import decode_magic_token
    payload = decode_magic_token(token)
    case_id = payload["case_id"]

    # Fetch case + assessment from DB and return
    # ... (DB queries omitted for brevity)

    return {"case_id": case_id, "status": "portal_loaded"}


@specialist_router.post(
    "/advice",
    summary="Submit specialist advice — pushes to PHW via WebSocket"
)
async def submit_advice(
    case_id: str,
    advice_type: str,
    custom_notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Doctor submits advice. This:
    1. Saves to DB
    2. Updates case status to "advised"
    3. Pushes advice to PHW via WebSocket immediately

    Sample Request:
    {
        "case_id": "uuid",
        "advice_type": "urgent_referral",
        "custom_notes": "Transfer to district hospital. Initiate MgSO4 loading dose en route.",
        "investigations": ["CBC", "LFT", "urine protein"],
        "follow_up_hours": 2
    }
    """
    from app.websocket.manager import push_specialist_advice_to_phw

    advice_record = {
        "case_id": case_id,
        "advice_type": advice_type,
        "custom_notes": custom_notes,
        "submitted_at": datetime.utcnow().isoformat(),
    }

    # Save to DB, update case status
    # ...

    # Push to PHW via WebSocket
    await push_specialist_advice_to_phw(case_id, advice_record)

    return {"status": "advice_submitted", "case_id": case_id}


# ═══════════════════════════════════════════════════════════════════════════════
# FILE: app/api/v1/router.py
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(analyze_router, tags=["Decision Engine"])
api_router.include_router(escalate_router, tags=["Escalation"])
api_router.include_router(specialist_router, tags=["Specialist"])
