"""
SQLAlchemy 2.0 Async ORM Models
"""

from sqlalchemy import (
    Column, String, Integer, SmallInteger, Boolean, Text,
    Numeric, ForeignKey, Enum as SAEnum, DateTime, BigInteger, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(Text, nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(SAEnum("phw", "specialist", "admin", name="user_role"), nullable=False)
    facility_id = Column(String(100))
    facility_name = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    last_login_at = Column(DateTime(timezone=True))
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True))

    cases_as_phw = relationship("Case", foreign_keys="Case.phw_id", back_populates="phw")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name_encrypted = Column(Text, nullable=False)
    dob_encrypted = Column(Text)
    phone_encrypted = Column(Text)
    age = Column(SmallInteger, nullable=False)
    sex = Column(String(10), nullable=False)
    village = Column(String(255))
    district = Column(String(255))
    state = Column(String(100))
    vulnerability_flags = Column(JSONB, default=dict)
    fhir_resource = Column(JSONB)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    facility_id = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True))

    cases = relationship("Case", back_populates="patient")


class Case(Base):
    __tablename__ = "cases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    phw_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    specialist_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    status = Column(
        SAEnum("intake", "analyzed", "escalated", "specialist_reviewing",
               "advised", "closed", "cancelled", name="case_status"),
        default="intake", nullable=False
    )
    chief_complaint = Column(Text)
    escalation_reason = Column(Text)
    specialist_magic_token = Column(Text, unique=True)
    magic_token_expires_at = Column(DateTime(timezone=True))
    closed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True))

    patient = relationship("Patient", back_populates="cases")
    phw = relationship("User", foreign_keys=[phw_id], back_populates="cases_as_phw")
    vitals = relationship("Vitals", back_populates="case", order_by="desc(Vitals.recorded_at)")
    medications = relationship("CaseMedication", back_populates="case")
    symptoms = relationship("CaseSymptom", back_populates="case")
    risk_assessments = relationship("RiskAssessment", back_populates="case")
    specialist_advice = relationship("SpecialistAdvice", back_populates="case")


class Vitals(Base):
    __tablename__ = "vitals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id"), nullable=False)
    recorded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    systolic_bp = Column(SmallInteger)
    diastolic_bp = Column(SmallInteger)
    heart_rate = Column(SmallInteger)
    respiratory_rate = Column(SmallInteger)
    spo2 = Column(Numeric(4, 1))
    temperature = Column(Numeric(4, 1))
    blood_glucose_mgdl = Column(SmallInteger)
    weight_kg = Column(Numeric(5, 1))
    gcs_score = Column(SmallInteger)
    raw_readings = Column(JSONB, default=dict)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    case = relationship("Case", back_populates="vitals")


class CaseMedication(Base):
    __tablename__ = "case_medications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id"), nullable=False)
    rxnorm_code = Column(String(20))
    drug_name = Column(String(255), nullable=False)
    dose = Column(String(100))
    frequency = Column(String(100))
    route = Column(String(50))
    is_current = Column(Boolean, default=True)
    added_at = Column(DateTime(timezone=True), server_default=func.now())

    case = relationship("Case", back_populates="medications")


class CaseSymptom(Base):
    __tablename__ = "case_symptoms"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id"), nullable=False)
    symptom_name = Column(String(255), nullable=False)
    is_red_flag = Column(Boolean, default=False)
    severity = Column(String(20))
    duration_hours = Column(Integer)
    notes = Column(Text)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    case = relationship("Case", back_populates="symptoms")


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id"), nullable=False)
    vitals_id = Column(UUID(as_uuid=True), ForeignKey("vitals.id"))
    rule_triggered = Column(Boolean, default=False)
    rule_level = Column(String(20))
    rule_reasons = Column(JSONB, default=list)
    ml_risk_probability = Column(Numeric(4, 3))
    ml_risk_level = Column(String(20))
    shap_values = Column(JSONB, default=dict)
    shap_top_features = Column(JSONB, default=list)
    shap_text_interpretation = Column(Text)
    med_warnings = Column(JSONB, default=list)
    med_override_triggered = Column(Boolean, default=False)
    final_risk_level = Column(String(20), nullable=False)
    final_risk_score = Column(Numeric(4, 3))
    recommendation = Column(Text)
    escalation_suggested = Column(Boolean, default=False)
    sbar_situation = Column(Text)
    sbar_background = Column(Text)
    sbar_assessment = Column(Text)
    sbar_recommendation = Column(Text)
    sbar_generated_at = Column(DateTime(timezone=True))
    model_version = Column(String(50))
    assessed_at = Column(DateTime(timezone=True), server_default=func.now())

    case = relationship("Case", back_populates="risk_assessments")


class SpecialistAdvice(Base):
    __tablename__ = "specialist_advice"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id"), nullable=False)
    risk_assessment_id = Column(UUID(as_uuid=True), ForeignKey("risk_assessments.id"), nullable=False)
    specialist_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    acknowledged_at = Column(DateTime(timezone=True))
    advice_type = Column(String(30), nullable=False)
    custom_notes = Column(Text)
    medications_advised = Column(JSONB, default=list)
    investigations = Column(JSONB, default=list)
    follow_up_hours = Column(Integer)
    submitted_at = Column(DateTime(timezone=True), server_default=func.now())

    case = relationship("Case", back_populates="specialist_advice")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(UUID(as_uuid=True))
    ip_address = Column(INET)
    user_agent = Column(Text)
    request_id = Column(UUID(as_uuid=True))
    old_value = Column(JSONB)
    new_value = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
