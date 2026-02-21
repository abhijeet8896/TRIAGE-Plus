from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "REPLACE-WITH-SECURE-RANDOM-64-CHAR-STRING"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://cdss:cdss_pass@localhost:5432/cdss_db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # JWT
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    MAGIC_LINK_EXPIRE_MINUTES: int = 120

    # ML
    RISK_MODEL_PATH: str = "app/ml/models/xgb_risk_model.pkl"
    SHAP_EXPLAINER_PATH: str = "app/ml/models/shap_explainer.pkl"

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Clinical thresholds (NEWS2-aligned)
    SPO2_CRITICAL: float = 90.0
    SBP_CRITICAL: int = 90
    RR_CRITICAL: int = 30
    TEMP_CRITICAL: float = 40.0

    # PHI Encryption
    PHI_ENCRYPTION_KEY: str = "QSr-KD5QVWXVDdZgnFnSKZvCbQydZm9IjZf_9qzDmVk="

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
