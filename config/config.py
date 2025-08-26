"""
Configuration file for the Clinical Trial Outcome Predictor
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
API_DIR = BASE_DIR / "api"
WEB_DIR = BASE_DIR / "web"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"

# Web Interface Configuration
WEB_HOST = os.getenv("WEB_HOST", "localhost")
WEB_PORT = int(os.getenv("WEB_PORT", "8501"))

# Model Configuration
MODEL_CONFIG = {
    "patient_model": {
        "input_size": 50,
        "hidden_sizes": [128, 64, 32],
        "output_size": 1,
        "dropout": 0.3,
        "learning_rate": 0.001
    },
    "genomic_model": {
        "sequence_length": 1000,
        "embedding_dim": 64,
        "num_filters": 128,
        "filter_sizes": [3, 4, 5],
        "dropout": 0.3,
        "learning_rate": 0.001
    },
    "ensemble_model": {
        "num_models": 3,
        "voting_method": "soft"
    }
}

# LLM Configuration
LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "openai"),
    "model": os.getenv("LLM_MODEL", "gpt-4"),
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "max_tokens": 1000,
    "temperature": 0.1
}

# Data Configuration
DATA_CONFIG = {
    "clinical_trials_url": "https://clinicaltrials.gov/api/query/study_fields",
    "max_trials": 10000,
    "batch_size": 32,
    "validation_split": 0.2,
    "test_split": 0.1
}

# Feature Configuration
FEATURE_CONFIG = {
    "demographic_features": [
        "age", "gender", "race", "ethnicity", "bmi", "smoking_status"
    ],
    "biomarker_features": [
        "creatinine", "albumin", "hemoglobin", "platelet_count",
        "white_blood_cells", "sodium", "potassium", "glucose"
    ],
    "protocol_features": [
        "phase", "enrollment", "duration", "primary_outcome",
        "inclusion_criteria", "exclusion_criteria", "intervention_type"
    ]
}

# Training Configuration
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "early_stopping_patience": 10,
    "model_checkpoint_path": str(MODELS_DIR / "checkpoints"),
    "best_model_path": str(MODELS_DIR / "best_models")
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(BASE_DIR / "logs" / "app.log")
}

# Security Configuration
SECURITY_CONFIG = {
    "secret_key": os.getenv("SECRET_KEY", "your-secret-key-here"),
    "algorithm": "HS256",
    "access_token_expire_minutes": 30
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "api": {
            "host": API_HOST,
            "port": API_PORT,
            "debug": API_DEBUG
        },
        "web": {
            "host": WEB_HOST,
            "port": WEB_PORT
        },
        "models": MODEL_CONFIG,
        "llm": LLM_CONFIG,
        "data": DATA_CONFIG,
        "features": FEATURE_CONFIG,
        "training": TRAINING_CONFIG,
        "logging": LOGGING_CONFIG,
        "security": SECURITY_CONFIG
    }
