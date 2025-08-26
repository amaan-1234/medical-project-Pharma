"""
FastAPI backend for Clinical Trial Outcome Predictor
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uvicorn
import numpy as np
import joblib
import torch
import json
from datetime import datetime

from config.config import get_config, API_HOST, API_PORT, API_DEBUG
from utils.llm_utils import ProtocolAnalyzer
from utils.validation_utils import ModelValidator

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Trial Outcome Predictor API",
    description="AI-powered API for predicting clinical trial success using multi-modal analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PatientData(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., description="Patient gender (M/F)")
    race: Optional[str] = Field(None, description="Patient race")
    ethnicity: Optional[str] = Field(None, description="Patient ethnicity")
    bmi: Optional[float] = Field(None, ge=10, le=60, description="Body mass index")
    smoking_status: Optional[str] = Field(None, description="Smoking status")
    diabetes: Optional[int] = Field(None, ge=0, le=1, description="Diabetes status (0/1)")
    hypertension: Optional[int] = Field(None, ge=0, le=1, description="Hypertension status (0/1)")
    heart_disease: Optional[int] = Field(None, ge=0, le=1, description="Heart disease status (0/1)")
    creatinine: Optional[float] = Field(None, ge=0.1, le=10.0, description="Creatinine level")
    albumin: Optional[float] = Field(None, ge=1.0, le=8.0, description="Albumin level")
    hemoglobin: Optional[float] = Field(None, ge=5.0, le=25.0, description="Hemoglobin level")
    platelet_count: Optional[float] = Field(None, ge=50, le=1000, description="Platelet count")
    white_blood_cells: Optional[float] = Field(None, ge=1.0, le=20.0, description="White blood cell count")
    sodium: Optional[float] = Field(None, ge=120, le=160, description="Sodium level")
    potassium: Optional[float] = Field(None, ge=2.0, le=8.0, description="Potassium level")
    glucose: Optional[float] = Field(None, ge=50, le=400, description="Glucose level")

class TrialProtocol(BaseModel):
    protocol_text: str = Field(..., min_length=100, description="Clinical trial protocol text")
    trial_phase: Optional[str] = Field(None, description="Trial phase (Phase 1/2/3/4)")
    enrollment_target: Optional[int] = Field(None, ge=10, le=10000, description="Target enrollment")
    duration_months: Optional[int] = Field(None, ge=1, le=120, description="Trial duration in months")
    intervention_type: Optional[str] = Field(None, description="Intervention type")

class PredictionRequest(BaseModel):
    patient_data: PatientData
    trial_protocol: Optional[TrialProtocol] = None
    use_llm_analysis: bool = Field(True, description="Whether to use LLM for protocol analysis")

class PredictionResponse(BaseModel):
    success_probability: float = Field(..., ge=0, le=1, description="Predicted success probability")
    prediction: str = Field(..., description="Predicted outcome (Success/Failure)")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence score")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    model_used: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")

class ProtocolAnalysisResponse(BaseModel):
    analysis: Dict[str, Any] = Field(..., description="Protocol analysis results")
    risk_score: float = Field(..., ge=0, le=1, description="Calculated risk score")
    summary: str = Field(..., description="Human-readable summary")
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    timestamp: str = Field(..., description="Current timestamp")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    llm_available: bool = Field(..., description="Whether LLM is available")

# Global variables for loaded models
models = {}
scaler = None
protocol_analyzer = None
model_validator = ModelValidator()

def load_models():
    """Load trained models and scaler"""
    global models, scaler, protocol_analyzer
    
    try:
        models_dir = Path("models/best_models")
        
        # Load scaler
        scaler_path = models_dir / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(str(scaler_path))
            print("✓ Scaler loaded successfully")
        else:
            print("⚠ Scaler not found - predictions may not work correctly")
        
        # Load Random Forest model
        rf_path = models_dir / "random_forest_model.pkl"
        if rf_path.exists():
            models['random_forest'] = joblib.load(str(rf_path))
            print("✓ Random Forest model loaded successfully")
        else:
            print("⚠ Random Forest model not found")
        
        # Load Neural Network model
        nn_path = models_dir / "patient_model.pth"
        if nn_path.exists():
            from utils.model_utils import PatientDataModel
            nn_model = PatientDataModel(input_size=17, hidden_sizes=[128, 64, 32])
            nn_model.load_state_dict(torch.load(str(nn_path), map_location='cpu'))
            nn_model.eval()
            models['neural_network'] = nn_model
            print("✓ Neural Network model loaded successfully")
        else:
            print("⚠ Neural Network model not found")
        
        # Initialize protocol analyzer
        protocol_analyzer = ProtocolAnalyzer()
        print("✓ Protocol analyzer initialized")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

def get_patient_features(patient_data: PatientData) -> np.ndarray:
    """Convert patient data to feature array"""
    features = []
    
    # Demographics
    features.extend([
        patient_data.age,
        1.0 if patient_data.gender.upper() in ['M', 'MALE'] else 0.0,
        1.0 if patient_data.race == 'White' else 0.0,
        1.0 if patient_data.ethnicity == 'Hispanic' else 0.0,
        patient_data.bmi or 25.0,  # Default BMI
        1.0 if patient_data.smoking_status == 'Current' else 0.0,
        patient_data.diabetes or 0.0,
        patient_data.hypertension or 0.0,
        patient_data.heart_disease or 0.0
    ])
    
    # Biomarkers
    features.extend([
        patient_data.creatinine or 1.1,
        patient_data.albumin or 4.0,
        patient_data.hemoglobin or 14.0,
        patient_data.platelet_count or 250.0,
        patient_data.white_blood_cells or 7.0,
        patient_data.sodium or 140.0,
        patient_data.potassium or 4.0,
        patient_data.glucose or 100.0
    ])
    
    return np.array(features).reshape(1, -1)

def predict_trial_outcome(patient_features: np.ndarray, 
                         protocol_analysis: Optional[Dict] = None,
                         patient_data: Optional[Dict] = None) -> Dict[str, Any]:
    """Make prediction using loaded models"""
    
    if scaler is None:
        raise HTTPException(status_code=500, detail="Scaler not loaded")
    
    if not models:
        raise HTTPException(status_code=500, detail="No models loaded")
    
    # Scale features
    features_scaled = scaler.transform(patient_features)
    
    predictions = {}
    
    # Random Forest prediction
    if 'random_forest' in models:
        rf_model = models['random_forest']
        rf_proba = rf_model.predict_proba(features_scaled)[0]
        predictions['random_forest'] = {
            'probability': rf_proba[1],
            'prediction': 'Success' if rf_proba[1] > 0.5 else 'Failure'
        }
    
    # Neural Network prediction
    if 'neural_network' in models:
        nn_model = models['neural_network']
        with torch.no_grad():
            nn_input = torch.FloatTensor(features_scaled)
            nn_output = nn_model(nn_input)
            nn_proba = nn_output.item()
            predictions['neural_network'] = {
                'probability': nn_proba,
                'prediction': 'Success' if nn_proba > 0.5 else 'Failure'
            }
    
    # Ensemble prediction (average of available models)
    if predictions:
        avg_probability = np.mean([pred['probability'] for pred in predictions.values()])
        ensemble_prediction = 'Success' if avg_probability > 0.5 else 'Failure'
        
        # Identify risk factors
        risk_factors = []
        if patient_features[0, 0] > 70:  # Age > 70
            risk_factors.append("Advanced age")
        if patient_data and patient_data.get('bmi', 0) > 30:
            risk_factors.append("High BMI")
        if patient_data and patient_data.get('diabetes', 0):
            risk_factors.append("Diabetes")
        if patient_data and patient_data.get('hypertension', 0):
            risk_factors.append("Hypertension")
        if patient_data and patient_data.get('heart_disease', 0):
            risk_factors.append("Heart disease")
        
        # Add protocol risk factors if available
        if protocol_analysis:
            protocol_risk = protocol_analysis.get('risk_score', 0)
            if protocol_risk > 0.7:
                risk_factors.append("High protocol risk")
            elif protocol_risk > 0.5:
                risk_factors.append("Moderate protocol risk")
        
        return {
            'success_probability': avg_probability,
            'prediction': ensemble_prediction,
            'confidence_score': 0.8,  # Placeholder
            'risk_factors': risk_factors,
            'model_used': 'ensemble',
            'individual_predictions': predictions
        }
    else:
        raise HTTPException(status_code=500, detail="No models available for prediction")

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("Starting Clinical Trial Outcome Predictor API...")
    load_models()
    print("API startup complete!")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Clinical Trial Outcome Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(models) > 0,
        llm_available=protocol_analyzer is not None
    )

@app.post("/predict/trial", response_model=PredictionResponse)
async def predict_trial_outcome_endpoint(request: PredictionRequest):
    """Predict clinical trial outcome"""
    
    # Validate input data
    is_valid, validation_messages = model_validator.validate_prediction_input(request.patient_data.dict())
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {validation_messages}")
    
    try:
        # Convert patient data to features
        patient_features = get_patient_features(request.patient_data)
        
        # Analyze protocol if provided
        protocol_analysis = None
        if request.trial_protocol and request.use_llm_analysis:
            protocol_analysis = protocol_analyzer.analyze_protocol(request.trial_protocol.protocol_text)
        
        # Make prediction
        prediction_result = predict_trial_outcome(patient_features, protocol_analysis, request.patient_data.dict())
        
        return PredictionResponse(
            success_probability=prediction_result['success_probability'],
            prediction=prediction_result['prediction'],
            confidence_score=prediction_result['confidence_score'],
            risk_factors=prediction_result['risk_factors'],
            model_used=prediction_result['model_used'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/analyze/protocol", response_model=ProtocolAnalysisResponse)
async def analyze_protocol_endpoint(request: TrialProtocol):
    """Analyze clinical trial protocol"""
    
    if not protocol_analyzer:
        raise HTTPException(status_code=500, detail="Protocol analyzer not available")
    
    try:
        # Analyze protocol
        analysis = protocol_analyzer.analyze_protocol(request.protocol_text)
        
        # Calculate risk score
        risk_score = protocol_analyzer.calculate_risk_score(analysis)
        
        # Generate summary
        summary = protocol_analyzer.generate_protocol_summary(analysis)
        
        return ProtocolAnalysisResponse(
            analysis=analysis,
            risk_score=risk_score,
            summary=summary,
            confidence=analysis.get('confidence_score', 0.7)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Protocol analysis failed: {str(e)}")

@app.get("/models/status")
async def get_model_status():
    """Get status of loaded models"""
    return {
        "models_loaded": list(models.keys()),
        "scaler_loaded": scaler is not None,
        "protocol_analyzer_available": protocol_analyzer is not None,
        "total_models": len(models)
    }

@app.get("/trials/history")
async def get_trial_history():
    """Get sample historical trial data"""
    try:
        from utils.data_utils import load_sample_data
        data = load_sample_data()
        
        # Return summary statistics
        return {
            "total_trials": len(data['trial_data']),
            "total_patients": len(data['patient_data']),
            "phase_distribution": data['trial_data']['phase'].value_counts().to_dict(),
            "intervention_types": data['trial_data']['intervention_type'].value_counts().to_dict(),
            "success_rate": data['trial_data']['success_rate'].mean()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load trial history: {str(e)}")

if __name__ == "__main__":
    print(f"Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_DEBUG,
        log_level="info"
    )
