"""
Data utility functions for clinical trial data processing
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import requests
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Load sample clinical trial data for demonstration
    Returns a dictionary with different data types
    """
    # Create sample patient demographic data
    np.random.seed(42)
    n_samples = 1000
    
    patient_data = pd.DataFrame({
        'patient_id': range(1, n_samples + 1),
        'age': np.random.normal(65, 15, n_samples).clip(18, 90),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Other'], n_samples),
        'ethnicity': np.random.choice(['Hispanic', 'Non-Hispanic'], n_samples),
        'bmi': np.random.normal(28, 6, n_samples).clip(16, 50),
        'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n_samples),
        'diabetes': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'hypertension': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'heart_disease': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    })
    
    # Create sample biomarker data
    biomarker_data = pd.DataFrame({
        'patient_id': range(1, n_samples + 1),
        'creatinine': np.random.normal(1.1, 0.3, n_samples).clip(0.5, 3.0),
        'albumin': np.random.normal(4.0, 0.5, n_samples).clip(2.5, 5.5),
        'hemoglobin': np.random.normal(14.0, 1.5, n_samples).clip(10.0, 18.0),
        'platelet_count': np.random.normal(250, 50, n_samples).clip(150, 450),
        'white_blood_cells': np.random.normal(7.0, 2.0, n_samples).clip(4.0, 11.0),
        'sodium': np.random.normal(140, 3, n_samples).clip(135, 145),
        'potassium': np.random.normal(4.0, 0.5, n_samples).clip(3.5, 5.0),
        'glucose': np.random.normal(100, 20, n_samples).clip(70, 200)
    })
    
    # Create sample trial protocol data
    trial_data = pd.DataFrame({
        'trial_id': range(1, 101),
        'phase': np.random.choice(['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'], 100),
        'enrollment': np.random.choice([50, 100, 200, 500, 1000, 2000], 100),
        'duration_months': np.random.choice([6, 12, 18, 24, 36, 48], 100),
        'intervention_type': np.random.choice(['Drug', 'Device', 'Procedure', 'Behavioral'], 100),
        'primary_outcome': np.random.choice(['Survival', 'Quality of Life', 'Disease Progression', 'Safety'], 100),
        'success_rate': np.random.beta(2, 3, 100)  # Beta distribution for realistic success rates
    })
    
    # Create sample trial outcomes
    outcomes = []
    for i in range(100):
        n_patients = trial_data.iloc[i]['enrollment']
        success_rate = trial_data.iloc[i]['success_rate']
        n_success = int(n_patients * success_rate)
        n_failure = n_patients - n_success
        
        trial_outcomes = pd.DataFrame({
            'trial_id': [i + 1] * n_patients,
            'patient_id': range(1, n_patients + 1),
            'outcome': ['Success'] * n_success + ['Failure'] * n_failure,
            'follow_up_days': np.random.exponential(180, n_patients).clip(30, 365)
        })
        outcomes.append(trial_outcomes)
    
    outcome_data = pd.concat(outcomes, ignore_index=True)
    
    return {
        'patient_data': patient_data,
        'biomarker_data': biomarker_data,
        'trial_data': trial_data,
        'outcome_data': outcome_data
    }

def preprocess_patient_data(patient_df: pd.DataFrame, biomarker_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess patient demographic and biomarker data for model training
    
    Args:
        patient_df: Patient demographic data
        biomarker_df: Patient biomarker data
        
    Returns:
        X: Feature matrix
        y: Target vector (trial success)
    """
    # Merge patient and biomarker data
    merged_data = patient_df.merge(biomarker_df, on='patient_id', how='inner')
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['gender', 'race', 'ethnicity', 'smoking_status']
    
    for col in categorical_cols:
        merged_data[col] = le.fit_transform(merged_data[col])
    
    # Select features
    feature_cols = [
        'age', 'gender', 'race', 'ethnicity', 'bmi', 'smoking_status',
        'diabetes', 'hypertension', 'heart_disease',
        'creatinine', 'albumin', 'hemoglobin', 'platelet_count',
        'white_blood_cells', 'sodium', 'potassium', 'glucose'
    ]
    
    X = merged_data[feature_cols].values
    
    # For demonstration, create synthetic targets based on risk factors
    # In real scenario, this would come from actual trial outcomes
    risk_score = (
        merged_data['age'] / 100 +
        merged_data['bmi'] / 50 +
        merged_data['diabetes'] * 0.3 +
        merged_data['hypertension'] * 0.2 +
        merged_data['heart_disease'] * 0.5 +
        (merged_data['creatinine'] - 1.0) * 0.5 +
        (merged_data['glucose'] - 100) / 100
    )
    
    # Convert to binary outcome (success/failure)
    y = (risk_score < risk_score.median()).astype(int)
    
    return X, y

def extract_protocol_features(protocol_text: str) -> Dict[str, float]:
    """
    Extract features from clinical trial protocol text using rule-based approach
    In production, this would use LLM analysis
    
    Args:
        protocol_text: Raw protocol text
        
    Returns:
        Dictionary of extracted features
    """
    text_lower = protocol_text.lower()
    
    features = {
        'has_placebo': 1.0 if 'placebo' in text_lower else 0.0,
        'has_randomization': 1.0 if 'random' in text_lower else 0.0,
        'has_blinding': 1.0 if 'blind' in text_lower else 0.0,
        'has_multicenter': 1.0 if 'multicenter' in text_lower else 0.0,
        'has_adaptive_design': 1.0 if 'adaptive' in text_lower else 0.0,
        'has_interim_analysis': 1.0 if 'interim' in text_lower else 0.0,
        'protocol_length': len(protocol_text) / 1000,  # Normalized length
        'has_safety_endpoints': 1.0 if 'safety' in text_lower else 0.0,
        'has_efficacy_endpoints': 1.0 if 'efficacy' in text_lower else 0.0,
        'has_biomarker_analysis': 1.0 if 'biomarker' in text_lower else 0.0
    }
    
    return features

def create_training_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a complete training dataset for model development
    
    Returns:
        X_train, X_test, y_train, y_test: Training and test splits
    """
    # Load sample data
    data_dict = load_sample_data()
    
    # Preprocess patient data
    X, y = preprocess_patient_data(data_dict['patient_data'], data_dict['biomarker_data'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def save_sample_data(data_dict: Dict[str, pd.DataFrame], output_dir: str = "data"):
    """
    Save sample data to CSV files
    
    Args:
        data_dict: Dictionary containing dataframes
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for name, df in data_dict.items():
        file_path = output_path / f"{name}.csv"
        df.to_csv(file_path, index=False)
        print(f"Saved {name} to {file_path}")

if __name__ == "__main__":
    # Generate and save sample data
    data = load_sample_data()
    save_sample_data(data)
    print("Sample data generated and saved successfully!")
