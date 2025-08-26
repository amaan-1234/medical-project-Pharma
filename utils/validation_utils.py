"""
Validation utility functions for clinical trial data and models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
from datetime import datetime, date
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates clinical trial data for quality and consistency
    """
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_patient_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate patient demographic and biomarker data
        
        Args:
            df: Patient data dataframe
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check required columns
        required_cols = ['patient_id', 'age', 'gender']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.validation_errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if 'age' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['age']):
                self.validation_errors.append("Age column must be numeric")
            else:
                # Check age range
                invalid_age = df[(df['age'] < 0) | (df['age'] > 120)]
                if len(invalid_age) > 0:
                    self.validation_errors.append(f"Invalid age values found: {len(invalid_age)} records")
        
        # Check for missing values in critical fields
        critical_cols = ['patient_id', 'age']
        for col in critical_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    self.validation_warnings.append(f"Missing values in {col}: {missing_count} records")
        
        # Check for duplicate patient IDs
        if 'patient_id' in df.columns:
            duplicates = df['patient_id'].duplicated().sum()
            if duplicates > 0:
                self.validation_errors.append(f"Duplicate patient IDs found: {duplicates} records")
        
        # Check gender values
        if 'gender' in df.columns:
            valid_genders = ['M', 'F', 'Male', 'Female', 'm', 'f']
            invalid_gender = df[~df['gender'].isin(valid_genders)]
            if len(invalid_gender) > 0:
                self.validation_warnings.append(f"Non-standard gender values found: {len(invalid_gender)} records")
        
        return len(self.validation_errors) == 0, self.validation_errors + self.validation_warnings
    
    def validate_biomarker_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate biomarker laboratory data
        
        Args:
            df: Biomarker data dataframe
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check required columns
        required_cols = ['patient_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.validation_errors.append(f"Missing required columns: {missing_cols}")
        
        # Check biomarker value ranges
        biomarker_ranges = {
            'creatinine': (0.5, 3.0),
            'albumin': (2.5, 5.5),
            'hemoglobin': (10.0, 18.0),
            'platelet_count': (150, 450),
            'white_blood_cells': (4.0, 11.0),
            'sodium': (135, 145),
            'potassium': (3.5, 5.0),
            'glucose': (70, 200)
        }
        
        for biomarker, (min_val, max_val) in biomarker_ranges.items():
            if biomarker in df.columns:
                if not pd.api.types.is_numeric_dtype(df[biomarker]):
                    self.validation_warnings.append(f"{biomarker} column is not numeric")
                else:
                    # Check for extreme outliers
                    outliers = df[(df[biomarker] < min_val * 0.5) | (df[biomarker] > max_val * 2)]
                    if len(outliers) > 0:
                        self.validation_warnings.append(f"Potential outliers in {biomarker}: {len(outliers)} records")
        
        return len(self.validation_errors) == 0, self.validation_errors + self.validation_warnings
    
    def validate_trial_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate clinical trial protocol data
        
        Args:
            df: Trial data dataframe
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check required columns
        required_cols = ['trial_id', 'phase', 'enrollment']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.validation_errors.append(f"Missing required columns: {missing_cols}")
        
        # Check trial phase values
        if 'phase' in df.columns:
            valid_phases = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4']
            invalid_phases = df[~df['phase'].isin(valid_phases)]
            if len(invalid_phases) > 0:
                self.validation_warnings.append(f"Non-standard phase values found: {len(invalid_phases)} records")
        
        # Check enrollment numbers
        if 'enrollment' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['enrollment']):
                self.validation_errors.append("Enrollment column must be numeric")
            else:
                # Check for reasonable enrollment values
                invalid_enrollment = df[(df['enrollment'] < 10) | (df['enrollment'] > 10000)]
                if len(invalid_enrollment) > 0:
                    self.validation_warnings.append(f"Unusual enrollment values found: {len(invalid_enrollment)} records")
        
        # Check for duplicate trial IDs
        if 'trial_id' in df.columns:
            duplicates = df['trial_id'].duplicated().sum()
            if duplicates > 0:
                self.validation_errors.append(f"Duplicate trial IDs found: {duplicates} records")
        
        return len(self.validation_errors) == 0, self.validation_errors + self.validation_warnings
    
    def validate_outcome_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate trial outcome data
        
        Args:
            df: Outcome data dataframe
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check required columns
        required_cols = ['trial_id', 'patient_id', 'outcome']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.validation_errors.append(f"Missing required columns: {missing_cols}")
        
        # Check outcome values
        if 'outcome' in df.columns:
            valid_outcomes = ['Success', 'Failure', 'success', 'failure', 'SUCCESS', 'FAILURE']
            invalid_outcomes = df[~df['outcome'].isin(valid_outcomes)]
            if len(invalid_outcomes) > 0:
                self.validation_warnings.append(f"Non-standard outcome values found: {len(invalid_outcomes)} records")
        
        # Check follow-up duration if available
        if 'follow_up_days' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['follow_up_days']):
                self.validation_warnings.append("Follow-up days column is not numeric")
            else:
                # Check for reasonable follow-up periods
                invalid_followup = df[(df['follow_up_days'] < 0) | (df['follow_up_days'] > 1000)]
                if len(invalid_followup) > 0:
                    self.validation_warnings.append(f"Unusual follow-up periods found: {len(invalid_followup)} records")
        
        return len(self.validation_errors) == 0, self.validation_errors + self.validation_warnings

class ModelValidator:
    """
    Validates machine learning models and predictions
    """
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_prediction_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate input data for model predictions
        
        Args:
            input_data: Dictionary containing input features
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check required fields
        required_fields = ['age', 'gender']
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            self.validation_errors.append(f"Missing required fields: {missing_fields}")
        
        # Validate age
        if 'age' in input_data:
            age = input_data['age']
            if not isinstance(age, (int, float)) or age < 0 or age > 120:
                self.validation_errors.append("Age must be a number between 0 and 120")
        
        # Validate gender
        if 'gender' in input_data:
            gender = input_data['gender']
            valid_genders = ['M', 'F', 'Male', 'Female', 'm', 'f']
            if gender not in valid_genders:
                self.validation_errors.append(f"Gender must be one of: {valid_genders}")
        
        # Validate biomarker values if present
        biomarker_ranges = {
            'creatinine': (0.5, 3.0),
            'albumin': (2.5, 5.5),
            'hemoglobin': (10.0, 18.0),
            'platelet_count': (150, 450),
            'white_blood_cells': (4.0, 11.0),
            'sodium': (135, 145),
            'potassium': (3.5, 5.0),
            'glucose': (70, 200)
        }
        
        for biomarker, (min_val, max_val) in biomarker_ranges.items():
            if biomarker in input_data:
                value = input_data[biomarker]
                if not isinstance(value, (int, float)):
                    self.validation_warnings.append(f"{biomarker} must be a number")
                elif value < min_val or value > max_val:
                    self.validation_warnings.append(f"{biomarker} value {value} is outside normal range ({min_val}-{max_val})")
        
        return len(self.validation_errors) == 0, self.validation_errors + self.validation_warnings
    
    def validate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_pred_proba: Optional[np.ndarray] = None) -> Tuple[bool, List[str]]:
        """
        Validate model performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check input shapes
        if len(y_true) != len(y_pred):
            self.validation_errors.append("True and predicted labels must have the same length")
            return False, self.validation_errors
        
        if y_pred_proba is not None and len(y_true) != len(y_pred_proba):
            self.validation_errors.append("True labels and predicted probabilities must have the same length")
            return False, self.validation_errors
        
        # Check for all same predictions
        unique_preds = np.unique(y_pred)
        if len(unique_preds) == 1:
            self.validation_warnings.append("Model predicts only one class - possible model issue")
        
        # Check for reasonable accuracy
        accuracy = np.mean(y_true == y_pred)
        if accuracy < 0.5:
            self.validation_warnings.append(f"Model accuracy ({accuracy:.3f}) is below 50% - possible model issue")
        
        # Check for class imbalance
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        if len(unique_true) == 2:
            min_class_ratio = min(counts_true) / max(counts_true)
            if min_class_ratio < 0.1:
                self.validation_warnings.append("Severe class imbalance detected - consider rebalancing techniques")
        
        return len(self.validation_errors) == 0, self.validation_errors + self.validation_warnings

def validate_trial_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[bool, List[str]]]:
    """
    Comprehensive validation of all clinical trial data
    
    Args:
        data_dict: Dictionary containing different data types
        
    Returns:
        Dictionary of validation results for each data type
    """
    validator = DataValidator()
    results = {}
    
    # Validate each data type
    if 'patient_data' in data_dict:
        results['patient_data'] = validator.validate_patient_data(data_dict['patient_data'])
    
    if 'biomarker_data' in data_dict:
        results['biomarker_data'] = validator.validate_biomarker_data(data_dict['biomarker_data'])
    
    if 'trial_data' in data_dict:
        results['trial_data'] = validator.validate_trial_data(data_dict['trial_data'])
    
    if 'outcome_data' in data_dict:
        results['outcome_data'] = validator.validate_outcome_data(data_dict['outcome_data'])
    
    return results

def print_validation_report(validation_results: Dict[str, Tuple[bool, List[str]]]):
    """
    Print a formatted validation report
    
    Args:
        validation_results: Dictionary of validation results
    """
    print("\n" + "="*60)
    print("CLINICAL TRIAL DATA VALIDATION REPORT")
    print("="*60)
    
    all_valid = True
    
    for data_type, (is_valid, messages) in validation_results.items():
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"\n{data_type.upper()}: {status}")
        
        if messages:
            for msg in messages:
                if "error" in msg.lower():
                    print(f"  ERROR: {msg}")
                    all_valid = False
                else:
                    print(f"  WARNING: {msg}")
        else:
            print("  No issues found")
    
    print("\n" + "="*60)
    if all_valid:
        print("OVERALL STATUS: ✓ ALL DATA VALIDATED SUCCESSFULLY")
    else:
        print("OVERALL STATUS: ✗ VALIDATION FAILED - PLEASE REVIEW ERRORS")
    print("="*60)

if __name__ == "__main__":
    # Test validation functions
    from utils.data_utils import load_sample_data
    
    print("Testing data validation...")
    data = load_sample_data()
    
    # Validate all data
    validation_results = validate_trial_data(data)
    
    # Print validation report
    print_validation_report(validation_results)
