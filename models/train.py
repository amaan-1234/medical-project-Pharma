"""
Main training script for Clinical Trial Outcome Predictor models
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from utils.data_utils import load_sample_data, create_training_dataset, save_sample_data
from utils.model_utils import (
    train_patient_model, train_random_forest, evaluate_model,
    save_model, plot_training_history, PatientDataModel
)
from utils.validation_utils import validate_trial_data, print_validation_report
from config.config import MODEL_CONFIG, TRAINING_CONFIG

def main():
    """
    Main training function
    """
    print("="*60)
    print("CLINICAL TRIAL OUTCOME PREDICTOR - MODEL TRAINING")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create necessary directories
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    checkpoints_dir = models_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    best_models_dir = models_dir / "best_models"
    best_models_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate and validate sample data
    print("\n1. Generating and validating sample data...")
    data_dict = load_sample_data()
    
    # Validate data
    validation_results = validate_trial_data(data_dict)
    print_validation_report(validation_results)
    
    # Save sample data
    save_sample_data(data_dict, "data")
    print("Sample data saved to data/ directory")
    
    # Step 2: Prepare training data
    print("\n2. Preparing training data...")
    X_train, X_test, y_train, y_test = create_training_dataset()
    
    # Split training data into train and validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set size: {X_train_final.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Feature dimension: {X_train_final.shape[1]}")
    
    # Step 3: Train Neural Network Model
    print("\n3. Training Neural Network Model...")
    
    # Update config with actual input size
    patient_config = MODEL_CONFIG['patient_model'].copy()
    patient_config['input_size'] = X_train_final.shape[1]
    patient_config['epochs'] = TRAINING_CONFIG['epochs']
    patient_config['batch_size'] = TRAINING_CONFIG['batch_size']
    patient_config['early_stopping_patience'] = TRAINING_CONFIG['early_stopping_patience']
    
    print("Training patient data neural network...")
    patient_model, patient_history = train_patient_model(
        X_train_final, y_train_final, X_val, y_val, patient_config
    )
    
    # Plot training history
    plot_path = checkpoints_dir / "patient_model_training.png"
    plot_training_history(patient_history, str(plot_path))
    
    # Step 4: Train Random Forest Model
    print("\n4. Training Random Forest Model...")
    print("Training random forest baseline...")
    rf_model = train_random_forest(X_train_final, y_train_final)
    
    # Step 5: Evaluate Models
    print("\n5. Evaluating Models...")
    
    # Evaluate neural network
    print("\nEvaluating Neural Network Model...")
    patient_metrics = evaluate_model(patient_model, X_test, y_test, "Neural Network")
    
    # Evaluate random forest
    print("\nEvaluating Random Forest Model...")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Step 6: Save Models
    print("\n6. Saving Models...")
    
    # Save neural network
    patient_model_path = best_models_dir / "patient_model.pth"
    save_model(patient_model, str(patient_model_path))
    
    # Save random forest
    rf_model_path = best_models_dir / "random_forest_model.pkl"
    save_model(rf_model, str(rf_model_path))
    
    # Save scaler for preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train_final)
    scaler_path = best_models_dir / "scaler.pkl"
    joblib.dump(scaler, str(scaler_path))
    
    # Step 7: Generate Training Report
    print("\n7. Generating Training Report...")
    
    # Create comprehensive report
    report = {
        'training_summary': {
            'total_samples': len(X_train) + len(X_test),
            'training_samples': len(X_train_final),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'features': X_train_final.shape[1],
            'classes': len(np.unique(y_train))
        },
        'model_performance': {
            'neural_network': patient_metrics,
            'random_forest': rf_metrics
        },
        'training_config': patient_config,
        'data_validation': validation_results
    }
    
    # Save report
    report_path = models_dir / "training_report.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Training report saved to {report_path}")
    
    # Step 8: Final Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print(f"\nModels saved to: {best_models_dir}")
    print(f"Checkpoints saved to: {checkpoints_dir}")
    print(f"Training report: {report_path}")
    
    print("\nModel Performance Summary:")
    print(f"Neural Network - Accuracy: {patient_metrics['accuracy']:.4f}, AUC: {patient_metrics.get('auc', 'N/A')}")
    print(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, AUC: {rf_metrics.get('auc', 'N/A')}")
    
    print("\nNext steps:")
    print("1. Use the trained models for predictions")
    print("2. Start the API server: python api/main.py")
    print("3. Launch the web interface: streamlit run web/app.py")
    
    return report

def quick_test():
    """
    Quick test function to verify models work
    """
    print("\nRunning quick test...")
    
    # Load saved models
    models_dir = Path("models/best_models")
    
    if not models_dir.exists():
        print("Models not found. Please run training first.")
        return
    
    # Load scaler
    scaler_path = models_dir / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(str(scaler_path))
        print("✓ Scaler loaded successfully")
    else:
        print("✗ Scaler not found")
        return
    
    # Load random forest model
    rf_path = models_dir / "random_forest_model.pkl"
    if rf_path.exists():
        rf_model = joblib.load(str(rf_path))
        print("✓ Random Forest model loaded successfully")
    else:
        print("✗ Random Forest model not found")
        return
    
    # Test prediction
    test_data = np.random.randn(1, 17)  # 17 features
    test_data_scaled = scaler.transform(test_data)
    
    prediction = rf_model.predict(test_data_scaled)
    probability = rf_model.predict_proba(test_data_scaled)
    
    print(f"✓ Test prediction successful:")
    print(f"  Prediction: {'Success' if prediction[0] == 1 else 'Failure'}")
    print(f"  Probability: {probability[0][1]:.3f}")
    
    print("\n✓ All tests passed! Models are ready for use.")

if __name__ == "__main__":
    try:
        # Run main training
        report = main()
        
        # Run quick test
        quick_test()
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
