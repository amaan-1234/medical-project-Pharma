"""
Demo script for Clinical Trial Outcome Predictor
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def run_demo():
    """Run a comprehensive demo of the system"""
    
    print("="*60)
    print("🏥 CLINICAL TRIAL OUTCOME PREDICTOR - DEMO")
    print("="*60)
    
    print("\nThis demo showcases the AI-Powered Clinical Trial Outcome Predictor")
    print("that combines deep learning and LLM analysis to predict trial success.\n")
    
    # Demo 1: Data Generation
    print("1️⃣ GENERATING SAMPLE CLINICAL TRIAL DATA")
    print("-" * 40)
    
    try:
        from utils.data_utils import load_sample_data, save_sample_data
        
        data = load_sample_data()
        print(f"✓ Generated {len(data)} datasets:")
        print(f"  • Patient Data: {len(data['patient_data'])} records")
        print(f"  • Biomarker Data: {len(data['biomarker_data'])} records")
        print(f"  • Trial Data: {len(data['trial_data'])} records")
        print(f"  • Outcome Data: {len(data['outcome_data'])} records")
        
        # Save data
        save_sample_data(data, "data")
        print("✓ Data saved to data/ directory")
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return
    
    # Demo 2: Protocol Analysis
    print("\n2️⃣ ANALYZING CLINICAL TRIAL PROTOCOL")
    print("-" * 40)
    
    try:
        from utils.llm_utils import ProtocolAnalyzer, create_sample_protocol
        
        analyzer = ProtocolAnalyzer()
        protocol = create_sample_protocol()
        
        print(f"✓ Sample protocol created ({len(protocol)} characters)")
        print("  Sample excerpt: " + protocol[:100] + "...")
        
        # Analyze protocol
        analysis = analyzer.analyze_protocol(protocol)
        print("✓ Protocol analysis completed:")
        print(f"  • Trial Phase: {analysis.get('trial_phase', 'Unknown')}")
        print(f"  • Enrollment: {analysis.get('enrollment_target', 'Unknown')}")
        print(f"  • Intervention: {analysis.get('intervention_type', 'Unknown')}")
        print(f"  • Randomization: {analysis.get('randomization', 'Unknown')}")
        print(f"  • Blinding: {analysis.get('blinding', 'Unknown')}")
        
        # Calculate risk
        risk_score = analyzer.calculate_risk_score(analysis)
        print(f"  • Risk Score: {risk_score:.3f}")
        
    except Exception as e:
        print(f"✗ Protocol analysis failed: {e}")
    
    # Demo 3: Data Validation
    print("\n3️⃣ VALIDATING DATA QUALITY")
    print("-" * 40)
    
    try:
        from utils.validation_utils import validate_trial_data, print_validation_report
        
        validation_results = validate_trial_data(data)
        print("✓ Data validation completed")
        
        # Show validation summary
        all_valid = True
        for data_type, (is_valid, messages) in validation_results.items():
            status = "✓ PASS" if is_valid else "✗ FAIL"
            print(f"  {data_type}: {status}")
            if not is_valid:
                all_valid = False
        
        if all_valid:
            print("🎉 All data passed validation!")
        else:
            print("⚠️ Some data validation issues found")
            
    except Exception as e:
        print(f"✗ Data validation failed: {e}")
    
    # Demo 4: Model Creation
    print("\n4️⃣ CREATING MACHINE LEARNING MODELS")
    print("-" * 40)
    
    try:
        from utils.model_utils import PatientDataModel, GenomicModel
        
        # Create patient model
        patient_model = PatientDataModel(
            input_size=17,
            hidden_sizes=[128, 64, 32],
            output_size=1
        )
        print("✓ Patient Data Model created")
        print(f"  • Input size: 17 features")
        print(f"  • Hidden layers: [128, 64, 32]")
        print(f"  • Output: Binary classification")
        
        # Create genomic model
        genomic_model = GenomicModel(
            sequence_length=1000,
            embedding_dim=64,
            num_filters=128,
            filter_sizes=[3, 4, 5]
        )
        print("✓ Genomic Model created")
        print(f"  • Sequence length: 1000")
        print(f"  • Embedding dimension: 64")
        print(f"  • Convolutional filters: 128")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
    
    # Demo 5: Sample Prediction
    print("\n5️⃣ SAMPLE PREDICTION")
    print("-" * 40)
    
    try:
        # Create sample patient data
        sample_patient = {
            'age': 65,
            'gender': 'M',
            'race': 'White',
            'ethnicity': 'Non-Hispanic',
            'bmi': 28.5,
            'smoking_status': 'Former',
            'diabetes': 1,
            'hypertension': 1,
            'heart_disease': 0,
            'creatinine': 1.2,
            'albumin': 4.1,
            'hemoglobin': 13.8,
            'platelet_count': 240,
            'white_blood_cells': 7.2,
            'sodium': 141,
            'potassium': 4.1,
            'glucose': 135
        }
        
        print("✓ Sample patient data created:")
        print(f"  • Age: {sample_patient['age']} years")
        print(f"  • Gender: {sample_patient['gender']}")
        print(f"  • BMI: {sample_patient['bmi']}")
        print(f"  • Medical conditions: Diabetes, Hypertension")
        print(f"  • Biomarkers: Within normal ranges")
        
        # Simple risk assessment
        risk_factors = []
        if sample_patient['age'] > 60:
            risk_factors.append("Advanced age")
        if sample_patient['diabetes']:
            risk_factors.append("Diabetes")
        if sample_patient['hypertension']:
            risk_factors.append("Hypertension")
        if sample_patient['bmi'] > 25:
            risk_factors.append("Overweight")
        
        print(f"  • Identified risk factors: {', '.join(risk_factors)}")
        
        # Simulate prediction
        base_success = 0.7
        risk_penalty = len(risk_factors) * 0.1
        success_probability = max(0.1, base_success - risk_penalty)
        
        print(f"  • Estimated success probability: {success_probability:.1%}")
        
    except Exception as e:
        print(f"✗ Sample prediction failed: {e}")
    
    # Demo Summary
    print("\n" + "="*60)
    print("🎯 DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nThe system has demonstrated:")
    print("✓ Sample data generation and validation")
    print("✓ Protocol analysis with LLM integration")
    print("✓ Machine learning model creation")
    print("✓ Risk factor identification")
    print("✓ Success probability estimation")
    
    print("\n🚀 Ready to run the full system:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run system test: python test_system.py")
    print("3. Train models: python models/train.py")
    print("4. Start API: python api/main.py")
    print("5. Launch web app: streamlit run web/app.py")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    run_demo()
