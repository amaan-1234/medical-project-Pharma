"""
Test script for Clinical Trial Outcome Predictor
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from config.config import get_config
        print("✓ Config module imported")
        
        from utils.data_utils import load_sample_data
        print("✓ Data utils imported")
        
        from utils.model_utils import PatientDataModel
        print("✓ Model utils imported")
        
        from utils.llm_utils import ProtocolAnalyzer
        print("✓ LLM utils imported")
        
        from utils.validation_utils import DataValidator
        print("✓ Validation utils imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_data_generation():
    """Test data generation"""
    print("\nTesting data generation...")
    
    try:
        from utils.data_utils import load_sample_data, save_sample_data
        
        # Generate sample data
        data = load_sample_data()
        print(f"✓ Generated {len(data)} datasets")
        
        # Check data structure
        expected_keys = ['patient_data', 'biomarker_data', 'trial_data', 'outcome_data']
        for key in expected_keys:
            if key in data:
                print(f"  ✓ {key}: {len(data[key])} records")
            else:
                print(f"  ✗ Missing {key}")
                return False
        
        # Save data
        save_sample_data(data, "test_data")
        print("✓ Data saved successfully")
        
        return True
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False

def test_protocol_analysis():
    """Test protocol analysis"""
    print("\nTesting protocol analysis...")
    
    try:
        from utils.llm_utils import ProtocolAnalyzer, create_sample_protocol
        
        # Create analyzer
        analyzer = ProtocolAnalyzer()
        print("✓ Protocol analyzer created")
        
        # Create sample protocol
        protocol = create_sample_protocol()
        print(f"✓ Sample protocol created ({len(protocol)} characters)")
        
        # Analyze protocol
        analysis = analyzer.analyze_protocol(protocol)
        print("✓ Protocol analysis completed")
        
        # Check analysis results
        expected_keys = ['trial_phase', 'enrollment_target', 'intervention_type']
        for key in expected_keys:
            if key in analysis:
                print(f"  ✓ {key}: {analysis[key]}")
            else:
                print(f"  ✗ Missing {key}")
        
        # Calculate risk score
        risk_score = analyzer.calculate_risk_score(analysis)
        print(f"✓ Risk score calculated: {risk_score:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Protocol analysis failed: {e}")
        return False

def test_validation():
    """Test data validation"""
    print("\nTesting data validation...")
    
    try:
        from utils.validation_utils import validate_trial_data, print_validation_report
        from utils.data_utils import load_sample_data
        
        # Load data
        data = load_sample_data()
        
        # Validate data
        validation_results = validate_trial_data(data)
        print("✓ Data validation completed")
        
        # Print report
        print_validation_report(validation_results)
        
        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        from utils.model_utils import PatientDataModel, GenomicModel
        
        # Create patient model
        patient_model = PatientDataModel(
            input_size=17,
            hidden_sizes=[128, 64, 32],
            output_size=1
        )
        print("✓ Patient model created")
        
        # Create genomic model
        genomic_model = GenomicModel(
            sequence_length=1000,
            embedding_dim=64,
            num_filters=128,
            filter_sizes=[3, 4, 5]
        )
        print("✓ Genomic model created")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("CLINICAL TRIAL OUTCOME PREDICTOR - SYSTEM TEST")
    print("="*60)
    
    tests = [
        test_imports,
        test_data_generation,
        test_protocol_analysis,
        test_validation,
        test_model_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run model training: python models/train.py")
        print("2. Start API server: python api/main.py")
        print("3. Launch web interface: streamlit run web/app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
