"""
Startup script for Clinical Trial Outcome Predictor
"""
import sys
import subprocess
import os
from pathlib import Path

def print_banner():
    """Print the system banner"""
    print("="*70)
    print("🏥 AI-POWERED CLINICAL TRIAL OUTCOME PREDICTOR")
    print("="*70)
    print("Multi-modal AI system combining deep learning and LLMs")
    print("for predicting clinical trial success rates")
    print("="*70)

def show_menu():
    """Show the main menu"""
    print("\n📋 Available Options:")
    print("1. 🧪 Run System Test")
    print("2. 🎯 Run Demo")
    print("3. 🚀 Train Models")
    print("4. 🌐 Start API Server")
    print("5. 💻 Launch Web Interface")
    print("6. 📊 View Data Explorer")
    print("7. ❓ Help & Documentation")
    print("8. 🚪 Exit")
    print()

def run_command(command, description):
    """Run a command with description"""
    print(f"\n🚀 {description}")
    print(f"Running: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ {description} interrupted by user")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'pandas', 'numpy', 'scikit-learn',
        'fastapi', 'uvicorn', 'streamlit', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before continuing.")
        return
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == "1":
                # Run system test
                run_command("python test_system.py", "System Test")
                
            elif choice == "2":
                # Run demo
                run_command("python demo.py", "System Demo")
                
            elif choice == "3":
                # Train models
                print("\n🚀 Starting Model Training...")
                print("This will take several minutes. Please wait...")
                run_command("python models/train.py", "Model Training")
                
            elif choice == "4":
                # Start API server
                print("\n🌐 Starting API Server...")
                print("API will be available at: http://localhost:8000")
                print("Press Ctrl+C to stop the server")
                run_command("python api/main.py", "API Server")
                
            elif choice == "5":
                # Launch web interface
                print("\n💻 Launching Web Interface...")
                print("Web app will open at: http://localhost:8501")
                print("Press Ctrl+C to stop the server")
                run_command("streamlit run web/app.py", "Web Interface")
                
            elif choice == "6":
                # Data explorer
                print("\n📊 Launching Data Explorer...")
                print("This will open a Jupyter notebook for data exploration")
                run_command("jupyter notebook notebooks/", "Data Explorer")
                
            elif choice == "7":
                # Help and documentation
                print("\n📚 Help & Documentation")
                print("=" * 50)
                print("📖 README.md - Complete project documentation")
                print("🔧 requirements.txt - Required Python packages")
                print("⚙️ config/config.py - Configuration settings")
                print("📁 data/ - Sample clinical trial data")
                print("🤖 models/ - Machine learning models")
                print("🌐 api/ - FastAPI backend server")
                print("💻 web/ - Streamlit web interface")
                print("🛠️ utils/ - Utility functions")
                print("\n🚀 Quick Start:")
                print("1. Install dependencies: pip install -r requirements.txt")
                print("2. Run test: python test_system.py")
                print("3. Train models: python models/train.py")
                print("4. Start API: python api/main.py")
                print("5. Launch web: streamlit run web/app.py")
                
            elif choice == "8":
                # Exit
                print("\n👋 Thank you for using the Clinical Trial Outcome Predictor!")
                print("Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-8.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
