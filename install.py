"""
Installation script for Clinical Trial Outcome Predictor
"""
import subprocess
import sys
import os

def install_package(package):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main installation function"""
    print("🏥 Clinical Trial Outcome Predictor - Installation")
    print("=" * 50)
    
    print("\n📦 Installing required packages...")
    
    # Core packages
    core_packages = [
        "torch",
        "pandas", 
        "numpy",
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "streamlit",
        "plotly"
    ]
    
    # Optional packages
    optional_packages = [
        "matplotlib",
        "seaborn",
        "jupyter",
        "openai",
        "langchain"
    ]
    
    print("\n🔧 Installing core packages...")
    for package in core_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"  ✅ {package} installed successfully")
        else:
            print(f"  ❌ Failed to install {package}")
    
    print("\n🎨 Installing optional packages...")
    for package in optional_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"  ✅ {package} installed successfully")
        else:
            print(f"  ⚠️ Failed to install {package} (optional)")
    
    print("\n🎯 Installation completed!")
    print("\n🚀 Next steps:")
    print("1. Run system test: python test_system.py")
    print("2. Run demo: python demo.py")
    print("3. Use interactive startup: python start.py")
    
    print("\n📚 For more information, see README.md and PROJECT_OVERVIEW.md")

if __name__ == "__main__":
    main()
