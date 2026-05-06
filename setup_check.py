"""
Setup verification script for ALS Web Application
Checks if all required dependencies and files are in place
"""

import os
import sys
import importlib

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, but you have Python {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_required_files():
    """Check if required files exist"""
    required_files = [
        'app.py',
        'logistic_regression_model.pkl',
        'templates/index.html'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            all_exist = False
    
    return all_exist

def check_directories():
    """Check and create required directories"""
    dirs = ['templates', 'uploads']
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
                print(f"✓ Created directory: {dir_name}")
            except Exception as e:
                print(f"❌ Failed to create directory {dir_name}: {e}")
                return False
        else:
            print(f"✓ Directory exists: {dir_name}")
    
    return True

def check_dependencies():
    """Check if all required Python packages are installed"""
    required_packages = {
        'flask': 'Flask',
        'parselmouth': 'Parselmouth',
        'joblib': 'joblib',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'scipy': 'SciPy',
        'speech_features': 'python-speech-features',
        'werkzeug': 'Werkzeug'
    }
    
    all_installed = True
    for import_name, package_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - NOT installed")
            all_installed = False
    
    return all_installed

def main():
    print("=" * 50)
    print("ALS Web Application - Setup Verification")
    print("=" * 50)
    print()
    
    print("1. Checking Python Version...")
    python_ok = check_python_version()
    print()
    
    print("2. Checking Required Files...")
    files_ok = check_required_files()
    print()
    
    print("3. Checking Directories...")
    dirs_ok = check_directories()
    print()
    
    print("4. Checking Python Dependencies...")
    deps_ok = check_dependencies()
    print()
    
    print("=" * 50)
    if python_ok and files_ok and dirs_ok and deps_ok:
        print("✓ ALL CHECKS PASSED!")
        print()
        print("You can now run the application:")
        print("  python app.py")
        print()
        print("Then open: http://localhost:5000")
        return 0
    else:
        print("❌ SETUP INCOMPLETE")
        print()
        if not python_ok:
            print("→ Install Python 3.8 or higher")
        if not deps_ok:
            print("→ Run: pip install -r requirements_web.txt")
        if not files_ok:
            print("→ Ensure logistic_regression_model.pkl is in the project directory")
        print()
        return 1

if __name__ == '__main__':
    sys.exit(main())
