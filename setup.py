"""
Setup script to prepare the project environment.
Creates necessary directories and checks dependencies.
"""

import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary project directories."""
    directories = [
        'data',
        'models',
        'results',
        'utils'
    ]
    
    print("Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}/")
    
    # Create .gitkeep files to track empty directories
    for directory in ['data', 'models', 'results']:
        gitkeep_path = Path(directory) / '.gitkeep'
        gitkeep_path.touch(exist_ok=True)


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        'torch',
        'torchvision',
        'streamlit',
        'plotly',
        'numpy',
        'cv2',
        'sklearn',
        'matplotlib',
        'seaborn',
        'PIL',
        'kaggle',
        'trimesh',
        'nltk',
        'pandas',
        'tqdm',
        'reportlab'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            elif package == 'kaggle':
                # Import kaggle without triggering authentication
                import importlib.util
                spec = importlib.util.find_spec('kaggle')
                if spec is None:
                    raise ImportError()
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
        except Exception as e:
            # Kaggle API may fail on import if not configured, but package is installed
            if package == 'kaggle' and 'Missing' in str(e):
                print(f"  ✓ {package} (installed, needs configuration)")
            else:
                print(f"  ⚠️ {package} (error: {e})")
                missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️  Missing packages detected!")
        print("Install them using:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def check_kaggle_api():
    """Check if Kaggle API is configured."""
    print("\nChecking Kaggle API configuration...")
    
    kaggle_json_paths = [
        Path.home() / '.kaggle' / 'kaggle.json',
        Path('C:/Users') / os.environ.get('USERNAME', '') / '.kaggle' / 'kaggle.json'
    ]
    
    kaggle_configured = False
    for path in kaggle_json_paths:
        if path.exists():
            print(f"  ✓ Found kaggle.json at {path}")
            kaggle_configured = True
            break
    
    if not kaggle_configured:
        print("  ⚠️  Kaggle API not configured")
        print("\n" + "="*70)
        print("KAGGLE API SETUP REQUIRED")
        print("="*70)
        print("\nOption 1: Run the interactive setup script (EASIEST)")
        print("  python setup_kaggle.py")
        print("\nOption 2: Manual setup")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Click 'Create New API Token'")
        print("  3. Save kaggle.json to:")
        print(f"     {Path.home() / '.kaggle' / 'kaggle.json'}")
        print("\nOption 3: Download dataset manually")
        print("  1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("  2. Click 'Download' button")
        print("  3. Extract to: data/chest_xray/")
        print("="*70)
        return False
    
    return True


def download_nltk_data():
    """Download required NLTK data."""
    print("\nDownloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("  ✓ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not download NLTK data: {e}")
        return False


def main():
    """Run all setup steps."""
    print("="*70)
    print("AI PNEUMONIA DETECTION SYSTEM - SETUP")
    print("="*70)
    
    # Create directories
    create_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check Kaggle API
    kaggle_ok = check_kaggle_api()
    
    # Download NLTK data
    if deps_ok:
        download_nltk_data()
    
    # Final summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    
    if deps_ok and kaggle_ok:
        print("\n✓ Setup completed successfully!")
        print("\nNext steps:")
        print("  1. Train the model: python train.py")
        print("  2. Evaluate: python evaluate.py")
        print("  3. Run web app: streamlit run app.py")
    else:
        print("\n⚠️  Setup incomplete. Please address the issues above.")
        if not deps_ok:
            print("  → Install dependencies: pip install -r requirements.txt")
        if not kaggle_ok:
            print("  → Configure Kaggle API (see instructions above)")
    
    print("="*70)


if __name__ == "__main__":
    main()
