"""
Interactive Kaggle API Setup Helper
Guides user through setting up Kaggle API credentials step-by-step.
"""

import os
import json
from pathlib import Path


def setup_kaggle_credentials():
    """Interactive setup for Kaggle API credentials."""
    
    print("="*70)
    print("KAGGLE API CREDENTIALS SETUP")
    print("="*70)
    print()
    
    # Determine kaggle directory path
    home_dir = Path.home()
    kaggle_dir = home_dir / '.kaggle'
    kaggle_json_path = kaggle_dir / 'kaggle.json'
    
    print("Step 1: Check if kaggle.json already exists")
    print(f"Looking for: {kaggle_json_path}")
    
    if kaggle_json_path.exists():
        print(f"✓ Found existing kaggle.json at {kaggle_json_path}")
        
        # Verify it has required fields
        try:
            with open(kaggle_json_path, 'r') as f:
                config = json.load(f)
            
            if 'username' in config and 'key' in config:
                print("✓ Credentials appear valid")
                print(f"  Username: {config['username']}")
                print()
                return True
            else:
                print("⚠️ kaggle.json exists but missing username or key")
                print("  Will recreate...")
        except Exception as e:
            print(f"⚠️ Error reading kaggle.json: {e}")
            print("  Will recreate...")
    else:
        print("✗ kaggle.json not found")
    
    print()
    print("="*70)
    print("GETTING YOUR KAGGLE API CREDENTIALS")
    print("="*70)
    print()
    print("Follow these steps:")
    print()
    print("1. Go to: https://www.kaggle.com/account")
    print("   (You'll need to sign in or create a Kaggle account)")
    print()
    print("2. Scroll down to the 'API' section")
    print()
    print("3. Click 'Create New API Token'")
    print()
    print("4. A file 'kaggle.json' will download to your Downloads folder")
    print()
    print("5. Open that file - you'll see something like:")
    print('   {"username":"your_username","key":"abc123def456..."}')
    print()
    print("="*70)
    print()
    
    # Get credentials from user
    print("Enter your Kaggle credentials (from the downloaded kaggle.json):")
    print()
    
    username = input("Kaggle Username: ").strip()
    if not username:
        print("✗ Username cannot be empty")
        return False
    
    api_key = input("Kaggle API Key (long string): ").strip()
    if not api_key:
        print("✗ API Key cannot be empty")
        return False
    
    print()
    
    # Create kaggle directory if it doesn't exist
    try:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {kaggle_dir}")
    except Exception as e:
        print(f"✗ Error creating directory: {e}")
        return False
    
    # Create kaggle.json
    credentials = {
        "username": username,
        "key": api_key
    }
    
    try:
        with open(kaggle_json_path, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        print(f"✓ Created: {kaggle_json_path}")
        
        # Set file permissions (important for security)
        try:
            os.chmod(kaggle_json_path, 0o600)
            print("✓ Set secure file permissions")
        except:
            print("⚠️ Could not set file permissions (Windows - this is OK)")
        
        print()
        print("="*70)
        print("✓ KAGGLE API SETUP COMPLETE!")
        print("="*70)
        print()
        print("You can now download datasets from Kaggle.")
        print("Next step: Run 'python train.py' to start training")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating kaggle.json: {e}")
        return False


def verify_kaggle_setup():
    """Verify Kaggle API is working."""
    print("Verifying Kaggle API setup...")
    
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print("✓ Kaggle API authenticated successfully!")
        print(f"  Username: {api.config_values['username']}")
        return True
        
    except ImportError:
        print("✗ Kaggle package not installed")
        print("  Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return False


def main():
    """Main setup workflow."""
    
    # Setup credentials
    if not setup_kaggle_credentials():
        print()
        print("Setup failed. Please try again.")
        return
    
    # Verify setup
    print()
    if verify_kaggle_setup():
        print()
        print("🎉 All set! You're ready to download the dataset and train the model.")
        print()
        print("Next steps:")
        print("  1. Download dataset: python -c \"from utils.data_prep import download_kaggle_dataset; download_kaggle_dataset()\"")
        print("  2. Train model: python train.py")
    else:
        print()
        print("⚠️ Setup completed but verification failed.")
        print("Please check your credentials and try again.")


if __name__ == "__main__":
    main()
