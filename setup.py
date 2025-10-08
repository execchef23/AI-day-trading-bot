#!/usr/bin/env python3
"""Setup script for AI Day Trading Bot"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{description}...")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        return False
    return True

def main():
    """Main setup function"""
    print("🤖 AI Day Trading Bot - Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists('venv'):
        print("\n📦 Creating virtual environment...")
        if not run_command('python -m venv venv', 'Virtual environment creation'):
            sys.exit(1)
    else:
        print("\n✅ Virtual environment already exists")
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = 'venv\\Scripts\\activate.bat &&'
    else:  # Unix/MacOS
        activate_cmd = 'source venv/bin/activate &&'
    
    # Install requirements
    pip_install_cmd = f'{activate_cmd} pip install --upgrade pip'
    if not run_command(pip_install_cmd, 'Upgrading pip'):
        sys.exit(1)
    
    requirements_cmd = f'{activate_cmd} pip install -r requirements.txt'
    if not run_command(requirements_cmd, 'Installing requirements'):
        print("\n⚠️  Some packages might have failed to install.")
        print("This is often due to missing system dependencies.")
        print("\nFor common issues:")
        print("- TA-Lib: Install system dependencies first")
        print("- TensorFlow: Ensure compatible Python version")
        print("- XGBoost/LightGBM: Usually install without issues")
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("\n📝 Creating .env file from template...")
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ .env file created from template")
            print("\n⚠️  Please edit .env file with your API keys:")
            print("   - ALPHA_VANTAGE_API_KEY")
            print("   - POLYGON_API_KEY")
            print("   - DATABASE_URL (optional)")
        else:
            print("❌ .env.example not found")
    else:
        print("\n✅ .env file already exists")
    
    # Create necessary directories
    directories = ['data/raw', 'data/processed', 'data/historical', 'logs', 'models', 'plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print(f"\n✅ Created {len(directories)} necessary directories")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("3. Test data fetching: python scripts/fetch_data_example.py")
    print("4. Train models: python scripts/train_models_example.py")
    print("5. Run dashboard: streamlit run dashboard.py")

if __name__ == '__main__':
    main()