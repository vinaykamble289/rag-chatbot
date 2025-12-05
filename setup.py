"""
Setup script to install dependencies with proper versions
Run this instead of pip install -r requirements.txt
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("=" * 60)
    print("Installing PDF RAG Chatbot Dependencies")
    print("=" * 60)
    
    packages = [
        "streamlit==1.29.0",
        "pypdf2==3.0.1",
        "sentence-transformers==2.2.2",
        "faiss-cpu==1.7.4",
        "protobuf==3.20.3",  # Install this before transformers
        "transformers==4.35.0",
        "torch==2.1.0",
    ]
    
    for i, package in enumerate(packages, 1):
        print(f"\n[{i}/{len(packages)}] Installing {package}...")
        try:
            install_package(package)
            print(f"✅ {package} installed successfully")
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ All dependencies installed successfully!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  streamlit run app.py")
    print("\nOr test with:")
    print("  python test_local.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
