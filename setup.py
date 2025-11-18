"""
Setup script to prepare the environment and verify datasets.
"""
import os
from pathlib import Path
import subprocess
import sys


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    dirs = [
        'models',
        'logs',
        'results',
        'figures',
        'outputs'
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_name}/")
    
    return True


def check_datasets():
    """Check if datasets exist."""
    print("\nChecking datasets...")
    
    fire_dataset = Path('fire-dataset/archive/data')
    smoke_dataset = Path('smoke-dataset')
    
    fire_exists = fire_dataset.exists()
    smoke_exists = smoke_dataset.exists()
    
    if fire_exists:
        print("✓ Fire dataset found")
        # Count samples
        train_images = len(list((fire_dataset / 'train' / 'images').glob('*.jpg')))
        print(f"  - Train images: {train_images}")
    else:
        print("⚠ Fire dataset not found at fire-dataset/archive/data")
    
    if smoke_exists:
        print("✓ Smoke dataset found")
        # Count samples
        train_smoke = len(list((smoke_dataset / 'train' / 'smoke').glob('*.jpg')))
        train_cloud = len(list((smoke_dataset / 'train' / 'cloud').glob('*.jpg')))
        train_other = len(list((smoke_dataset / 'train' / 'other').glob('*.jpg')))
        print(f"  - Train images: {train_smoke + train_cloud + train_other}")
        print(f"    - Smoke: {train_smoke}")
        print(f"    - Cloud: {train_cloud}")
        print(f"    - Other: {train_other}")
    else:
        print("⚠ Smoke dataset not found at smoke-dataset/")
    
    if not fire_exists and not smoke_exists:
        print("\n❌ No datasets found!")
        print("Please ensure your datasets are in the correct locations:")
        print("  - fire-dataset/archive/data/")
        print("  - smoke-dataset/")
        return False
    
    return True


def install_dependencies():
    """Install required packages."""
    print("\nInstalling dependencies...")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("\n✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Failed to install dependencies")
        print("Please run manually: pip install -r requirements.txt")
        return False


def check_cuda():
    """Check if CUDA is available."""
    print("\nChecking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  - CUDA Version: {torch.version.cuda}")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ CUDA not available - will use CPU")
            print("  Training will be slower on CPU")
    except ImportError:
        print("⚠ PyTorch not installed yet")
    
    return True


def main():
    print("="*70)
    print("COMPUTER VISION PROJECT - SETUP")
    print("="*70)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    create_directories()
    
    # Check datasets
    datasets_ok = check_datasets()
    
    # Ask about installing dependencies
    print("\n" + "="*70)
    response = input("Install dependencies? (y/n): ").lower().strip()
    
    if response == 'y':
        if install_dependencies():
            check_cuda()
    
    # Summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    
    if datasets_ok:
        print("\n✅ Setup complete! You're ready to start training.")
        print("\nQuick start:")
        print("  python train.py --help             # See training options")
        print("  python train.py --model-type baseline --model-name simple --dataset smoke --epochs 10")
        print("\nFor more information:")
        print("  - Read QUICKSTART.md for a 5-minute guide")
        print("  - Read README.md for full documentation")
    else:
        print("\n⚠ Setup incomplete - datasets not found")
        print("Please add your datasets and run setup.py again")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

