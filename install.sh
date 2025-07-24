#!/bin/bash
# Complete Installation Script for Qwen ConnectX RL Training

set -e  # Exit on any error

echo "🚀 Starting Qwen ConnectX RL Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
    echo -e "${BLUE}📋 Step $1: $2${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running in virtual environment
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Not running in virtual environment!"
        echo "It's recommended to create and activate a virtual environment:"
        echo "  python3.10 -m venv venv"
        echo "  source venv/bin/activate"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Virtual environment detected: $VIRTUAL_ENV"
    fi
}

# Check Python version
check_python() {
    print_step "1" "Checking Python version"
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_success "Python $PYTHON_VERSION detected"
    
    if [[ "$PYTHON_VERSION" < "3.10" ]]; then
        print_error "Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
}

# Check CUDA availability
check_cuda() {
    print_step "2" "Checking CUDA/GPU availability"
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read gpu; do
            print_success "GPU detected: $gpu"
        done
    else
        print_warning "NVIDIA GPU not detected - training will be CPU-only (very slow)"
        read -p "Continue with CPU training? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Please install CUDA drivers and try again"
            exit 1
        fi
    fi
}

# Install dependencies
install_deps() {
    print_step "3" "Installing dependencies"
    
    # Upgrade pip
    echo "Upgrading pip..."
    python3 -m pip install --upgrade pip setuptools wheel
    
    # Install PyTorch (adjust for your CUDA version)
    if command -v nvidia-smi &> /dev/null; then
        echo "Installing PyTorch with CUDA support..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "Installing PyTorch CPU-only..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install project dependencies
    echo "Installing project dependencies..."
    python3 -m pip install -r requirements.txt
    
    # Install project in development mode
    echo "Installing project..."
    python3 -m pip install -e .
    
    print_success "Dependencies installed"
}

# Setup Hugging Face
setup_huggingface() {
    print_step "4" "Setting up Hugging Face access"
    
    # Check if already logged in
    if python3 -c "from huggingface_hub import whoami; whoami()" &> /dev/null; then
        HF_USER=$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])")
        print_success "Already logged into Hugging Face as: $HF_USER"
    else
        print_warning "Hugging Face login required for Qwen model access"
        echo "Please get your token from: https://huggingface.co/settings/tokens"
        read -p "Enter your Hugging Face token (or press Enter to skip): " HF_TOKEN
        
        if [[ -n "$HF_TOKEN" ]]; then
            echo "$HF_TOKEN" | python3 -c "from huggingface_hub import login; import sys; login(sys.stdin.read().strip())"
            print_success "Hugging Face login successful"
        else
            print_warning "Skipped Hugging Face login - you'll need to set HUGGINGFACE_HUB_TOKEN later"
        fi
    fi
}

# Setup Weights & Biases
setup_wandb() {
    print_step "5" "Setting up Weights & Biases (optional)"
    
    # Check if already logged in
    if python3 -c "import wandb; wandb.api.api_key" &> /dev/null; then
        print_success "Already logged into Weights & Biases"
    else
        read -p "Setup Weights & Biases for training logs? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            print_warning "Skipped W&B setup - training will still work without logging"
        else
            echo "Please get your API key from: https://wandb.ai/authorize"
            read -p "Enter your W&B API key (or press Enter to skip): " WANDB_KEY
            
            if [[ -n "$WANDB_KEY" ]]; then
                echo "$WANDB_KEY" | wandb login
                print_success "Weights & Biases login successful"
            else
                print_warning "Skipped W&B login - set WANDB_API_KEY environment variable later"
            fi
        fi
    fi
}

# Pre-download model
download_model() {
    print_step "6" "Pre-downloading Qwen model (optional)"
    
    read -p "Pre-download Qwen model (~6GB)? This will speed up first training run (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "Downloading Qwen/Qwen2.5-3B model..."
        python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
print('Downloading Qwen model...')
cache_dir = './model_cache'
os.makedirs(cache_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B', cache_dir=cache_dir, torch_dtype='float16')
print('Model download complete!')
"
        print_success "Model downloaded to ./model_cache/"
    else
        print_warning "Model will be downloaded on first training run"
    fi
}

# Run tests
run_tests() {
    print_step "7" "Running installation tests"
    
    echo "Testing installation..."
    if python3 test_pipeline.py; then
        print_success "All tests passed! 🎉"
    else
        print_error "Some tests failed - check the output above"
        exit 1
    fi
}

# Create directories
create_dirs() {
    echo "Creating necessary directories..."
    mkdir -p checkpoints model_cache wandb logs
    print_success "Directories created"
}

# Main installation flow
main() {
    echo "======================================"
    echo "🎮 Qwen ConnectX RL Installation"
    echo "======================================"
    echo ""
    
    check_venv
    check_python
    check_cuda
    create_dirs
    install_deps
    setup_huggingface
    setup_wandb
    download_model
    run_tests
    
    echo ""
    echo "======================================"
    echo "🎉 Installation Complete!"
    echo "======================================"
    echo ""
    echo "Quick start:"
    echo "  python training/qwen_ppo_train.py --config configs/qwen_ppo.yaml"
    echo ""
    echo "For detailed usage, see:"
    echo "  - COMPLETE_SETUP_GUIDE.md"
    echo "  - USAGE_GUIDE.md"
    echo ""
    echo "Training logs will be available at:"
    echo "  - W&B Dashboard: https://wandb.ai/your-username/connectx-qwen-rl"
    echo "  - Local checkpoints: ./checkpoints/qwen_ppo/"
    echo ""
}

# Run main function
main "$@"