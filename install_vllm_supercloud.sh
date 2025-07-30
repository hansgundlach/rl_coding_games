#!/bin/bash
# vLLM Installation Script for MIT Supercloud V100s (Offline)
# Run this on a machine with internet first, then transfer to Supercloud

set -e

echo "🚀 vLLM Installation Script for Supercloud V100s"
echo "================================================"

# Configuration
VLLM_VERSION="v0.8.3"  # Latest stable version
PYTHON_VERSION="3.9"   # Compatible with Supercloud
WHEEL_DIR="./vllm_wheels"
REQUIREMENTS_FILE="./vllm_requirements.txt"

# Create wheel directory
mkdir -p "$WHEEL_DIR"

echo "📦 Step 1: Downloading vLLM wheel and dependencies..."

# Download vLLM wheel for CUDA 12.1 (V100 compatible)
pip download vllm==0.8.3 \
    --platform linux_x86_64 \
    --python-version ${PYTHON_VERSION//./} \
    --implementation cp \
    --abi cp${PYTHON_VERSION//./}m \
    --dest "$WHEEL_DIR" \
    --no-deps

echo "📥 Downloaded vLLM wheel to $WHEEL_DIR"

# Download all dependencies
echo "📦 Downloading dependencies..."
pip download vllm==0.8.3 \
    --dest "$WHEEL_DIR" \
    --requirement-file /dev/null

# Create requirements file for offline installation
cat > "$REQUIREMENTS_FILE" << EOF
# vLLM Requirements for Supercloud V100s (Offline Installation)
# Generated on $(date)

# Core vLLM package
vllm==0.8.3

# Essential dependencies (these should be downloaded in wheels/)
torch>=2.1.0
transformers>=4.40.0
tokenizers>=0.19.1
huggingface_hub>=0.23.2
fastapi>=0.107.4
uvicorn>=0.22.0
pydantic>=2.0
numpy>=1.20.0
sentencepiece>=0.1.98
protobuf>=3.20.0
grpcio>=1.38.0
triton>=2.1.0
EOF

echo "📋 Created requirements file: $REQUIREMENTS_FILE"

# Create installation script for Supercloud
cat > "./install_vllm_offline.sh" << 'EOF'
#!/bin/bash
# Offline vLLM Installation Script for Supercloud
# Run this ON Supercloud after transferring wheel files

set -e

echo "🔧 Installing vLLM offline on Supercloud..."

# Check if in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Not in a virtual environment. Activating venv..."
    source venv/bin/activate || {
        echo "❌ No virtual environment found. Please create one first:"
        echo "   python -m venv venv"
        echo "   source venv/bin/activate"
        exit 1
    }
fi

# Set offline environment variables
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_DO_NOT_TRACK=1

echo "🌐 Set offline mode environment variables"

# Install from local wheels
if [[ -d "./vllm_wheels" ]]; then
    echo "📦 Installing vLLM from local wheels..."
    pip install --no-index --find-links ./vllm_wheels vllm
    echo "✅ vLLM installed successfully!"
else
    echo "❌ Error: vllm_wheels directory not found"
    echo "   Please ensure you've transferred the wheel files to Supercloud"
    exit 1
fi

# Verify installation
echo "🧪 Verifying installation..."
python -c "import vllm; print(f'✅ vLLM {vllm.__version__} installed successfully')" || {
    echo "❌ vLLM installation verification failed"
    exit 1
}

# Test CUDA availability (should work on V100s)
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA version: {torch.version.cuda}')
"

echo ""
echo "🎉 vLLM installation complete!"
echo ""
echo "💡 Usage Tips for Supercloud V100s:"
echo "   1. Always set offline environment variables before using vLLM"
echo "   2. Use gpu_memory_utilization=0.85 for V100s (16GB VRAM)"
echo "   3. Set max_model_len=2048 for memory efficiency"
echo "   4. Models must be pre-cached locally (no internet access)"
echo ""
echo "🚀 Ready to use vLLM with your GRPO training!"
EOF

chmod +x "./install_vllm_offline.sh"

echo ""
echo "✅ vLLM offline installation package prepared!"
echo ""
echo "📋 Next Steps:"
echo "1. Transfer these files to Supercloud:"
echo "   - $WHEEL_DIR/ (entire directory)"
echo "   - $REQUIREMENTS_FILE"  
echo "   - install_vllm_offline.sh"
echo ""
echo "2. On Supercloud, run:"
echo "   ./install_vllm_offline.sh"
echo ""
echo "3. Update your GRPO config:"
echo "   Set vllm.enabled: True in configs/grpo_code_execution.yaml"
echo ""
echo "🚀 Ready for 3-5x faster inference on V100s!"