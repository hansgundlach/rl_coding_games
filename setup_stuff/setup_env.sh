#!/bin/bash
# Environment setup script for easy GPU type and offline mode configuration

set -e

print_header() {
    echo "============================================="
    echo "$1"
    echo "============================================="
}

print_info() {
    echo "ℹ️  $1"
}

print_success() {
    echo "✅ $1"
}

print_header "GPU Environment Setup"

echo "Select your environment:"
echo "1) MIT Supercloud (V100, offline mode)"
echo "2) Lambda Labs (A100, online mode)"
echo "3) Custom configuration"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        print_info "Setting up for MIT Supercloud (V100, offline)"
        export GPU_TYPE=v100
        export OFFLINE_MODE=true
        echo "export GPU_TYPE=v100" >> ~/.bashrc
        echo "export OFFLINE_MODE=true" >> ~/.bashrc
        print_success "Environment configured for MIT Supercloud"
        print_info "Make sure models are pre-downloaded to ./model_cache/"
        ;;
    2)
        print_info "Setting up for Lambda Labs (A100, online)"
        export GPU_TYPE=a100
        export OFFLINE_MODE=false
        echo "export GPU_TYPE=a100" >> ~/.bashrc
        echo "export OFFLINE_MODE=false" >> ~/.bashrc
        print_success "Environment configured for Lambda Labs"
        ;;
    3)
        echo ""
        echo "GPU Type options:"
        echo "- v100: Tesla V100 (no BF16 support)"
        echo "- a100: A100/H100 (BF16 support)"
        echo "- auto: Auto-detect based on GPU name"
        echo ""
        read -p "Enter GPU type (v100/a100/auto): " gpu_type
        
        echo ""
        echo "Offline mode options:"
        echo "- true: Use only local model cache (for no internet)"
        echo "- false: Download models as needed (requires internet)"
        echo ""
        read -p "Enable offline mode? (true/false): " offline_mode
        
        export GPU_TYPE=$gpu_type
        export OFFLINE_MODE=$offline_mode
        echo "export GPU_TYPE=$gpu_type" >> ~/.bashrc
        echo "export OFFLINE_MODE=$offline_mode" >> ~/.bashrc
        print_success "Custom environment configured"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
print_header "Current Environment Settings"
echo "GPU_TYPE: $GPU_TYPE"
echo "OFFLINE_MODE: $OFFLINE_MODE"
echo ""
print_info "Settings have been added to ~/.bashrc"
print_info "Run 'source ~/.bashrc' or restart your shell to apply"
echo ""

if [ "$OFFLINE_MODE" = "true" ]; then
    print_info "To pre-download models for offline use, run:"
    echo "  python3 -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', cache_dir='./model_cache'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B', cache_dir='./model_cache')\""
fi
