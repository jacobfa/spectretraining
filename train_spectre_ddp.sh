#!/bin/bash

# 🌈 SPECTRE ViT Distributed Training Script 🌈
# ================================================
# Beautiful and colorful DDP training launcher for SPECTRE Vision Transformer
# Supports multi-GPU and multi-node training with comprehensive logging

# Color definitions for beautiful output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# Emoji definitions
ROCKET="🚀"
FIRE="🔥"
TARGET="🎯"
CHART="📊"
SAVE="💾"
GPU="🔧"
TIMER="⏱️"
CHECK="✅"
WARN="⚠️"
INFO="ℹ️"
STAR="⭐"

# Header function
print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}║                    ${STAR} SPECTRE ViT DDP Training ${STAR}                    ║${RESET}"
    echo -e "${CYAN}║              ${FIRE} Spectral Token Routing Vision Transformer ${FIRE}       ║${RESET}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
}

# Info printing function
print_info() {
    echo -e "${BLUE}${INFO}${RESET} ${BOLD}$1${RESET}"
}

# Success printing function  
print_success() {
    echo -e "${GREEN}${CHECK}${RESET} ${BOLD}$1${RESET}"
}

# Warning printing function
print_warning() {
    echo -e "${YELLOW}${WARN}${RESET} ${BOLD}$1${RESET}"
}

# Error printing function
print_error() {
    echo -e "${RED}❌${RESET} ${BOLD}$1${RESET}"
}

# Section header function
print_section() {
    echo ""
    echo -e "${PURPLE}${BOLD}═══ $1 ═══${RESET}"
    echo ""
}

# Configuration validation function
validate_config() {
    print_section "Configuration Validation"
    
    # Check if Python training script exists
    if [[ ! -f "train_spectre_vit.py" ]]; then
        print_error "Training script 'train_spectre_vit.py' not found!"
        exit 1
    fi
    print_success "Training script found"
    
    # Check if SPECTRE model exists
    if [[ ! -f "spectre.py" ]]; then
        print_error "SPECTRE model file 'spectre.py' not found!"
        exit 1
    fi
    print_success "SPECTRE model found"
    
    # Check if ImageNet directory exists
    if [[ ! -d "$DATA_DIR" ]]; then
        print_warning "ImageNet directory '$DATA_DIR' not found!"
        print_info "Please ensure the data directory exists before training"
    else
        print_success "ImageNet directory found"
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_success "Found $GPU_COUNT GPU(s) available"
        
        # Show GPU info
        echo -e "${DIM}GPU Details:${RESET}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | \
        while IFS=',' read -r name memory; do
            echo -e "  ${GPU} $name ($memory)"
        done
    else
        print_warning "nvidia-smi not found - GPU detection failed"
    fi
}

# Training configuration display
show_config() {
    print_section "Training Configuration"
    
    echo -e "${CYAN}${TARGET} Model Configuration:${RESET}"
    echo -e "  • Architecture: SPECTRE Vision Transformer"
    echo -e "  • Image Size: ${BOLD}${IMG_SIZE}x${IMG_SIZE}${RESET}"
    echo -e "  • Patch Size: ${BOLD}${PATCH_SIZE}x${PATCH_SIZE}${RESET}"
    echo -e "  • Embedding Dim: ${BOLD}${EMBED_DIM}${RESET}"
    echo -e "  • Depth: ${BOLD}${DEPTH}${RESET} layers"
    echo -e "  • Attention Heads: ${BOLD}${N_HEADS}${RESET}"
    echo -e "  • MLP Ratio: ${BOLD}${MLP_RATIO}${RESET}"
    if [[ "$USE_WAVELET" == "true" ]]; then
        echo -e "  • Wavelet Refinement: ${GREEN}${CHECK} Enabled${RESET}"
    else
        echo -e "  • Wavelet Refinement: ${DIM}Disabled${RESET}"
    fi
    
    echo ""
    echo -e "${YELLOW}${CHART} Training Configuration:${RESET}"
    echo -e "  • Epochs: ${BOLD}${EPOCHS}${RESET}"
    echo -e "  • Batch Size (per GPU): ${BOLD}${BATCH_SIZE}${RESET}"
    echo -e "  • Total Batch Size: ${BOLD}$((BATCH_SIZE * WORLD_SIZE))${RESET}"
    echo -e "  • Learning Rate: ${BOLD}${LR}${RESET}"
    echo -e "  • Weight Decay: ${BOLD}${WEIGHT_DECAY}${RESET}"
    echo -e "  • Warmup Epochs: ${BOLD}${WARMUP_EPOCHS}${RESET}"
    
    echo ""
    echo -e "${PURPLE}${GPU} System Configuration:${RESET}"
    echo -e "  • World Size: ${BOLD}${WORLD_SIZE}${RESET} processes"
    echo -e "  • GPUs per Node: ${BOLD}${GPUS_PER_NODE}${RESET}"
    echo -e "  • Workers per GPU: ${BOLD}${NUM_WORKERS}${RESET}"
    echo -e "  • Master Address: ${BOLD}${MASTER_ADDR}${RESET}"
    echo -e "  • Master Port: ${BOLD}${MASTER_PORT}${RESET}"
    
    echo ""
    echo -e "${GREEN}${SAVE} Output Configuration:${RESET}"
    echo -e "  • Data Directory: ${BOLD}${DATA_DIR}${RESET}"
    echo -e "  • Output Directory: ${BOLD}${OUTPUT_DIR}${RESET}"
    if [[ -n "$RESUME" ]]; then
        echo -e "  • Resume From: ${BOLD}${RESUME}${RESET}"
    fi
}

# Progress monitoring function
monitor_progress() {
    print_section "Training Monitor"
    print_info "Training started! Monitor progress with:"
    echo ""
    echo -e "${CYAN}${CHART} Real-time Logs:${RESET}"
    echo -e "  ${BOLD}tail -f ${OUTPUT_DIR}/spectre_vit_*/logs/training_rank_0.log${RESET}"
    echo ""
    echo -e "${PURPLE}${CHART} TensorBoard:${RESET}"
    echo -e "  ${BOLD}tensorboard --logdir ${OUTPUT_DIR}/spectre_vit_*/tensorboard${RESET}"
    echo ""
    echo -e "${YELLOW}${GPU} GPU Monitoring:${RESET}"
    echo -e "  ${BOLD}watch -n 1 nvidia-smi${RESET}"
    echo ""
}

# Cleanup function
cleanup() {
    print_section "Cleanup"
    print_info "Cleaning up processes..."
    pkill -f "train_spectre_vit.py" 2>/dev/null || true
    print_success "Cleanup completed"
}

# Signal handlers
trap cleanup EXIT
trap 'print_error "Training interrupted!"; cleanup; exit 1' INT TERM

# ===== CONFIGURATION SECTION =====
print_header

print_section "Configuration Setup"

# Default configuration
DATA_DIR="${DATA_DIR:-/data/jacob/ImageNet/}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs}"
RESUME="${RESUME:-}"

# Model configuration
IMG_SIZE="${IMG_SIZE:-224}"
PATCH_SIZE="${PATCH_SIZE:-16}"
EMBED_DIM="${EMBED_DIM:-768}"
DEPTH="${DEPTH:-12}"
N_HEADS="${N_HEADS:-12}"
MLP_RATIO="${MLP_RATIO:-4.0}"
USE_WAVELET="${USE_WAVELET:-false}"

# Training configuration
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# DDP configuration
WORLD_SIZE="${WORLD_SIZE:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-12355}"

# Auto-detect GPU count if not specified
if command -v nvidia-smi &> /dev/null && [[ "$WORLD_SIZE" == "1" ]]; then
    DETECTED_GPUS=$(nvidia-smi --list-gpus | wc -l)
    if [[ $DETECTED_GPUS -gt 1 ]]; then
        WORLD_SIZE=$DETECTED_GPUS
        GPUS_PER_NODE=$DETECTED_GPUS
        print_info "Auto-detected $DETECTED_GPUS GPUs, setting WORLD_SIZE=$WORLD_SIZE"
    fi
fi

# Validate configuration
validate_config

# Show configuration
show_config

# Confirmation prompt
echo ""
echo -e "${YELLOW}${TIMER} Ready to start training!${RESET}"
if [[ "${AUTO_START:-false}" != "true" ]]; then
    read -p "$(echo -e ${BOLD}Press Enter to continue or Ctrl+C to abort...${RESET})" -r
fi

# ===== TRAINING LAUNCH SECTION =====
print_section "${ROCKET} Launching SPECTRE ViT Training"

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="spectre_vit_${TIMESTAMP}"
SESSION_NAME="spectre_training_${TIMESTAMP}"

print_info "Starting training run: ${BOLD}${RUN_NAME}${RESET}"

# Build training command using torchrun (recommended over torch.distributed.launch)
if [[ $WORLD_SIZE -gt 1 ]]; then
    TRAIN_CMD="torchrun"
    TRAIN_CMD+=" --nproc_per_node=${GPUS_PER_NODE}"
    TRAIN_CMD+=" --nnodes=1"
    TRAIN_CMD+=" --node_rank=${NODE_RANK}"
    TRAIN_CMD+=" --master_addr=${MASTER_ADDR}"
    TRAIN_CMD+=" --master_port=${MASTER_PORT}"
    TRAIN_CMD+=" train_spectre_vit.py"
else
    # Single GPU training
    TRAIN_CMD="python train_spectre_vit.py"
fi
TRAIN_CMD+=" --data-dir='${DATA_DIR}'"
TRAIN_CMD+=" --output-dir='${OUTPUT_DIR}'"
TRAIN_CMD+=" --img-size=${IMG_SIZE}"
TRAIN_CMD+=" --patch-size=${PATCH_SIZE}"
TRAIN_CMD+=" --embed-dim=${EMBED_DIM}"
TRAIN_CMD+=" --depth=${DEPTH}"
TRAIN_CMD+=" --n-heads=${N_HEADS}"
TRAIN_CMD+=" --mlp-ratio=${MLP_RATIO}"
TRAIN_CMD+=" --epochs=${EPOCHS}"
TRAIN_CMD+=" --batch-size=${BATCH_SIZE}"
TRAIN_CMD+=" --lr=${LR}"
TRAIN_CMD+=" --weight-decay=${WEIGHT_DECAY}"
TRAIN_CMD+=" --warmup-epochs=${WARMUP_EPOCHS}"
TRAIN_CMD+=" --num-workers=${NUM_WORKERS}"

# Add wavelet option if enabled
if [[ "$USE_WAVELET" == "true" ]]; then
    TRAIN_CMD+=" --use-wavelet"
fi

# Add resume option if specified
if [[ -n "$RESUME" ]]; then
    TRAIN_CMD+=" --resume='${RESUME}'"
fi

# Display the command (truncated for readability)
echo -e "${DIM}Training Command:${RESET}"
echo -e "${DIM}${TRAIN_CMD}${RESET}" | fold -w 80 -s

echo ""

# Check if tmux is available and user wants persistent training
USE_TMUX="${USE_TMUX:-true}"

if [[ "$USE_TMUX" == "true" ]] && command -v tmux &> /dev/null; then
    print_section "🖥️  Tmux Session Setup"
    
    # Kill existing session if it exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        print_warning "Session '$SESSION_NAME' already exists. Killing it..."
        tmux kill-session -t "$SESSION_NAME"
        sleep 2
    fi
    
    # Create new tmux session
    print_info "Creating tmux session: ${BOLD}$SESSION_NAME${RESET}"
    tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)"
    
    # Configure session
    tmux set-option -t "$SESSION_NAME" -g mouse on
    tmux set-option -t "$SESSION_NAME" -g history-limit 10000
    
    # Send training command to tmux session
    tmux send-keys -t "$SESSION_NAME" "echo '🚀 SPECTRE ViT Training Session Started'" Enter
    tmux send-keys -t "$SESSION_NAME" "echo 'Session: $SESSION_NAME'" Enter
    tmux send-keys -t "$SESSION_NAME" "echo 'Time: $(date)'" Enter
    tmux send-keys -t "$SESSION_NAME" "echo ''" Enter
    tmux send-keys -t "$SESSION_NAME" "echo '=== Starting Training Command ==='" Enter
    tmux send-keys -t "$SESSION_NAME" "$TRAIN_CMD" Enter
    
    # Start monitoring info
    monitor_progress
    
    echo ""
    print_success "Training started in tmux session: ${BOLD}$SESSION_NAME${RESET}"
    echo ""
    
    # Show tmux commands
    echo -e "${CYAN}${BOLD}📋 Tmux Commands:${RESET}"
    echo -e "  ${YELLOW}Attach to session:${RESET}    tmux attach -t $SESSION_NAME"
    echo -e "  ${YELLOW}Detach from session:${RESET}  Ctrl+B, then D"
    echo -e "  ${YELLOW}Kill session:${RESET}         tmux kill-session -t $SESSION_NAME"
    echo -e "  ${YELLOW}List sessions:${RESET}        tmux list-sessions"
    echo ""
    
    # Ask if user wants to attach
    echo -e "${BOLD}Do you want to attach to the training session now? (y/N):${RESET}"
    read -r attach_choice
    
    if [[ "$attach_choice" =~ ^[Yy]$ ]]; then
        print_info "Attaching to session..."
        echo -e "${DIM}Press Ctrl+B then D to detach when needed${RESET}"
        sleep 2
        tmux attach -t "$SESSION_NAME"
    else
        print_info "Session running in background."
        echo ""
        echo -e "${BOLD}${GREEN}To attach later, use:${RESET}"
        echo -e "  ${CYAN}tmux attach -t $SESSION_NAME${RESET}"
        echo ""
        echo -e "${BOLD}${GREEN}To check training status:${RESET}"
        echo -e "  ${CYAN}./check_training.sh${RESET}"
        echo ""
    fi
    
else
    # Fallback: run with nohup if tmux not available
    print_warning "Tmux not available or disabled. Using nohup for persistent training."
    
    print_success "Launching distributed training with nohup..."
    
    # Start monitoring info
    monitor_progress
    
    # Execute training with nohup
    echo -e "${BOLD}${FIRE} Training Output:${RESET}"
    echo -e "${CYAN}${'='*70}${RESET}"
    
    # Create log file for nohup
    NOHUP_LOG="training_${TIMESTAMP}.log"
    
    # Run the training command with nohup
    nohup bash -c "$TRAIN_CMD" > "$NOHUP_LOG" 2>&1 &
    TRAIN_PID=$!
    
    print_success "Training started with PID: $TRAIN_PID"
    print_info "Logs are being written to: ${BOLD}$NOHUP_LOG${RESET}"
    
    echo ""
    echo -e "${BOLD}${GREEN}To monitor training:${RESET}"
    echo -e "  ${CYAN}tail -f $NOHUP_LOG${RESET}"
    echo -e "  ${CYAN}ps aux | grep $TRAIN_PID${RESET}"
    echo ""
    echo -e "${BOLD}${GREEN}To kill training if needed:${RESET}"
    echo -e "  ${CYAN}kill $TRAIN_PID${RESET}"
    echo ""
    
    # Don't wait for completion in nohup mode
    exit 0
fi 
