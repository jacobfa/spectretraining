#!/bin/bash

# 🔄 SPECTRE ViT Resume Training Script with Tmux (FIXED) 🔄
# ==========================================================
# Resume SPECTRE ViT training from latest checkpoint with DDP in tmux session

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
RESTART="🔄"
CHART="📊"
SAVE="💾"
GPU="🔧"
TIMER="⏱️"
CHECK="✅"
WARN="⚠️"
INFO="ℹ️"
STAR="⭐"
TMUX="🖥️"

# Header function
print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}║              ${RESTART} SPECTRE ViT Resume Training (FIXED) ${RESTART}         ║${RESET}"
    echo -e "${CYAN}║          ${TMUX} Tmux Session + DDP + Checkpoint Resume ${TMUX}           ║${RESET}"
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

# Find latest checkpoint function
find_latest_checkpoint() {
    local runs_dir="${1:-./runs}"
    local latest_checkpoint=""
    local latest_epoch=0
    local latest_run=""
    
    print_section "Checkpoint Discovery"
    
    if [[ ! -d "$runs_dir" ]]; then
        print_error "Runs directory '$runs_dir' not found!"
        return 1
    fi
    
    print_info "Searching for checkpoints in: ${BOLD}$runs_dir${RESET}"
    
    # Find all run directories
    for run_dir in "$runs_dir"/spectre_vit_*; do
        if [[ -d "$run_dir/checkpoints" ]]; then
            print_info "Found run: $(basename "$run_dir")"
            
            # Check for latest checkpoint in this run
            if [[ -f "$run_dir/checkpoints/latest.pth" ]]; then
                # Extract epoch from checkpoint if possible
                local checkpoint_file="$run_dir/checkpoints/latest.pth"
                local epoch_info=$(python3 -c "
import torch
import sys
try:
    checkpoint = torch.load('$checkpoint_file', map_location='cpu')
    epoch = checkpoint.get('epoch', 0)
    print(epoch)
except Exception as e:
    print(0, file=sys.stderr)
    print(f'Error: {e}', file=sys.stderr)
" 2>/dev/null)
                
                echo -e "  ${DIM}Latest checkpoint: epoch $epoch_info${RESET}"
                
                if [[ $epoch_info -gt $latest_epoch ]]; then
                    latest_epoch=$epoch_info
                    latest_checkpoint="$checkpoint_file"
                    latest_run="$run_dir"
                fi
            fi
            
            # Also check for specific epoch checkpoints
            for checkpoint in "$run_dir/checkpoints"/checkpoint_epoch_*.pth; do
                if [[ -f "$checkpoint" ]]; then
                    local epoch_num=$(basename "$checkpoint" | sed 's/.*epoch_\([0-9]\+\).*/\1/')
                    echo -e "  ${DIM}Found epoch checkpoint: $epoch_num${RESET}"
                    
                    if [[ $epoch_num -gt $latest_epoch ]]; then
                        latest_epoch=$epoch_num
                        latest_checkpoint="$checkpoint"
                        latest_run="$run_dir"
                    fi
                fi
            done
        fi
    done
    
    if [[ -n "$latest_checkpoint" ]]; then
        print_success "Latest checkpoint found!"
        echo -e "  ${BOLD}Run:${RESET} $(basename "$latest_run")"
        echo -e "  ${BOLD}Checkpoint:${RESET} $latest_checkpoint"
        echo -e "  ${BOLD}Epoch:${RESET} $latest_epoch"
        
        # Export for use in calling script
        export RESUME_CHECKPOINT="$latest_checkpoint"
        export RESUME_RUN_DIR="$latest_run"
        export RESUME_EPOCH="$latest_epoch"
        return 0
    else
        print_warning "No checkpoints found in $runs_dir"
        return 1
    fi
}

# Test training script function
test_training_script() {
    print_section "Training Script Validation"
    
    # Test basic imports
    print_info "Testing Python imports..."
    python3 -c "
import torch
import torchvision
from spectre import create_spectre_vit
print('✅ All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
" 2>/dev/null

    if [[ $? -eq 0 ]]; then
        print_success "Python imports working correctly"
    else
        print_error "Python import test failed!"
        print_info "Please check your Python environment and dependencies"
        return 1
    fi
    
    # Test script syntax
    print_info "Testing training script syntax..."
    python3 -m py_compile train_spectre_vit.py
    if [[ $? -eq 0 ]]; then
        print_success "Training script syntax is valid"
    else
        print_error "Training script has syntax errors!"
        return 1
    fi
    
    return 0
}

# Create tmux session function
create_tmux_session() {
    local session_name="$1"
    local command="$2"
    
    print_section "Tmux Session Setup"
    
    # Check if tmux is installed
    if ! command -v tmux &> /dev/null; then
        print_error "tmux is not installed! Please install it first:"
        echo -e "  ${BOLD}Ubuntu/Debian:${RESET} sudo apt-get install tmux"
        echo -e "  ${BOLD}CentOS/RHEL:${RESET} sudo yum install tmux"
        return 1
    fi
    
    # Kill existing session if it exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        print_warning "Session '$session_name' already exists. Killing it..."
        tmux kill-session -t "$session_name"
        sleep 2  # Give it time to properly close
    fi
    
    # Create new session with proper configuration
    print_info "Creating tmux session: ${BOLD}$session_name${RESET}"
    
    # Create session in detached mode with proper shell
    tmux new-session -d -s "$session_name" -c "$(pwd)" bash
    
    # Configure tmux session settings for better experience
    tmux set-option -t "$session_name" -g mouse on
    tmux set-option -t "$session_name" -g history-limit 10000
    tmux set-option -t "$session_name" -g default-terminal "screen-256color"
    
    # Set up environment in session
    tmux send-keys -t "$session_name" "cd $(pwd)" Enter
    tmux send-keys -t "$session_name" "clear" Enter
    
    # Show session info
    tmux send-keys -t "$session_name" "echo '🚀 SPECTRE ViT Training Session Started'" Enter
    tmux send-keys -t "$session_name" "echo 'Session: $session_name'" Enter
    tmux send-keys -t "$session_name" "echo 'Directory: $(pwd)'" Enter
    tmux send-keys -t "$session_name" "echo 'Time: $(date)'" Enter
    tmux send-keys -t "$session_name" "echo ''" Enter
    
    # Send the training command
    print_info "Starting training command in tmux session..."
    echo -e "${DIM}Command: $command${RESET}"
    
    # Add a separator for clarity
    tmux send-keys -t "$session_name" "echo '=== Starting Training Command ==='" Enter
    tmux send-keys -t "$session_name" "$command" Enter
    
    # Wait a moment for the command to start
    sleep 3
    
    # Check if session is still alive (command didn't fail immediately)
    if ! tmux has-session -t "$session_name" 2>/dev/null; then
        print_error "Session died immediately - check your command!"
        return 1
    fi
    
    return 0
}

# Enhanced attach function with better guidance
attach_tmux_session() {
    local session_name="$1"
    
    echo ""
    print_success "Training started successfully in tmux session: ${BOLD}$session_name${RESET}"
    echo ""
    
    # Show tmux commands
    echo -e "${CYAN}${BOLD}📋 Tmux Commands Reference:${RESET}"
    echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${RESET}"
    echo -e "${YELLOW}│${RESET} ${BOLD}Session Management:${RESET}                                    ${YELLOW}│${RESET}"
    echo -e "${YELLOW}│${RESET}   Attach to session:    ${GREEN}tmux attach -t $session_name${RESET}       ${YELLOW}│${RESET}"
    echo -e "${YELLOW}│${RESET}   Detach from session:  ${GREEN}Ctrl+B, then D${RESET}                    ${YELLOW}│${RESET}"
    echo -e "${YELLOW}│${RESET}   Kill session:         ${RED}tmux kill-session -t $session_name${RESET}  ${YELLOW}│${RESET}"
    echo -e "${YELLOW}│${RESET}   List all sessions:    ${GREEN}tmux list-sessions${RESET}                 ${YELLOW}│${RESET}"
    echo -e "${YELLOW}│${RESET}                                                             ${YELLOW}│${RESET}"
    echo -e "${YELLOW}│${RESET} ${BOLD}Within Session:${RESET}                                       ${YELLOW}│${RESET}"
    echo -e "${YELLOW}│${RESET}   Scroll up/down:       ${GREEN}Ctrl+B, then PageUp/PageDown${RESET}      ${YELLOW}│${RESET}"
    echo -e "${YELLOW}│${RESET}   Copy mode (scroll):   ${GREEN}Ctrl+B, then [${RESET}                   ${YELLOW}│${RESET}"
    echo -e "${YELLOW}│${RESET}   Search in history:    ${GREEN}Ctrl+B, then Ctrl+S${RESET}              ${YELLOW}│${RESET}"
    echo -e "${YELLOW}│${RESET}   Exit copy mode:       ${GREEN}q${RESET}                                 ${YELLOW}│${RESET}"
    echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${RESET}"
    echo ""
    
    # Show monitoring commands
    echo -e "${PURPLE}${BOLD}📊 Monitoring Commands:${RESET}"
    echo -e "${CYAN}┌─────────────────────────────────────────────────────────────┐${RESET}"
    if [[ -n "$RESUME_RUN_DIR" ]]; then
        echo -e "${CYAN}│${RESET} Watch training logs:   ${GREEN}tail -f $RESUME_RUN_DIR/logs/training_rank_0.log${RESET}"
        echo -e "${CYAN}│${RESET} TensorBoard:           ${GREEN}tensorboard --logdir $RESUME_RUN_DIR/tensorboard${RESET}"
    fi
    echo -e "${CYAN}│${RESET} GPU usage:             ${GREEN}watch -n 1 nvidia-smi${RESET}                           ${CYAN}│${RESET}"
    echo -e "${CYAN}│${RESET} Training monitor:      ${GREEN}./monitor_training.sh -m${RESET}                        ${CYAN}│${RESET}"
    echo -e "${CYAN}│${RESET} System resources:      ${GREEN}htop${RESET}                                           ${CYAN}│${RESET}"
    echo -e "${CYAN}└─────────────────────────────────────────────────────────────┘${RESET}"
    echo ""
    
    # Show session status
    echo -e "${BLUE}${BOLD}📈 Session Status:${RESET}"
    echo -e "  ${GREEN}✅ Session Name:${RESET} $session_name"
    echo -e "  ${GREEN}✅ Session ID:${RESET} $(tmux display-message -t "$session_name" -p '#{session_id}' 2>/dev/null || echo 'N/A')"
    echo -e "  ${GREEN}✅ Working Directory:${RESET} $(pwd)"
    echo -e "  ${GREEN}✅ Started:${RESET} $(date)"
    echo ""
    
    # Ask if user wants to attach immediately
    echo -e "${BOLD}${YELLOW}🔗 Do you want to attach to the session now? (y/N):${RESET}"
    read -r attach_choice
    
    if [[ "$attach_choice" =~ ^[Yy]$ ]]; then
        print_info "Attaching to session..."
        echo -e "${DIM}Press Ctrl+B then D to detach when needed${RESET}"
        sleep 2
        tmux attach -t "$session_name"
    else
        print_info "Session running in background."
        echo ""
        echo -e "${BOLD}${GREEN}To attach later, use:${RESET}"
        echo -e "  ${CYAN}tmux attach -t $session_name${RESET}"
        echo ""
        echo -e "${BOLD}${GREEN}To check if training is running:${RESET}"
        echo -e "  ${CYAN}tmux list-sessions${RESET}"
        echo -e "  ${CYAN}ps aux | grep python${RESET}"
        echo ""
    fi
}

# Function to check if session is healthy
check_session_health() {
    local session_name="$1"
    
    if tmux has-session -t "$session_name" 2>/dev/null; then
        local pane_count=$(tmux list-panes -t "$session_name" | wc -l)
        local window_count=$(tmux list-windows -t "$session_name" | wc -l)
        
        print_success "Session '$session_name' is running"
        echo -e "  • Windows: $window_count"
        echo -e "  • Panes: $pane_count"
        return 0
    else
        print_error "Session '$session_name' is not running"
        return 1
    fi
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
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_success "Found $GPU_COUNT GPU(s) available"
        
        # Show GPU memory info
        echo -e "${DIM}GPU Memory Status:${RESET}"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | while IFS=',' read -r idx name mem_used mem_total; do
            echo -e "  GPU $idx: $name - ${mem_used}/${mem_total}"
        done
    else
        print_warning "nvidia-smi not found - GPU detection failed"
    fi
    
    # Load required modules if available (for HPC systems)
    if command -v module &> /dev/null; then
        print_info "Loading Python and CUDA modules..."
        module load python/3.8 cuda/11.8 2>/dev/null || true
        module load python3 cuda 2>/dev/null || true
        module load pytorch 2>/dev/null || true
    fi
    
    # Test training script
    if ! test_training_script; then
        exit 1
    fi
}

# Setup distributed training environment with proper parameters
setup_ddp_environment() {
    print_section "DDP Environment Setup"
    
    # Set up environment variables for distributed training
    export WORLD_SIZE=${WORLD_SIZE}
    export MASTER_ADDR=${MASTER_ADDR}
    export MASTER_PORT=${MASTER_PORT}
    
    # NCCL configuration for better DDP performance
    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_IFNAME=lo
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    
    # OMP configuration to prevent thread conflicts
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    
    # CUDA configuration
    if [[ $WORLD_SIZE -gt 1 ]]; then
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((WORLD_SIZE-1)))
    fi
    
    # Additional PyTorch DDP settings
    export TORCH_DISTRIBUTED_DEBUG=INFO
    export TORCH_SHOW_CPP_STACKTRACES=1
    
    if [[ $WORLD_SIZE -gt 1 ]]; then
        print_info "DDP Environment configured for $WORLD_SIZE GPUs"
        print_info "Master: ${MASTER_ADDR}:${MASTER_PORT}"
        print_info "CUDA Devices: ${CUDA_VISIBLE_DEVICES}"
        print_info "NCCL Backend: Optimized for multi-GPU training"
    else
        print_info "Single GPU training configured"
    fi
}

# Main function
main() {
    print_header
    
    # Parse arguments
    RUNS_DIR="${1:-./runs}"
    SESSION_NAME="${2:-spectre_vit_training}"
    FORCE_RESUME="${3:-false}"
    
    # Default configuration (can be overridden with environment variables)
    DATA_DIR="${DATA_DIR:-/data/jacob/ImageNet/}"
    OUTPUT_DIR="${OUTPUT_DIR:-./runs}"
    
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
    
    # Advanced training configuration
    USE_ADVANCED_AUG="${USE_ADVANCED_AUG:-true}"
    MIXUP_ALPHA="${MIXUP_ALPHA:-0.2}"
    CUTMIX_ALPHA="${CUTMIX_ALPHA:-1.0}"
    MIXUP_PROB="${MIXUP_PROB:-0.5}"
    LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
    CLIP_GRAD="${CLIP_GRAD:-1.0}"
    USE_EMA="${USE_EMA:-true}"
    EMA_DECAY="${EMA_DECAY:-0.9999}"
    SCHEDULER="${SCHEDULER:-cosine}"
    
    # DDP configuration - more conservative defaults
    WORLD_SIZE="${WORLD_SIZE:-1}"
    GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
    NODE_RANK="${NODE_RANK:-0}"
    MASTER_ADDR="${MASTER_ADDR:-localhost}"
    MASTER_PORT="${MASTER_PORT:-12355}"
    
    # Auto-detect GPU count if not specified, but limit to 4 for stability
    if command -v nvidia-smi &> /dev/null && [[ "$WORLD_SIZE" == "1" ]]; then
        DETECTED_GPUS=$(nvidia-smi --list-gpus | wc -l)
        if [[ $DETECTED_GPUS -gt 1 ]]; then
            # Limit to 4 GPUs for more stable training
            WORLD_SIZE=$(( DETECTED_GPUS > 8 ? 8 : DETECTED_GPUS ))
            GPUS_PER_NODE=$WORLD_SIZE
            print_info "Auto-detected $DETECTED_GPUS GPUs, using $WORLD_SIZE for training"
        fi
    fi
    
    # Validate configuration
    validate_config
    
    # Setup DDP environment with proper parameters
    setup_ddp_environment
    
    # Find latest checkpoint
    if ! find_latest_checkpoint "$RUNS_DIR"; then
        if [[ "$FORCE_RESUME" != "true" ]]; then
            print_error "No checkpoints found! Use 'train_spectre_ddp.sh' to start new training."
            echo ""
            echo -e "${BOLD}Or use this script with 'true' as third argument to start new training:${RESET}"
            echo -e "  ${DIM}$0 $RUNS_DIR $SESSION_NAME true${RESET}"
            exit 1
        else
            print_warning "No checkpoints found but force specified. Starting new training..."
            RESUME_CHECKPOINT=""
        fi
    fi
    
    # Show configuration
    print_section "Resume Configuration"
    
    echo -e "${CYAN}${RESTART} Resume Information:${RESET}"
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        echo -e "  • Resume from: ${BOLD}$RESUME_CHECKPOINT${RESET}"
        echo -e "  • Starting epoch: ${BOLD}$((RESUME_EPOCH + 1))${RESET}"
        echo -e "  • Run directory: ${BOLD}$RESUME_RUN_DIR${RESET}"
    else
        echo -e "  • ${YELLOW}Starting new training (no checkpoint)${RESET}"
    fi
    
    echo ""
    echo -e "${PURPLE}${TMUX} Session Information:${RESET}"
    echo -e "  • Session name: ${BOLD}$SESSION_NAME${RESET}"
    echo -e "  • World size: ${BOLD}$WORLD_SIZE${RESET} processes"
    echo -e "  • GPUs per node: ${BOLD}$GPUS_PER_NODE${RESET}"
    
    echo ""
    echo -e "${GREEN}${SAVE} Training Configuration:${RESET}"
    echo -e "  • Data directory: ${BOLD}$DATA_DIR${RESET}"
    echo -e "  • Output directory: ${BOLD}$OUTPUT_DIR${RESET}"
    echo -e "  • Epochs: ${BOLD}$EPOCHS${RESET}"
    echo -e "  • Batch size (per GPU): ${BOLD}$BATCH_SIZE${RESET}"
    echo -e "  • Total batch size: ${BOLD}$((BATCH_SIZE * WORLD_SIZE))${RESET}"
    echo -e "  • Learning rate: ${BOLD}$LR${RESET}"
    
    echo ""
    echo -e "${STAR}${PURPLE} Advanced Training Features:${RESET}"
    echo -e "  • Advanced augmentation: ${BOLD}$USE_ADVANCED_AUG${RESET}"
    echo -e "  • MixUp alpha: ${BOLD}$MIXUP_ALPHA${RESET}"
    echo -e "  • CutMix alpha: ${BOLD}$CUTMIX_ALPHA${RESET}"
    echo -e "  • Label smoothing: ${BOLD}$LABEL_SMOOTHING${RESET}"
    echo -e "  • Gradient clipping: ${BOLD}$CLIP_GRAD${RESET}"
    echo -e "  • EMA enabled: ${BOLD}$USE_EMA${RESET}"
    echo -e "  • Scheduler: ${BOLD}$SCHEDULER${RESET}"
    
    # Confirmation
    echo ""
    echo -e "${YELLOW}${TIMER} Ready to resume training!${RESET}"
    echo -e "${BOLD}Continue? (Y/n):${RESET}"
    read -r continue_choice
    
    if [[ "$continue_choice" =~ ^[Nn]$ ]]; then
        print_info "Training cancelled by user"
        exit 0
    fi
    
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
    
    # Add arguments
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
    
    # Add advanced training arguments
    if [[ "$USE_ADVANCED_AUG" == "true" ]]; then
        TRAIN_CMD+=" --use-advanced-aug"
    fi
    TRAIN_CMD+=" --mixup-alpha=${MIXUP_ALPHA}"
    TRAIN_CMD+=" --cutmix-alpha=${CUTMIX_ALPHA}"
    TRAIN_CMD+=" --mixup-prob=${MIXUP_PROB}"
    TRAIN_CMD+=" --label-smoothing=${LABEL_SMOOTHING}"
    TRAIN_CMD+=" --clip-grad=${CLIP_GRAD}"
    if [[ "$USE_EMA" == "true" ]]; then
        TRAIN_CMD+=" --use-ema"
    fi
    TRAIN_CMD+=" --ema-decay=${EMA_DECAY}"
    TRAIN_CMD+=" --scheduler=${SCHEDULER}"
    
    # Add wavelet option if enabled
    if [[ "$USE_WAVELET" == "true" ]]; then
        TRAIN_CMD+=" --use-wavelet"
    fi
    
    # Add resume option if checkpoint found
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        TRAIN_CMD+=" --resume='${RESUME_CHECKPOINT}'"
    fi
    
    # Create and start tmux session
    print_section "${ROCKET} Starting Training in Tmux"
    
    if create_tmux_session "$SESSION_NAME" "$TRAIN_CMD"; then
        print_success "Training session created successfully!"
        
        # Give it a moment to start
        sleep 3
        
        # Check if the session is still running (training didn't immediately crash)
        if check_session_health "$SESSION_NAME"; then
            attach_tmux_session "$SESSION_NAME"
        else
            print_error "Training session crashed immediately!"
            print_info "This usually indicates a configuration or environment issue."
            echo ""
            print_info "Troubleshooting suggestions:"
            echo -e "  • Check GPU availability: ${CYAN}nvidia-smi${RESET}"
            echo -e "  • Try single-GPU mode: ${CYAN}WORLD_SIZE=1 $0${RESET}"
            echo -e "  • Test environment: ${CYAN}python test_ddp_setup.py${RESET}"
            echo -e "  • Check dataset path: ${CYAN}ls -la ${DATA_DIR}${RESET}"
            echo ""
        fi
    else
        print_error "Failed to create tmux session"
        exit 1
    fi
}

# Usage function
usage() {
    echo "Usage: $0 [RUNS_DIR] [SESSION_NAME] [FORCE]"
    echo ""
    echo "Arguments:"
    echo "  RUNS_DIR      Directory containing training runs (default: ./runs)"
    echo "  SESSION_NAME  Name for tmux session (default: spectre_vit_training)"
    echo "  FORCE         Set to 'true' to start new training if no checkpoints found"
    echo ""
    echo "Environment variables (optional):"
    echo ""
    echo "  Basic Training:"
    echo "    DATA_DIR      Path to ImageNet dataset (default: /data/jacob/ImageNet/)"
    echo "    EPOCHS        Number of epochs (default: 300)"
    echo "    BATCH_SIZE    Batch size per GPU (default: 256)"
    echo "    LR            Learning rate (default: 1e-3)"
    echo "    USE_WAVELET   Enable wavelet refinement (default: false)"
    echo "    WORLD_SIZE    Number of processes for DDP (auto-detected, max 4)"
    echo ""
    echo "  Advanced Training:"
    echo "    USE_ADVANCED_AUG  Enable advanced augmentations (default: true)"
    echo "    MIXUP_ALPHA       MixUp alpha parameter (default: 0.2)"
    echo "    CUTMIX_ALPHA      CutMix alpha parameter (default: 1.0)"
    echo "    MIXUP_PROB        Probability of MixUp/CutMix (default: 0.5)"
    echo "    LABEL_SMOOTHING   Label smoothing epsilon (default: 0.1)"
    echo "    CLIP_GRAD         Gradient clipping norm (default: 1.0)"
    echo "    USE_EMA           Enable EMA (default: true)"
    echo "    EMA_DECAY         EMA decay rate (default: 0.9999)"
    echo "    SCHEDULER         LR scheduler type (default: cosine)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Resume with advanced features"
    echo "  $0 ./my_runs my_session              # Custom runs dir and session name"
    echo "  $0 ./runs training_session true      # Force start even without checkpoints"
    echo "  WORLD_SIZE=1 $0                      # Single GPU training"
    echo "  EPOCHS=100 BATCH_SIZE=128 $0         # Custom basic parameters"
    echo ""
    echo "  Advanced examples:"
    echo "  USE_ADVANCED_AUG=false $0            # Disable advanced augmentations"
    echo "  MIXUP_ALPHA=0.4 CUTMIX_ALPHA=1.5 $0  # Stronger augmentation"
    echo "  USE_EMA=true EMA_DECAY=0.999 $0       # Custom EMA settings"
    echo "  SCHEDULER=step LABEL_SMOOTHING=0.2 $0 # Custom scheduler and smoothing"
}

# Handle help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Run main function
main "$@" 
