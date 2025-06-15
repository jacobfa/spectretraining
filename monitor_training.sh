#!/bin/bash

# 📊 SPECTRE Training Monitor 📊
# ===============================
# Quick monitoring script for SPECTRE training sessions

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# Emoji definitions
CHART="📊"
TIMER="⏱️"
GPU="🔧"
INFO="ℹ️"
CHECK="✅"
TMUX="🖥️"
FIRE="🔥"

print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}║                     ${CHART} SPECTRE Training Monitor ${CHART}                  ║${RESET}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
}

print_info() {
    echo -e "${BLUE}${INFO}${RESET} ${BOLD}$1${RESET}"
}

print_success() {
    echo -e "${GREEN}${CHECK}${RESET} ${BOLD}$1${RESET}"
}

show_tmux_sessions() {
    echo -e "${PURPLE}${BOLD}Active Tmux Sessions:${RESET}"
    if command -v tmux &> /dev/null; then
        if tmux list-sessions 2>/dev/null | grep -q .; then
            tmux list-sessions | while read -r line; do
                echo -e "  ${GREEN}●${RESET} $line"
            done
        else
            echo -e "  ${DIM}No active tmux sessions${RESET}"
        fi
    else
        echo -e "  ${RED}tmux not installed${RESET}"
    fi
    echo ""
}

show_gpu_status() {
    echo -e "${YELLOW}${BOLD}GPU Status:${RESET}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
        while IFS=',' read -r idx name util mem_used mem_total temp; do
            util=$(echo $util | xargs)
            mem_used=$(echo $mem_used | xargs)
            mem_total=$(echo $mem_total | xargs)
            temp=$(echo $temp | xargs)
            
            echo -e "  ${GPU} GPU $idx: ${BOLD}$name${RESET}"
            echo -e "    Utilization: ${util}% | Memory: ${mem_used}MB/${mem_total}MB | Temp: ${temp}°C"
        done
    else
        echo -e "  ${RED}nvidia-smi not available${RESET}"
    fi
    echo ""
}

show_latest_run() {
    local runs_dir="${1:-./runs}"
    
    echo -e "${CYAN}${BOLD}Latest Training Run:${RESET}"
    
    if [[ ! -d "$runs_dir" ]]; then
        echo -e "  ${DIM}No runs directory found${RESET}"
        return
    fi
    
    # Find latest run
    local latest_run=$(find "$runs_dir" -name "spectre_vit_*" -type d | sort | tail -n 1)
    
    if [[ -z "$latest_run" ]]; then
        echo -e "  ${DIM}No training runs found${RESET}"
        return
    fi
    
    echo -e "  ${BOLD}Run:${RESET} $(basename "$latest_run")"
    
    # Show latest checkpoint info
    if [[ -f "$latest_run/checkpoints/latest.pth" ]]; then
        local epoch_info=$(python3 -c "
import torch
try:
    checkpoint = torch.load('$latest_run/checkpoints/latest.pth', map_location='cpu')
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc1', 0)
    print(f'Epoch: {epoch}, Best Acc: {best_acc:.2f}%')
except:
    print('Unable to read checkpoint')
" 2>/dev/null)
        echo -e "  ${BOLD}Progress:${RESET} $epoch_info"
    fi
    
    # Show recent log entries
    if [[ -f "$latest_run/logs/training_rank_0.log" ]]; then
        echo -e "  ${BOLD}Recent Log:${RESET}"
        tail -n 3 "$latest_run/logs/training_rank_0.log" | while read -r line; do
            echo -e "    ${DIM}$line${RESET}"
        done
    fi
    
    echo ""
    echo -e "${YELLOW}${BOLD}Quick Actions:${RESET}"
    echo -e "  ${CYAN}View logs:${RESET}      tail -f $latest_run/logs/training_rank_0.log"
    echo -e "  ${CYAN}TensorBoard:${RESET}    tensorboard --logdir $latest_run/tensorboard"
    echo -e "  ${CYAN}View stats:${RESET}     cat $latest_run/train_stats.json | jq '.[-1]'"
    echo ""
}

show_training_stats() {
    local runs_dir="${1:-./runs}"
    
    echo -e "${GREEN}${BOLD}Training Statistics:${RESET}"
    
    # Find latest run with stats
    local latest_run=$(find "$runs_dir" -name "spectre_vit_*" -type d | sort | tail -n 1)
    
    if [[ -z "$latest_run" || ! -f "$latest_run/train_stats.json" ]]; then
        echo -e "  ${DIM}No training statistics available${RESET}"
        return
    fi
    
    # Extract latest stats using Python
    python3 -c "
import json
import sys

try:
    with open('$latest_run/train_stats.json', 'r') as f:
        stats = json.load(f)
    
    if not stats:
        print('  No statistics available')
        sys.exit(0)
    
    latest = stats[-1]
    print(f\"  ${BOLD}Latest Epoch:${RESET} {latest.get('epoch', 'N/A')}\")
    print(f\"  ${BOLD}Train Loss:${RESET} {latest.get('loss', 0):.4f}\")
    print(f\"  ${BOLD}Train Acc@1:${RESET} {latest.get('acc1', 0):.2f}%\")
    print(f\"  ${BOLD}Learning Rate:${RESET} {latest.get('lr', 0):.2e}\")
    print(f\"  ${BOLD}Epoch Time:${RESET} {latest.get('time', 0):.1f}s\")
    
    # Show validation stats if available
    val_file = '$latest_run/val_stats.json'
    try:
        with open(val_file, 'r') as f:
            val_stats = json.load(f)
        if val_stats:
            val_latest = val_stats[-1]
            print(f\"  ${BOLD}Val Loss:${RESET} {val_latest.get('loss', 0):.4f}\")
            print(f\"  ${BOLD}Val Acc@1:${RESET} {val_latest.get('acc1', 0):.2f}%\")
            if val_latest.get('is_best', False):
                print(f\"  ${GREEN}★ New best model!${RESET}\")
    except:
        pass
        
except Exception as e:
    print(f'  Error reading stats: {e}')
" 2>/dev/null
    
    echo ""
}

# Monitor mode - refresh every few seconds
monitor_mode() {
    local runs_dir="${1:-./runs}"
    local interval="${2:-5}"
    
    echo -e "${FIRE} ${BOLD}Starting monitor mode (refresh every ${interval}s)${RESET}"
    echo -e "${DIM}Press Ctrl+C to exit${RESET}"
    echo ""
    
    while true; do
        clear
        print_header
        show_tmux_sessions
        show_gpu_status
        show_latest_run "$runs_dir"
        show_training_stats "$runs_dir"
        
        echo -e "${DIM}Last updated: $(date)${RESET}"
        sleep "$interval"
    done
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] [RUNS_DIR]"
    echo ""
    echo "Options:"
    echo "  -m, --monitor [INTERVAL]  Monitor mode with auto-refresh (default: 5s)"
    echo "  -t, --tmux                Show tmux sessions only"
    echo "  -g, --gpu                 Show GPU status only"
    echo "  -s, --stats               Show training statistics only"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Arguments:"
    echo "  RUNS_DIR                  Directory containing training runs (default: ./runs)"
    echo ""
    echo "Examples:"
    echo "  $0                        # Show status once"
    echo "  $0 -m                     # Monitor mode (refresh every 5s)"
    echo "  $0 -m 10                  # Monitor mode (refresh every 10s)"
    echo "  $0 -t                     # Show only tmux sessions"
    echo "  $0 ./my_runs              # Monitor specific runs directory"
}

# Main function
main() {
    local runs_dir="./runs"
    local monitor=false
    local interval=5
    local tmux_only=false
    local gpu_only=false
    local stats_only=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--monitor)
                monitor=true
                if [[ $2 =~ ^[0-9]+$ ]]; then
                    interval=$2
                    shift
                fi
                shift
                ;;
            -t|--tmux)
                tmux_only=true
                shift
                ;;
            -g|--gpu)
                gpu_only=true
                shift
                ;;
            -s|--stats)
                stats_only=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                runs_dir="$1"
                shift
                ;;
        esac
    done
    
    # Execute based on options
    if [[ "$monitor" == true ]]; then
        monitor_mode "$runs_dir" "$interval"
    elif [[ "$tmux_only" == true ]]; then
        print_header
        show_tmux_sessions
    elif [[ "$gpu_only" == true ]]; then
        print_header
        show_gpu_status
    elif [[ "$stats_only" == true ]]; then
        print_header
        show_training_stats "$runs_dir"
    else
        # Show everything once
        print_header
        show_tmux_sessions
        show_gpu_status
        show_latest_run "$runs_dir"
        show_training_stats "$runs_dir"
    fi
}

# Run main function
main "$@" 
