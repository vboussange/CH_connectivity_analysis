#!/bin/bash

# Configuration
GPUS=(0 1 2 3 4 5)
groups=("Amphibians" "Bees" "Beetles" "Birds" "Bryophytes" "Mammals" "Reptiles" "Fishes" "Vascular_plants" "Spiders" "Dragonflies" "Grasshoppers" "Butterflies" "Fungi" "Molluscs" "Lichens" "May_stone_caddisflies")
habs=("Aqu" "Ter")
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/elasticity_analysis_$(date +%Y%m%d_%H%M%S).log"
PROGRESS_FILE="${LOG_DIR}/progress_status.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR" > /dev/null 2>&1

# Function to log messages (silent - only to file)
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" >> "$LOG_FILE"
}

# Function to update progress (silent - only to file)
update_progress() {
    local completed=$1
    local total=$2
    local percentage=$((completed * 100 / total))
    echo "Progress: $completed/$total jobs completed ($percentage%)" > "$PROGRESS_FILE"
    log "Progress: $completed/$total jobs completed ($percentage%)"
}

# Generate all jobs
jobs=()
for group in "${groups[@]}"; do
    for hab in "${habs[@]}"; do
        jobs+=("$group $hab")
    done
done

total_jobs=${#jobs[@]}
completed=0

log "Starting analysis with $total_jobs jobs on ${#GPUS[@]} GPUs"
update_progress $completed $total_jobs

# Process jobs in batches equal to number of GPUs
for ((i=0; i<${#jobs[@]}; i+=${#GPUS[@]})); do
    # Start a batch of jobs (one per GPU)
    for j in "${!GPUS[@]}"; do
        index=$((i+j))
        
        # Check if we still have jobs to process
        if [ $index -lt ${#jobs[@]} ]; then
            read -r group hab <<< "${jobs[$index]}"
            gpu=${GPUS[$j]}
            
            job_log="${LOG_DIR}/${group}_${hab}_gpu${gpu}.log"
            log "Starting job: Group=${group}, Habitat=${hab}, on GPU ${gpu}"
            
            # Run the job and redirect output to job-specific log file (silently)
            python group_elasticity_analysis.py --group "$group" --hab "$hab" --gpu_id "$gpu" > "$job_log" 2>&1 &
        fi
    done
    
    # Wait for all jobs in this batch to complete
    wait
    
    # Update progress after batch completion
    completed=$((i + ${#GPUS[@]}))
    if [ $completed -gt $total_jobs ]; then
        completed=$total_jobs
    fi
    update_progress $completed $total_jobs
done

log "All elasticity calculations completed successfully."
echo "All elasticity calculations completed successfully." > "$PROGRESS_FILE"