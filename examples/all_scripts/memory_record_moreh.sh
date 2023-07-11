# Capture the process ID (PID) of the training script
pid=$1
task=$2
model=$3
batch_size=$4
log_dir="logs_${model}"
mkdir -p $log_dir
# Initialize a variable to track peak memory usage
peak_memory=0
# Monitor GPU memory usage using moreh-smi in a loop
while true; do
    # Query GPU memory usage of the training process
    output=$(moreh-smi)  
    gpu_memory=$(echo "$output" | awk -F '|' '/\*/ {gsub(/[^0-9]/,"",$5); print $5}')
    
    # Update peak memory usage
    if [[ -n $gpu_memory && $gpu_memory =~ ^[0-9]+$ ]] && (( gpu_memory > peak_memory )); then
        peak_memory=$gpu_memory
        total_memory=$(echo "$output" | awk -F '|' '/\*/ {gsub(/[^0-9]/,"",$6); print $6}')
    fi
    # Check if the training process has completed or terminated
    if ! kill -0 $pid 2>/dev/null; then
        break
    fi

    # Sleep for a certain interval before querying GPU memory again (e.g., every second)
    sleep 1
done    
# Log the peak memory usage
echo Task: $task, Model: $model, Batch_size: $batch_size, Peak memory: $peak_memory, Total memory: $total_memory >> "../log_terminal/${task}_memory.log"