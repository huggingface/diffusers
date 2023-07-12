# Capture the process ID (PID) of the training script
pid=$1
task=$2
model=$3
batch_size=$4
# Initialize a variable to track peak memory usage
peak_memory=0
# Monitor GPU memory usage using nvidia-smi in a loop
while true; do
    # Get the GPU memory usage using nvidia-smi (replace 0 with your GPU index if necessary)
    gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    # Update the peak memory if the current memory usage is higher
    if (( gpu_memory > peak_memory )); then
        peak_memory=$gpu_memory
    fi
    # Check if the training process has completed or terminated
    if ! kill -0 $pid 2>/dev/null; then
        break
    fi

    # Sleep for a certain interval before querying GPU memory again (e.g., every second)
    sleep 1
done    
# Log the peak memory usage
echo Task: $task, Model: $model, Batch size: $batch_size, Peak memory: $peak_memory >> "../log_terminal/${task}_memory.log"