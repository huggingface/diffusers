# Check input variables
# ==================================

if [[ "$@" == *"--task"* ]]; then # --task: task name
  # If the flag is set, assign the value from the user prompt
  task="$(echo "$@" | sed -n 's/.*--task \([^ ]*\).*/\1/p')"
else
  echo "Usage: $0 <declare task name>"
  exit 1
fi

if [[ "$@" == *"--model"* ]]; then # --model: model name
  model="$(echo "$@" | sed -n 's/.*--model \([^ ]*\).*/\1/p')"
  run_mode=1
else 
  run_mode=2
fi

if [[ "$@" == *"--batch_size"* ]]; then # --batch_size: batch size
  batch_size="$(echo "$@" | sed -n 's/.*--batch_size \([^ ]*\).*/\1/p')"
fi

train_script=train.sh
if [[ "$@" == *"--train_script"* ]]; then # --train_script: training script
  train_script="$(echo "$@" | sed -n 's/.*--train_script \([^ ]*\).*/\1/p')"
fi

model_batchsize_file=model_batchsize.txt
if [[ "$@" == *"--model_batchsize_file"* ]]; then # --model_batchsize_file: file containing model name and batch size respectively
  model_batchsize_file="$(echo "$@" | sed -n 's/.*--model_batchsize_file \([^ ]*\).*/\1/p')"
fi


# Check input files
# ==================================
bash all_scripts/check_file_exist.sh $train_script $task
bash all_scripts/check_file_exist.sh $model_batchsize_file $task

# Create env for task
# ==================================
bash setup.sh $task

# Create log and output dir inside examples folder
# ==================================

LOG_DIR=$(pwd)/logs/$task
mkdir -p $LOG_DIR

OUTPUT_DIR=$(pwd)/outputs/$task
mkdir -p $OUTPUT_DIR

# Go into task folder and run train script
# ==================================
execute_training() {
            model=$1
            batch_size=$2
            echo Model: $model
            echo Batch_size: $batch_size

            model_name=${model#*/}

            terminal_log_file="${LOG_DIR}/${model_name}_terminal.log"
            memory_log_file="${LOG_DIR}/${model_name}_memory.log"

            output_dir="${OUTPUT_DIR}/${model_name}"
            log_dir="${LOG_DIR}/${model_name}.json"

            commands_to_run="
                cd $task
                bash $train_script $task $model $batch_size $output_dir $log_dir                
            " 
            bash record.sh 0 $memory_log_file $terminal_log_file "$commands_to_run"

            echo Done training $model
}

if [ "$run_mode" == "2" ]; then
    echo Training all models in $model_batchsize_file
    while read model batch_size ; do
        execute_training $model $batch_size
    done < $task/$model_batchsize_file
else
    echo Training the model $model
    execute_training $model $batch_size
fi

# Delete env for task
# ==================================
base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')

# echo "deleting env.."
# source ${base_env}/etc/profile.d/conda.sh
# conda deactivate
# conda env remove -n ${task}

echo Done