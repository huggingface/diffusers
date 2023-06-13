# Check input files
# ==================================
if [ -z "$1" ]; then
  echo "Usage: $0 <declare task name>"
  exit 1
elif [ -z "$2" ]; then
  echo "Usage: $0 <declare run mode after task name: 1 - single model; 2 - multi model>. Default: 2"
fi

if [ "$2" == "1" ] && [ -z "$3" ] && [ -z "$4" ]; then
  echo "Usage: $0 <declare model name and batch size after task name and run mode>"
  exit 1
fi
task=$1
echo Task: $task

run_mode=${2:-2}
model=$3
batch_size=$4

train_script=${5:-"train.sh"}
bash all_scripts/check_file_exist.sh $train_script $task


model_batchsize_file=${6:-model_batchsize.txt}
bash all_scripts/check_file_exist.sh $model_batchsize_file $task


# Create env for task
# ==================================
bash setup.sh $task

# Create log and output dir inside task folder
# ==================================

LOG_DIR=$(pwd)/logs/$task
mkdir -p $LOG_DIR

OUTPUT_DIR=$(pwd)/outputs/$task
mkdir -p $OUTPUT_DIR

# Go into task folder and run train script
# ==================================
# cd $task
execute_training() {
            model=$1
            batch_size=$2
            echo Model: model
            echo Batch_size: batch_size

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
    echo Running multi-model training
    while read model batch_size ; do
        execute_training $model $batch_size
    done < $task/$model_batchsize_file
else
    echo Running single-model training
    execute_training $model $batch_size
fi
base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')

echo "deleting env.."
source ${base_env}/etc/profile.d/conda.sh
conda deactivate
conda env remove -n ${task}

echo Done