# Check input variables
# ==================================

if [[ "$@" == *"--task"* ]]; then # --task: task name
  # If the flag is set, assign the value from the user prompt
  task="$(echo "$@" | sed -n 's/.*--task \([^ ]*\).*/\1/p')"
else
  echo -e "Declare --task task_name> \
          \n - dreambooth \
          \n - instruct_pix2pix \
          \n - controlnet \
          \n - text_to_image \
          \n - unconditional_image_generation \
  "
  exit 1
fi

if [[ "$@" == *"--model"* ]]; then # --model: model name
  model="$(echo "$@" | sed -n 's/.*--model \([^ ]*\).*/\1/p')"
  run_mode=1 # run single model
else 
  run_mode=2 # run all models in model_batchsize_file
  echo "Note: --model not specified, will run all models in model_batchsize_file"
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

            output_dir="${OUTPUT_DIR}/${model_name}"
            log_dir="${LOG_DIR}/${model_name}.json"


            cd $task
            # echo "Running script $train_script task $task model $model bs $batch_size output $output_dir log $log_dir"
            bash $train_script --task $task --model_name $model --batch_size $batch_size --output_dir $output_dir --log_dir $log_dir                

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