# Check input variables
# ==================================
train_script=train.sh
model_batchsize_file=model_batchsize.txt

while getopts t:m:b:s: flag
do
    case "${flag}" in
        t) task=${OPTARG};;
        m) model=${OPTARG};;
        b) batch_size=${OPTARG};;
        s) train_script=${OPTARG};;
    esac
done

# If task is specified, setup env for task. Otherwise, set run_all_tasks=True, env for each task will be setup later
if [[ "$@" == *"-t"* ]]; then 
  bash setup.sh $task
else
  echo -e "*** Note: -t (task name) not specified, will run all tasks. \
          \n To run single task, declare <-t task_name> \
          \n - dreambooth \
          \n - instruct_pix2pix \
          \n - controlnet \
          \n - text_to_image \
          \n - unconditional_image_generation \
          \n - textual_inversion \
  "
  run_all_tasks=True
fi

# If model is not specified, set run_all_model=True and warn user that all models in this task will be run
if [[ ! "$@" == *"-m"* ]]; then  
  echo "*** Note: -m (model name) not specified, will run all models in model_batchsize_file for task $task"
  run_all_model=True
fi


# Create functions to execute training and delete env
# ==================================
# Function to execute training
execute_training() {
            model=$1
            batch_size=$2
            echo Model: $model
            echo Batch_size: $batch_size

            model_name=${model#*/}

            LOG_DIR=../logs/$task
            mkdir -p $LOG_DIR
            log_dir="${LOG_DIR}/${model_name}.json"

            OUTPUT_DIR=../outputs/$task
            mkdir -p $OUTPUT_DIR
            output_dir="${OUTPUT_DIR}/${model_name}"            

            mkdir -p ../log_terminal/${task}
            bash $train_script -t $task -m $model -o $output_dir -l $log_dir -b $batch_size >> "../log_terminal/${task}/${model_name}.log" 2>&1 &
            pid=$!
            bash ../all_scripts/memory_record_moreh.sh $pid $task $model $batch_size
            echo Done training model $model for task $task
}

# Function to delete env
base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')
execute_deleting_env() {
            task=$1
            echo "deleting env for task $task.."
            source ${base_env}/etc/profile.d/conda.sh
            conda deactivate
            conda env remove -n ${task}
}

# Check condition to run all tasks, all models in task, or specific model in task
# ==================================
if [[ "$run_all_tasks" == "True" ]]; then # Task is not provided, run all tasks
    echo "Running all tasks.."
    for task in dreambooth instruct_pix2pix controlnet text_to_image unconditional_image_generation textual_inversion; do
        echo "Running task $task"
        bash setup.sh $task
        cd $task
        while read model batch_size ; do
            execute_training $model $batch_size
        done < $model_batchsize_file
        # execute_deleting_env $task
        cd ..
    done
elif [[ "$run_all_model" == "True" ]]; then # Task is provided, model is not provided, run all models in task
    echo "Running all models in task $task"
    cd $task
    while read model batch_size ; do
        execute_training $model $batch_size
    done < $model_batchsize_file
    # execute_deleting_env $task
else # Task and model are provided, run specific model in task
    echo Training the model $model in task $task
    cd $task
    execute_training $model $batch_size
    # execute_deleting_env $task
fi

echo Done