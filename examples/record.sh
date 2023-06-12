device=${1:-0}
memory_log_file=${2:-"memory.log"}
terminal_log_file=${3:-"terminal.log"}
commands_to_run=$4


if [[ $HOSTNAME =~ "haca100" ]]; then
    export CUDA_VISIBLE_DEVICES=$device
    bash memory_record_moreh.sh $device 2>&1 >> $memory_log_file &
else
    export MOREH_VISIBLE_DEVICES=$device
    bash all_scripts/memory_record_moreh.sh $device 2>&1 >> $memory_log_file &
fi

daemon_pid=$!

bash -c "$commands_to_run" 2>&1 | tee $terminal_log_file

kill -9 $daemon_pid