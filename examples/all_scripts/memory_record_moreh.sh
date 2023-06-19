device_id=$1

echo Using moreh device $device_id

set -f

query_process_info() {
    moreh-smi > /tmp/moreh_smi_output.txt
    
    memory=$(grep -n "$device_id     |  KT AI" /tmp/moreh_smi_output.txt | awk -F'GB ' '{print $2}' | awk -F'MiB' '{print $1}')

    echo $memory
}


while true; do 
    info=$(query_process_info)
    if [[ -z $info ]]; then
	continue
    else
    	echo $info 
    fi 
    sleep 2s
done