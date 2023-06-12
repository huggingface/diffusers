file=$1
parent_dir=$2

file=$parent_dir/$file

[[ ! -f $file ]] && echo "File ${file} not exist" && exit 1

echo $file