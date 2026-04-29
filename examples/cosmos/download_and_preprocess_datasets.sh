dataset_dir='gr1_dataset'
train_dir=$dataset_dir/train
test_dir=$dataset_dir/test

# Download and Preprocess Training Dataset
hf download nvidia/GR1-100 --repo-type dataset --local-dir datasets/benchmark_train/hf_gr1/ && \
mkdir -p datasets/benchmark_train/gr1/videos && \
mv datasets/benchmark_train/hf_gr1/gr1/*mp4 datasets/benchmark_train/gr1/videos && \
mv datasets/benchmark_train/hf_gr1/metadata.csv datasets/benchmark_train/gr1/

python create_prompts_for_gr1_dataset.py --dataset_path datasets/benchmark_train/gr1

# Download Eval Dataset
hf download nvidia/EVAL-175 --repo-type dataset --local-dir dream_gen_benchmark


# Rename dataset directory
mkdir $dataset_dir
mv datasets/benchmark_train/gr1 $train_dir
mv dream_gen_benchmark/gr1_object $test_dir
echo Download training data to $train_dir
echo Download test data to $test_dir

# Clean up staging directories
rm -rf datasets/ dream_gen_benchmark/
