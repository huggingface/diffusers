data_dir=YOUR_VIDEO_DIR # contains all the files end with _seed*.mp4
out_dir=YOUR_OUTPUT_DIR

echo $out_dir

find $data_dir -name "*.mp4" | wc -l

# LLM as judge (instruction following)
uv run examples/gr00t-dreams/inference_video_IF.py \
  --video-dir $data_dir \
  --output-dir $out_dir

# Aggregate results
python aggregate_scores.py $out_dir | tee $out_dir/summary.txt
echo "Save to $out_dir/summary.txt"
