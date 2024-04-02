import os
from PIL import Image
import json
from tqdm import tqdm

# Define the directories
img_root = '/home/gkalstn000/dataset/inpainting/image'
mask_root = '/home/gkalstn000/dataset/inpainting/conditioning_image'

# Load the metadata
json_file = '/home/gkalstn000/dataset/base_funetuning/metafile.json'
with open(json_file, 'r') as f:
    meta = json.load(f)

# Define the path for the JSONL file you want to write to
jsonl_file = '/home/gkalstn000/dataset/inpainting/metafile.jsonl'

filelist = os.listdir(img_root)

train_list = filelist[-10:]
test_list = filelist[:10]

# Open the JSONL file in write mode
with open(jsonl_file, 'w') as f:
    # Iterate through each image file
    for filename in tqdm(train_list):
        if filename in meta:  # Check if the filename exists in the metadata
            # Construct the image and mask paths
            img_path = os.path.join(img_root, filename)
            mask_path = os.path.join(mask_root, filename)

            # Retrieve the caption for this file from the metadata
            caption = meta[filename]['caption']

            # Create a dictionary object for this entry
            json_obj = {
                "file_name": img_path,
                "text": caption,
                "conditioning_image": mask_path
            }

            # Convert the dictionary to a JSON string and write it to the file with a newline
            f.write(json.dumps(json_obj) + '\n')
