import os
from PIL import Image
import json
from tqdm import tqdm
import shutil


def make_datadir(mode, jsonl_file, filelist):
    with open(jsonl_file, 'w') as f:
        # Iterate through each image file
        for filename in tqdm(filelist):
            path = f'/home/gkalstn000/dataset/inpainting/{mode}'
            if filename in meta and filename in masklist:  # Check if the filename exists in the metadata
                # Construct the image and mask paths
                img_path = os.path.join(path, 'images', filename)
                shutil.copy(os.path.join(img_root, filename), img_path)
                mask_path = os.path.join(path, 'conditioning_images', filename)
                shutil.copy(os.path.join(mask_root, filename), mask_path)
                # Retrieve the caption for this file from the metadata
                caption = meta[filename]['caption']

                # Create a dictionary object for this entry
                json_obj = {
                    "image": img_path,
                    "text": caption,
                    "conditioning_image": mask_path
                }

                # Convert the dictionary to a JSON string and write it to the file with a newline
                f.write(json.dumps(json_obj) + '\n')


# Define the directories
img_root = '/home/gkalstn000/dataset/inpainting/images'
mask_root = '/home/gkalstn000/dataset/inpainting/conditioning_images'

# Load the metadata
json_file = '/home/gkalstn000/dataset/base_funetuning/metafile.json'
with open(json_file, 'r') as f:
    meta = json.load(f)

# Define the path for the JSONL file you want to write to
jsonl_train_file = '/home/gkalstn000/dataset/inpainting/train.jsonl'
jsonl_test_file = '/home/gkalstn000/dataset/inpainting/test.jsonl'


filelist = os.listdir(img_root)
masklist = os.listdir(mask_root)

train_list = filelist[:-10]
test_list = filelist[-10:]

make_datadir(mode='train', jsonl_file=jsonl_train_file, filelist=train_list)
make_datadir(mode='test', jsonl_file=jsonl_test_file, filelist=test_list)

