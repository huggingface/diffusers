import os
from PIL import Image
import json
from tqdm import tqdm
import shutil


def make_datadir(mode, jsonl_file, filelist):
    with open(jsonl_file, 'w') as f:
        # Iterate through each image file
        for filename in tqdm(filelist):
            path = f'/home/gkalstn000/dataset/canny'
            if filename in meta and filename in canny_simple_list and filename in canny_total_list:  # Check if the filename exists in the metadata
                # Construct the image and mask paths
                img_path = os.path.join(img_root, filename)
                canny_simple_path = os.path.join(path, 'canny_simple', filename)
                canny_total_path = os.path.join(path, 'canny_total', filename)
                # Retrieve the caption for this file from the metadata
                caption = meta[filename]['caption']

                # Create a dictionary object for this entry
                json_obj_simple = {
                    "image": img_path,
                    "text": caption,
                    "canny": canny_simple_path
                }
                json_obj_total = {
                    "image": img_path,
                    "text": caption,
                    "canny": canny_total_path
                }

                # Convert the dictionary to a JSON string and write it to the file with a newline
                f.write(json.dumps(json_obj_simple) + '\n')
                f.write(json.dumps(json_obj_total) + '\n')


# Define the directories
img_root = '/home/gkalstn000/dataset/images'
canny_simple = '/home/gkalstn000/dataset/canny/canny_simple'
canny_total = '/home/gkalstn000/dataset/canny/canny_total'

# Load the metadata
caption_type = 'natural'
json_file = f'/home/gkalstn000/dataset/canny/caption_{caption_type}.json'
with open(json_file, 'r') as f:
    meta = json.load(f)

# Define the path for the JSONL file you want to write to
jsonl_train_file = f'/home/gkalstn000/dataset/canny/train_{caption_type}.jsonl'
jsonl_test_file = f'/home/gkalstn000/dataset/canny/test_{caption_type}.jsonl'

img_list = os.listdir(img_root)
canny_simple_list = os.listdir(canny_simple)
canny_total_list = os.listdir(canny_total)


train_list = img_list[:-10]
test_list = img_list[-10:]

make_datadir(mode=f'train_{caption_type}', jsonl_file=jsonl_train_file, filelist=train_list)
make_datadir(mode=f'test_{caption_type}', jsonl_file=jsonl_test_file, filelist=test_list)

