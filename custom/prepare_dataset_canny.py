import os
from PIL import Image
import json
from tqdm import tqdm
import shutil


def make_datadir(mode, jsonl_file, filelist):
    with open(jsonl_file, 'w') as f:
        # Iterate through each image file
        for filename in tqdm(filelist):
            # Construct the image and mask paths
            img_path = os.path.join(img_root, filename)
            canny_simple_path = os.path.join(canny_simple_root, filename)
            canny_total_path = os.path.join(canny_total_root, filename)
            mask_background_path = os.path.join(mask_background_root, filename)

            # Retrieve the caption for this file from the metadata
            caption = meta_dict[filename]['caption']

            # Create a dictionary object for this entry
            json_obj = {
                "image": img_path,
                "text": caption,
                "canny_simple": canny_simple_path,
                "canny_total": canny_total_path,
                "mask_background": mask_background_path,
            }

            # Convert the dictionary to a JSON string and write it to the file with a newline
            f.write(json.dumps(json_obj) + '\n')


# Define the directories
root = '/home/gkalstn000/dataset/_files'
img_root = os.path.join(root, 'images')
canny_simple_root = os.path.join(root, 'canny_simple')
canny_total_root = os.path.join(root, 'canny_total')
mask_background_root = os.path.join(root, 'masks_background')

# Load the metadata
caption_type = 'format'
json_file = os.path.join(root, f'caption_{caption_type}.json')
with open(json_file, 'r') as f:
    meta = json.load(f)
meta_dict = {}

for filename, caption in zip(meta['images'], meta['captions']):
    meta_dict[filename] = {'caption': caption}


# Define the path for the JSONL file you want to write to
jsonl_train_file = f'/home/gkalstn000/dataset/canny_format/train.jsonl'
jsonl_test_file = f'/home/gkalstn000/dataset/canny_format/test.jsonl'

img_list = os.listdir(img_root)

train_list = img_list[:-10]
test_list = img_list[-10:]

make_datadir(mode=f'train', jsonl_file=jsonl_train_file, filelist=train_list)
make_datadir(mode=f'test', jsonl_file=jsonl_test_file, filelist=test_list)

