from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm
import math

def process_image(filename):
    filepath = os.path.join(img_root, filename)
    image = Image.open(filepath).convert('RGB')

    if image.size == (1024, 1024): return None

    image_resize = image.resize((1024, 1024))


    image_resize.save(filepath)

img_root = '/home/gkalstn000/dataset/inpainting/images'
file_list = os.listdir(img_root)

# 멀티 프로세싱을 사용하여 이미지 처리
with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(process_image, file_list), total=len(file_list)))