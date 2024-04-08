import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm


def delete_unmatched_files(paths, file_lists):
    """Unmatched 파일 삭제"""
    for path, file_list in zip(paths, file_lists):
        for filename in file_list:
            os.remove(os.path.join(path, filename))


def save_background_mask(mask_path, save_path, filename):
    """배경 마스크 저장"""
    image = Image.open(os.path.join(mask_path, filename)).convert('L')
    background_mask = Image.fromarray(255 - np.array(image))
    background_mask.save(os.path.join(save_path, filename))
    return background_mask


def save_canny_image(image_path, save_path, filename, mask=None):
    """Canny 이미지 저장"""
    image = Image.open(os.path.join(image_path, filename))
    image_array = np.array(image)
    if mask is not None:
        image_array = np.where(mask > 128, 0, image_array)
    canny_image = cv2.Canny(image_array, 100, 200)
    Image.fromarray(canny_image).save(os.path.join(save_path, filename))


# 경로 설정
root = '/home/gkalstn000/dataset/inpainting'
image_path = os.path.join(root, 'images')
mask_path = os.path.join(root, 'mask_foreground_images')

# 파일 목록 생성
image_list = os.listdir(image_path)
mask_list = os.listdir(mask_path)
filelist = list(set(image_list) & set(mask_list))

# 삭제할 파일 목록
images_to_delete = list(set(image_list) - set(filelist))
masks_to_delete = list(set(mask_list) - set(filelist))

# 파일 삭제
delete_unmatched_files([image_path, mask_path], [images_to_delete, masks_to_delete])

# 디렉토리 생성
paths = ['mask_background_images', 'canny_total', 'canny_simple']
for path in paths:
    os.makedirs(os.path.join(root, path), exist_ok=True)

for filename in tqdm(filelist, desc='Processing'):
    # 배경 마스크 저장
    background_mask = save_background_mask(mask_path, os.path.join(root, 'mask_background_images'), filename)

    # canny_total 저장
    save_canny_image(image_path, os.path.join(root, 'canny_total'), filename)

    # canny_simple 저장
    mask_array = np.array(background_mask)
    mask_array = np.expand_dims(mask_array, axis=2)
    mask_array = np.repeat(mask_array, 3, axis=2)
    save_canny_image(image_path, os.path.join(root, 'canny_simple'), filename, mask=mask_array)
