#  Copyright 2024 Custom Diffusion authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from io import BytesIO
from pathlib import Path

import requests
from clip_retrieval.clip_client import ClipClient
from PIL import Image
from tqdm import tqdm


def retrieve(class_prompt, class_data_dir, num_class_images):
    factor = 1.5
    num_images = int(factor * num_class_images)
    client = ClipClient(
        url="https://knn.laion.ai/knn-service", indice_name="laion_400m", num_images=num_images, aesthetic_weight=0.1
    )

    os.makedirs(f"{class_data_dir}/images", exist_ok=True)
    if len(list(Path(f"{class_data_dir}/images").iterdir())) >= num_class_images:
        return

    while True:
        class_images = client.query(text=class_prompt)
        if len(class_images) >= factor * num_class_images or num_images > 1e4:
            break
        else:
            num_images = int(factor * num_images)
            client = ClipClient(
                url="https://knn.laion.ai/knn-service",
                indice_name="laion_400m",
                num_images=num_images,
                aesthetic_weight=0.1,
            )

    count = 0
    total = 0
    pbar = tqdm(desc="downloading real regularization images", total=num_class_images)

    with open(f"{class_data_dir}/caption.txt", "w") as f1, open(f"{class_data_dir}/urls.txt", "w") as f2, open(
        f"{class_data_dir}/images.txt", "w"
    ) as f3:
        while total < num_class_images:
            images = class_images[count]
            count += 1
            try:
                img = requests.get(images["url"], timeout=30)
                if img.status_code == 200:
                    _ = Image.open(BytesIO(img.content))
                    with open(f"{class_data_dir}/images/{total}.jpg", "wb") as f:
                        f.write(img.content)
                    f1.write(images["caption"] + "\n")
                    f2.write(images["url"] + "\n")
                    f3.write(f"{class_data_dir}/images/{total}.jpg" + "\n")
                    total += 1
                    pbar.update(1)
                else:
                    continue
            except Exception:
                continue
    return


def parse_args():
    parser = argparse.ArgumentParser("", add_help=False)
    parser.add_argument("--class_prompt", help="text prompt to retrieve images", required=True, type=str)
    parser.add_argument("--class_data_dir", help="path to save images", required=True, type=str)
    parser.add_argument("--num_class_images", help="number of images to download", default=200, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    retrieve(args.class_prompt, args.class_data_dir, args.num_class_images)
