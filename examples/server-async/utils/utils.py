import gc
import logging
import os
import tempfile
import uuid

import torch


logger = logging.getLogger(__name__)


class Utils:
    def __init__(self, host: str = "0.0.0.0", port: int = 8500):
        self.service_url = f"http://{host}:{port}"
        self.image_dir = os.path.join(tempfile.gettempdir(), "images")
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.video_dir = os.path.join(tempfile.gettempdir(), "videos")
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def save_image(self, image):
        if hasattr(image, "to"):
            try:
                image = image.to("cpu")
            except Exception:
                pass

        if isinstance(image, torch.Tensor):
            from torchvision import transforms

            to_pil = transforms.ToPILImage()
            image = to_pil(image.squeeze(0).clamp(0, 1))

        filename = "img" + str(uuid.uuid4()).split("-")[0] + ".png"
        image_path = os.path.join(self.image_dir, filename)
        logger.info(f"Saving image to {image_path}")

        image.save(image_path, format="PNG", optimize=True)

        del image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return os.path.join(self.service_url, "images", filename)
