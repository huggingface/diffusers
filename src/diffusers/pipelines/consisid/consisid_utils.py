import importlib.util
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from ...utils import get_logger, load_image


logger = get_logger(__name__)

_insightface_available = importlib.util.find_spec("insightface") is not None
_consisid_eva_clip_available = importlib.util.find_spec("consisid_eva_clip") is not None
_facexlib_available = importlib.util.find_spec("facexlib") is not None

if _insightface_available:
    import insightface
    from insightface.app import FaceAnalysis
else:
    raise ImportError("insightface is not available. Please install it using 'pip install insightface'.")

if _consisid_eva_clip_available:
    from consisid_eva_clip import create_model_and_transforms
    from consisid_eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
else:
    raise ImportError("consisid_eva_clip is not available. Please install it using 'pip install consisid_eva_clip'.")

if _facexlib_available:
    from facexlib.parsing import init_parsing_model
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
else:
    raise ImportError("facexlib is not available. Please install it using 'pip install facexlib'.")


def resize_numpy_image_long(image, resize_long_edge=768):
    """
    Resize the input image to a specified long edge while maintaining aspect ratio.

    Args:
        image (numpy.ndarray): Input image (H x W x C or H x W).
        resize_long_edge (int): The target size for the long edge of the image. Default is 768.

    Returns:
        numpy.ndarray: Resized image with the long edge matching `resize_long_edge`, while maintaining the aspect
        ratio.
    """

    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32)


def to_gray(img):
    """
    Converts an RGB image to grayscale by applying the standard luminosity formula.

    Args:
        img (torch.Tensor): The input image tensor with shape (batch_size, channels, height, width).
                             The image is expected to be in RGB format (3 channels).

    Returns:
        torch.Tensor: The grayscale image tensor with shape (batch_size, 3, height, width).
                      The grayscale values are replicated across all three channels.
    """
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x


def process_face_embeddings(
    face_helper_1,
    clip_vision_model,
    face_helper_2,
    eva_transform_mean,
    eva_transform_std,
    app,
    device,
    weight_dtype,
    image,
    original_id_image=None,
    is_align_face=True,
):
    """
    Process face embeddings from an image, extracting relevant features such as face embeddings, landmarks, and parsed
    face features using a series of face detection and alignment tools.

    Args:
        face_helper_1: Face helper object (first helper) for alignment and landmark detection.
        clip_vision_model: Pre-trained CLIP vision model used for feature extraction.
        face_helper_2: Face helper object (second helper) for embedding extraction.
        eva_transform_mean: Mean values for image normalization before passing to EVA model.
        eva_transform_std: Standard deviation values for image normalization before passing to EVA model.
        app: Application instance used for face detection.
        device: Device (CPU or GPU) where the computations will be performed.
        weight_dtype: Data type of the weights for precision (e.g., `torch.float32`).
        image: Input image in RGB format with pixel values in the range [0, 255].
        original_id_image: (Optional) Original image for feature extraction if `is_align_face` is False.
        is_align_face: Boolean flag indicating whether face alignment should be performed.

    Returns:
        Tuple:
            - id_cond: Concatenated tensor of Ante face embedding and CLIP vision embedding
            - id_vit_hidden: Hidden state of the CLIP vision model, a list of tensors.
            - return_face_features_image_2: Processed face features image after normalization and parsing.
            - face_kps: Keypoints of the face detected in the image.
    """

    face_helper_1.clean_all()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # get antelopev2 embedding
    face_info = app.get(image_bgr)
    if len(face_info) > 0:
        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[
            -1
        ]  # only use the maximum face
        id_ante_embedding = face_info["embedding"]  # (512,)
        face_kps = face_info["kps"]
    else:
        id_ante_embedding = None
        face_kps = None

    # using facexlib to detect and align face
    face_helper_1.read_image(image_bgr)
    face_helper_1.get_face_landmarks_5(only_center_face=True)
    if face_kps is None:
        face_kps = face_helper_1.all_landmarks_5[0]
    face_helper_1.align_warp_face()
    if len(face_helper_1.cropped_faces) == 0:
        raise RuntimeError("facexlib align face fail")
    align_face = face_helper_1.cropped_faces[0]  # (512, 512, 3)  # RGB

    # in case insightface didn't detect face
    if id_ante_embedding is None:
        logger.warning("Failed to detect face using insightface. Extracting embedding with align face")
        id_ante_embedding = face_helper_2.get_feat(align_face)

    id_ante_embedding = torch.from_numpy(id_ante_embedding).to(device, weight_dtype)  # torch.Size([512])
    if id_ante_embedding.ndim == 1:
        id_ante_embedding = id_ante_embedding.unsqueeze(0)  # torch.Size([1, 512])

    # parsing
    if is_align_face:
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0  # torch.Size([1, 3, 512, 512])
        input = input.to(device)
        parsing_out = face_helper_1.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)  # torch.Size([1, 1, 512, 512])
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)  # torch.Size([1, 3, 512, 512])
        # only keep the face features
        return_face_features_image = torch.where(bg, white_image, to_gray(input))  # torch.Size([1, 3, 512, 512])
        return_face_features_image_2 = torch.where(bg, white_image, input)  # torch.Size([1, 3, 512, 512])
    else:
        original_image_bgr = cv2.cvtColor(original_id_image, cv2.COLOR_RGB2BGR)
        input = img2tensor(original_image_bgr, bgr2rgb=True).unsqueeze(0) / 255.0  # torch.Size([1, 3, 512, 512])
        input = input.to(device)
        return_face_features_image = return_face_features_image_2 = input

    # transform img before sending to eva-clip-vit
    face_features_image = resize(
        return_face_features_image, clip_vision_model.image_size, InterpolationMode.BICUBIC
    )  # torch.Size([1, 3, 336, 336])
    face_features_image = normalize(face_features_image, eva_transform_mean, eva_transform_std)
    id_cond_vit, id_vit_hidden = clip_vision_model(
        face_features_image.to(weight_dtype), return_all_features=False, return_hidden=True, shuffle=False
    )  # torch.Size([1, 768]),  list(torch.Size([1, 577, 1024]))
    id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
    id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

    id_cond = torch.cat(
        [id_ante_embedding, id_cond_vit], dim=-1
    )  # torch.Size([1, 512]), torch.Size([1, 768])  ->  torch.Size([1, 1280])

    return (
        id_cond,
        id_vit_hidden,
        return_face_features_image_2,
        face_kps,
    )  # torch.Size([1, 1280]), list(torch.Size([1, 577, 1024]))


def process_face_embeddings_infer(
    face_helper_1,
    clip_vision_model,
    face_helper_2,
    eva_transform_mean,
    eva_transform_std,
    app,
    device,
    weight_dtype,
    img_file_path,
    is_align_face=True,
):
    """
    Process face embeddings from an input image for inference, including alignment, feature extraction, and embedding
    concatenation.

    Args:
        face_helper_1: Face helper object (first helper) for alignment and landmark detection.
        clip_vision_model: Pre-trained CLIP vision model used for feature extraction.
        face_helper_2: Face helper object (second helper) for embedding extraction.
        eva_transform_mean: Mean values for image normalization before passing to EVA model.
        eva_transform_std: Standard deviation values for image normalization before passing to EVA model.
        app: Application instance used for face detection.
        device: Device (CPU or GPU) where the computations will be performed.
        weight_dtype: Data type of the weights for precision (e.g., `torch.float32`).
        img_file_path: Path to the input image file (string) or a numpy array representing an image.
        is_align_face: Boolean flag indicating whether face alignment should be performed (default: True).

    Returns:
        Tuple:
            - id_cond: Concatenated tensor of Ante face embedding and CLIP vision embedding.
            - id_vit_hidden: Hidden state of the CLIP vision model, a list of tensors.
            - image: Processed face image after feature extraction and alignment.
            - face_kps: Keypoints of the face detected in the image.
    """

    # Load and preprocess the input image
    if isinstance(img_file_path, str):
        image = np.array(load_image(image=img_file_path).convert("RGB"))
    else:
        image = np.array(ImageOps.exif_transpose(Image.fromarray(img_file_path)).convert("RGB"))

    # Resize image to ensure the longer side is 1024 pixels
    image = resize_numpy_image_long(image, 1024)
    original_id_image = image

    # Process the image to extract face embeddings and related features
    id_cond, id_vit_hidden, align_crop_face_image, face_kps = process_face_embeddings(
        face_helper_1,
        clip_vision_model,
        face_helper_2,
        eva_transform_mean,
        eva_transform_std,
        app,
        device,
        weight_dtype,
        image,
        original_id_image,
        is_align_face,
    )

    # Convert the aligned cropped face image (torch tensor) to a numpy array
    tensor = align_crop_face_image.cpu().detach()
    tensor = tensor.squeeze()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy() * 255
    tensor = tensor.astype(np.uint8)
    image = ImageOps.exif_transpose(Image.fromarray(tensor))

    return id_cond, id_vit_hidden, image, face_kps


def prepare_face_models(model_path, device, dtype):
    """
    Prepare all face models for the facial recognition task.

    Parameters:
    - model_path: Path to the directory containing model files.
    - device: The device (e.g., 'cuda', 'xpu', 'cpu') where models will be loaded.
    - dtype: Data type (e.g., torch.float32) for model inference.

    Returns:
    - face_helper_1: First face restoration helper.
    - face_helper_2: Second face restoration helper.
    - face_clip_model: CLIP model for face extraction.
    - eva_transform_mean: Mean value for image normalization.
    - eva_transform_std: Standard deviation value for image normalization.
    - face_main_model: Main face analysis model.
    """
    # get helper model
    face_helper_1 = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        device=device,
        model_rootpath=os.path.join(model_path, "face_encoder"),
    )
    face_helper_1.face_parse = None
    face_helper_1.face_parse = init_parsing_model(
        model_name="bisenet", device=device, model_rootpath=os.path.join(model_path, "face_encoder")
    )
    face_helper_2 = insightface.model_zoo.get_model(
        f"{model_path}/face_encoder/models/antelopev2/glintr100.onnx", providers=["CUDAExecutionProvider"]
    )
    face_helper_2.prepare(ctx_id=0)

    # get local facial extractor part 1
    model, _, _ = create_model_and_transforms(
        "EVA02-CLIP-L-14-336",
        os.path.join(model_path, "face_encoder", "EVA02_CLIP_L_336_psz14_s6B.pt"),
        force_custom_clip=True,
    )
    face_clip_model = model.visual
    eva_transform_mean = getattr(face_clip_model, "image_mean", OPENAI_DATASET_MEAN)
    eva_transform_std = getattr(face_clip_model, "image_std", OPENAI_DATASET_STD)
    if not isinstance(eva_transform_mean, (list, tuple)):
        eva_transform_mean = (eva_transform_mean,) * 3
    if not isinstance(eva_transform_std, (list, tuple)):
        eva_transform_std = (eva_transform_std,) * 3
    eva_transform_mean = eva_transform_mean
    eva_transform_std = eva_transform_std

    # get local facial extractor part 2
    face_main_model = FaceAnalysis(
        name="antelopev2", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )
    face_main_model.prepare(ctx_id=0, det_size=(640, 640))

    # move face models to device
    face_helper_1.face_det.eval()
    face_helper_1.face_parse.eval()
    face_clip_model.eval()
    face_helper_1.face_det.to(device)
    face_helper_1.face_parse.to(device)
    face_clip_model.to(device, dtype=dtype)

    return face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std
