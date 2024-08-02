"""
Copyright (c) Alibaba, Inc. and its affiliates.
"""
import math
import os
import sys
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from ocr_recog.RecModel import RecModel
from skimage.transform._geometric import _umeyama as get_sym_mat


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def min_bounding_rect(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("Bad contours, using fake bbox...")
        return np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # sort
    x_sorted = sorted(box, key=lambda x: x[0])
    left = x_sorted[:2]
    right = x_sorted[2:]
    left = sorted(left, key=lambda x: x[1])
    (tl, bl) = left
    right = sorted(right, key=lambda x: x[1])
    (tr, br) = right
    if tl[1] > bl[1]:
        (tl, bl) = (bl, tl)
    if tr[1] > br[1]:
        (tr, br) = (br, tr)
    return np.array([tl, tr, br, bl])


def adjust_image(box, img):
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    width = max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3]))
    height = max(np.linalg.norm(pts1[0] - pts1[3]), np.linalg.norm(pts1[1] - pts1[2]))
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # get transform matrix
    M = get_sym_mat(pts1, pts2, estimate_scale=True)
    C, H, W = img.shape
    T = np.array([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]])
    theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
    theta = torch.from_numpy(theta[:2, :]).unsqueeze(0).type(torch.float32).to(img.device)
    grid = F.affine_grid(theta, torch.Size([1, C, H, W]), align_corners=True)
    result = F.grid_sample(img.unsqueeze(0), grid, align_corners=True)
    result = torch.clamp(result.squeeze(0), 0, 255)
    # crop
    result = result[:, : int(height), : int(width)]
    return result


"""
mask: numpy.ndarray, mask of textual, HWC
src_img: torch.Tensor, source image, CHW
"""


def crop_image(src_img, mask):
    box = min_bounding_rect(mask)
    result = adjust_image(box, src_img)
    if len(result.shape) == 2:
        result = torch.stack([result] * 3, axis=-1)
    return result


def create_predictor(model_dir=None, model_lang="ch", is_onnx=False):
    model_file_path = model_dir
    if model_file_path is not None and not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(model_file_path))

    if is_onnx:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            model_file_path, providers=["CPUExecutionProvider"]
        )  # 'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'
        return sess
    else:
        if model_lang == "ch":
            n_class = 6625
        elif model_lang == "en":
            n_class = 97
        else:
            raise ValueError(f"Unsupported OCR recog model_lang: {model_lang}")
        rec_config = edict(
            in_channels=3,
            backbone=edict(type="MobileNetV1Enhance", scale=0.5, last_conv_stride=[1, 2], last_pool_type="avg"),
            neck=edict(type="SequenceEncoder", encoder_type="svtr", dims=64, depth=2, hidden_dims=120, use_guide=True),
            head=edict(type="CTCHead", fc_decay=0.00001, out_channels=n_class, return_feats=True),
        )

        rec_model = RecModel(rec_config)
        if model_file_path is not None:
            rec_model.load_state_dict(torch.load(model_file_path, map_location="cpu"))
            rec_model.eval()
        return rec_model.eval()


def _check_image_file(path):
    img_end = ("tiff", "tif", "bmp", "rgb", "jpg", "png", "jpeg")
    return path.lower().endswith(tuple(img_end))


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))
    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


class TextRecognizer(object):
    def __init__(self, args, predictor):
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.predictor = predictor
        self.chars = self.get_char_dict(args.rec_char_dict_path)
        self.char2id = {x: i for i, x in enumerate(self.chars)}
        self.is_onnx = not isinstance(self.predictor, torch.nn.Module)
        self.use_fp16 = args.use_fp16

    # img: CHW
    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[0]
        imgW = int((imgH * max_wh_ratio))

        h, w = img.shape[1:]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(imgH, resized_w),
            mode="bilinear",
            align_corners=True,
        )
        resized_image /= 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = torch.zeros((imgC, imgH, imgW), dtype=torch.float32).to(img.device)
        padding_im[:, :, 0:resized_w] = resized_image[0]
        return padding_im

    # img_list: list of tensors with shape chw 0-255
    def pred_imglist(self, img_list, show_debug=False):
        img_num = len(img_list)
        assert img_num > 0
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[2] / float(img.shape[1]))
        # Sorting can speed up the recognition process
        indices = torch.from_numpy(np.argsort(np.array(width_list)))
        batch_num = self.rec_batch_num
        preds_all = [None] * img_num
        preds_neck_all = [None] * img_num
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []

            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[1:]
                if h > w * 1.2:
                    img = img_list[indices[ino]]
                    img = torch.transpose(img, 1, 2).flip(dims=[1])
                    img_list[indices[ino]] = img
                    h, w = img.shape[1:]
                # wh_ratio = w * 1.0 / h
                # max_wh_ratio = max(max_wh_ratio, wh_ratio)  # comment to not use different ratio
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                if self.use_fp16:
                    norm_img = norm_img.half()
                norm_img = norm_img.unsqueeze(0)
                norm_img_batch.append(norm_img)
            norm_img_batch = torch.cat(norm_img_batch, dim=0)
            if show_debug:
                for i in range(len(norm_img_batch)):
                    _img = norm_img_batch[i].permute(1, 2, 0).detach().cpu().numpy()
                    _img = (_img + 0.5) * 255
                    _img = _img[:, :, ::-1]
                    file_name = f"{indices[beg_img_no + i]}"
                    if os.path.exists(file_name + ".jpg"):
                        file_name += "_2"  # ori image
                    cv2.imwrite(file_name + ".jpg", _img)
            if self.is_onnx:
                input_dict = {}
                input_dict[self.predictor.get_inputs()[0].name] = norm_img_batch.detach().cpu().numpy()
                outputs = self.predictor.run(None, input_dict)
                preds = {}
                preds["ctc"] = torch.from_numpy(outputs[0])
                preds["ctc_neck"] = [torch.zeros(1)] * img_num
            else:
                preds = self.predictor(norm_img_batch)
            for rno in range(preds["ctc"].shape[0]):
                preds_all[indices[beg_img_no + rno]] = preds["ctc"][rno]
                preds_neck_all[indices[beg_img_no + rno]] = preds["ctc_neck"][rno]

        return torch.stack(preds_all, dim=0), torch.stack(preds_neck_all, dim=0)

    def get_char_dict(self, character_dict_path):
        character_str = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                character_str.append(line)
        dict_character = list(character_str)
        dict_character = ["sos"] + dict_character + [" "]  # eos is space
        return dict_character

    def get_text(self, order):
        char_list = [self.chars[text_id] for text_id in order]
        return "".join(char_list)

    def decode(self, mat):
        text_index = mat.detach().cpu().numpy().argmax(axis=1)
        ignored_tokens = [0]
        selection = np.ones(len(text_index), dtype=bool)
        selection[1:] = text_index[1:] != text_index[:-1]
        for ignored_token in ignored_tokens:
            selection &= text_index != ignored_token
        return text_index[selection], np.where(selection)[0]

    def get_ctcloss(self, preds, gt_text, weight):
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight).to(preds.device)
        ctc_loss = torch.nn.CTCLoss(reduction="none")
        log_probs = preds.log_softmax(dim=2).permute(1, 0, 2)  # NTC-->TNC
        targets = []
        target_lengths = []
        for t in gt_text:
            targets += [self.char2id.get(i, len(self.chars) - 1) for i in t]
            target_lengths += [len(t)]
        targets = torch.tensor(targets).to(preds.device)
        target_lengths = torch.tensor(target_lengths).to(preds.device)
        input_lengths = torch.tensor([log_probs.shape[0]] * (log_probs.shape[1])).to(preds.device)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss = loss / input_lengths * weight
        return loss


def main():
    rec_model_dir = "./ocr_weights/ppv3_rec.pth"
    predictor = create_predictor(rec_model_dir)
    args = edict()
    args.rec_image_shape = "3, 48, 320"
    args.rec_char_dict_path = "./ocr_weights/ppocr_keys_v1.txt"
    args.rec_batch_num = 6
    text_recognizer = TextRecognizer(args, predictor)
    image_dir = "./test_imgs_cn"
    gt_text = ["韩国小馆"] * 14

    image_file_list = get_image_file_list(image_dir)
    valid_image_file_list = []
    img_list = []

    for image_file in image_file_list:
        img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(torch.from_numpy(img).permute(2, 0, 1).float())
    try:
        tic = time.time()
        times = []
        for i in range(10):
            preds, _ = text_recognizer.pred_imglist(img_list)  # get text
            preds_all = preds.softmax(dim=2)
            times += [(time.time() - tic) * 1000.0]
            tic = time.time()
        print(times)
        print(np.mean(times[1:]) / len(preds_all))
        weight = np.ones(len(gt_text))
        loss = text_recognizer.get_ctcloss(preds, gt_text, weight)
        for i in range(len(valid_image_file_list)):
            pred = preds_all[i]
            order, idx = text_recognizer.decode(pred)
            text = text_recognizer.get_text(order)
            print(f'{valid_image_file_list[i]}: pred/gt="{text}"/"{gt_text[i]}", loss={loss[i]:.2f}')
    except Exception as E:
        print(traceback.format_exc(), E)


if __name__ == "__main__":
    main()
