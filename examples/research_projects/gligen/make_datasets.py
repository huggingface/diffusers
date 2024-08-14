import argparse
import os
import random

import torch
import torchvision
import torchvision.transforms as TS
from PIL import Image
from ram import inference_ram
from ram.models import ram
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    CLIPTextModel,
    CLIPTokenizer,
)


torch.autograd.set_grad_enabled(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Caption Generation script", add_help=False)
    parser.add_argument("--data_root", type=str, required=True, help="path to COCO")
    parser.add_argument("--save_root", type=str, required=True, help="path to save")
    parser.add_argument("--ram_checkpoint", type=str, required=True, help="path to save")
    args = parser.parse_args()

    # ram_checkpoint = '/root/.cache/huggingface/hub/models--xinyu1205--recognize_anything_model/snapshots/ebc52dc741e86466202a5ab8ab22eae6e7d48bf1/ram_swin_large_14m.pth'
    # data_root = '/mnt/workspace/workgroup/zhizhonghuang/dataset/COCO/train2017'
    # save_root = '/root/gligen_data'
    box_threshold = 0.25
    text_threshold = 0.2

    import torch.distributed as dist

    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    ram_model = ram(pretrained=args.ram_checkpoint, image_size=384, vit="swin_l").cuda().eval()
    ram_processor = TS.Compose(
        [TS.Resize((384, 384)), TS.ToTensor(), TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    grounding_dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).cuda()

    blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xxl", torch_dtype=torch.float16
    ).cuda()

    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    image_paths = [os.path.join(args.data_root, x) for x in os.listdir(args.data_root)]
    random.shuffle(image_paths)

    for image_path in tqdm.tqdm(image_paths):
        pth_path = os.path.join(args.save_root, os.path.basename(image_path))
        if os.path.exists(pth_path):
            continue

        sample = {"file_path": os.path.basename(image_path), "annos": []}

        raw_image = Image.open(image_path).convert("RGB")

        res = inference_ram(ram_processor(raw_image).unsqueeze(0).cuda(), ram_model)

        text = res[0].replace(" |", ".")

        inputs = grounding_dino_processor(images=raw_image, text=text, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = grounding_dino_model(**inputs)

        results = grounding_dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[raw_image.size[::-1]],
        )
        boxes = results[0]["boxes"]
        labels = results[0]["labels"]
        scores = results[0]["scores"]
        indices = torchvision.ops.nms(boxes, scores, 0.5)
        boxes = boxes[indices]
        category_names = [labels[i] for i in indices]

        for i, bbox in enumerate(boxes):
            bbox = bbox.tolist()
            inputs = blip2_processor(images=raw_image.crop(bbox), return_tensors="pt")
            inputs = {k: v.cuda().to(torch.float16) for k, v in inputs.items()}
            outputs = blip2_model.generate(**inputs)
            caption = blip2_processor.decode(outputs[0], skip_special_tokens=True)
            inputs = clip_tokenizer(
                caption,
                padding="max_length",
                max_length=clip_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            text_embeddings_before_projection = clip_text_encoder(**inputs).pooler_output.squeeze(0)

            sample["annos"].append(
                {
                    "caption": caption,
                    "bbox": bbox,
                    "text_embeddings_before_projection": text_embeddings_before_projection,
                }
            )
        torch.save(sample, pth_path)
