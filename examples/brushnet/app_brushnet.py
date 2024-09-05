##!/usr/bin/python3
# -*- coding: utf-8 -*-
import gradio as gr
import os
import cv2
from PIL import Image
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import torch
from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import random

mobile_sam = sam_model_registry['vit_h'](checkpoint='data/ckpt/sam_vit_h_4b8939.pth').to("cuda")
mobile_sam.eval()
mobile_predictor = SamPredictor(mobile_sam)
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

# - - - - - examples  - - - - -  #
image_examples = [
    ["examples/brushnet/src/test_image.jpg", "A beautiful cake on the table", "examples/brushnet/src/test_mask.jpg", 0, [], [Image.open("examples/brushnet/src/test_result.png")]],
    ["examples/brushnet/src/example_1.jpg", "A man in Chinese traditional clothes", "examples/brushnet/src/example_1_mask.jpg", 1, [], [Image.open("examples/brushnet/src/example_1_result.png")]],
    ["examples/brushnet/src/example_2.jpg", "a charming woman with dress standing by the sea", "examples/brushnet/src/example_2_mask.jpg", 2, [], [Image.open("examples/brushnet/src/example_2_result.png")]],
    ["examples/brushnet/src/example_3.jpg", "a cut toy on the table", "examples/brushnet/src/example_3_mask.jpg", 3, [], [Image.open("examples/brushnet/src/example_3_result.png")]],
    ["examples/brushnet/src/example_4.jpeg", "a car driving in the wild", "examples/brushnet/src/example_4_mask.jpg", 4, [], [Image.open("examples/brushnet/src/example_4_result.png")]],
    ["examples/brushnet/src/example_5.jpg", "a charming woman wearing dress standing in the dark forest", "examples/brushnet/src/example_5_mask.jpg", 5, [], [Image.open("examples/brushnet/src/example_5_result.png")]],
]


# choose the base model here
base_model_path = "data/ckpt/realisticVisionV60B1_v51VAE"
# base_model_path = "runwayml/stable-diffusion-v1-5"

# input brushnet ckpt path
brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt"

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def process(input_image, 
    original_image, 
    original_mask, 
    input_mask, 
    selected_points, 
    prompt, 
    negative_prompt, 
    blended, 
    invert_mask, 
    control_strength, 
    seed, 
    randomize_seed, 
    guidance_scale, 
    num_inference_steps):
    if original_image is None:
        raise gr.Error('Please upload the input image')
    if (original_mask is None or len(selected_points)==0) and input_mask is None:
        raise gr.Error("Please click the region where you hope unchanged/changed, or upload a white-black Mask image")
    
    # load example image
    if isinstance(original_image, int):
        image_name = image_examples[original_image][0]
        original_image = cv2.imread(image_name)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    if input_mask is not None:
        H,W=original_image.shape[:2]
        original_mask = cv2.resize(input_mask, (W, H))
    else:
        original_mask = np.clip(255 - original_mask, 0, 255).astype(np.uint8)

    if invert_mask:
        original_mask=255-original_mask

    mask = 1.*(original_mask.sum(-1)>255)[:,:,np.newaxis]
    masked_image = original_image * (1-mask)

    init_image = Image.fromarray(masked_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray(original_mask.astype(np.uint8)).convert("RGB")

    generator = torch.Generator("cuda").manual_seed(random.randint(0,2147483647) if randomize_seed else seed)

    image = pipe(
        [prompt]*2, 
        init_image, 
        mask_image, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        generator=generator,
        brushnet_conditioning_scale=float(control_strength),
        negative_prompt=[negative_prompt]*2,
    ).images

    if blended:
        if control_strength<1.0:
            raise gr.Error('Using blurred blending with control strength less than 1.0 is not allowed')
        blended_image=[]
        # blur, you can adjust the parameters for better performance
        mask_blurred = cv2.GaussianBlur(mask*255, (21, 21), 0)/255
        mask_blurred = mask_blurred[:,:,np.newaxis]
        mask = 1-(1-mask) * (1-mask_blurred)
        for image_i in image:
            image_np=np.array(image_i)
            image_pasted=original_image * (1-mask) + image_np*mask

            image_pasted=image_pasted.astype(image_np.dtype)
            blended_image.append(Image.fromarray(image_pasted))
        
        image=blended_image

    return image

block = gr.Blocks(
        theme=gr.themes.Soft(
             radius_size=gr.themes.sizes.radius_none,
             text_size=gr.themes.sizes.text_md
         )
        ).queue()
with block:
    with gr.Row():
        with gr.Column():
            
            gr.HTML(f"""
                    <div style="text-align: center;">
                        <h1>BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion</h1>
                        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                            <a href=""></a>
                            <a href='https://tencentarc.github.io/BrushNet/'><img src='https://img.shields.io/badge/Project_Page-BrushNet-green' alt='Project Page'></a>
                            <a href='https://arxiv.org/abs/2403.06976'><img src='https://img.shields.io/badge/Paper-Arxiv-blue'></a>
                        </div>
                        </br>
                    </div>
            """)


    with gr.Accordion(label="🧭 Instructions:", open=True, elem_id="accordion"):
        with gr.Row(equal_height=True):
            gr.Markdown("""
            - ⭐️ <b>step1: </b>Upload or select one image from Example
            - ⭐️ <b>step2: </b>Click on Input-image to select the object to be retained (or upload a white-black Mask image, in which white color indicates the region you want to keep unchanged). You can tick the 'Invert Mask' box to switch region unchanged and change.
            - ⭐️ <b>step3: </b>Input prompt for generating new contents
            - ⭐️ <b>step4: </b>Click Run button
            """)                          
    with gr.Row():
        with gr.Column():
            with gr.Column(elem_id="Input"):
                with gr.Row():
                    with gr.Tabs(elem_classes=["feedback"]):
                        with gr.TabItem("Input Image"):
                            input_image = gr.Image(type="numpy", label="input",scale=2, height=640)
                original_image = gr.State(value=None,label="index")
                original_mask = gr.State(value=None)
                selected_points = gr.State([],label="select points")
                with gr.Row(elem_id="Seg"):
                    radio = gr.Radio(['foreground', 'background'], label='Click to seg: ', value='foreground',scale=2)
                    undo_button = gr.Button('Undo seg', elem_id="btnSEG",scale=1)
            prompt = gr.Textbox(label="Prompt", placeholder="Please input your prompt",value='',lines=1)
            negative_prompt = gr.Text(
                        label="Negative Prompt",
                        max_lines=5,
                        placeholder="Please input your negative prompt",
                        value='ugly, low quality',lines=1
                    )
            with gr.Group():
                with gr.Row():
                    blending = gr.Checkbox(label="Blurred Blending", value=False)
                    invert_mask = gr.Checkbox(label="Invert Mask", value=True)
            run_button = gr.Button("Run",elem_id="btn")
            
            with gr.Accordion("More input params (highly-recommended)", open=False, elem_id="accordion1"):
                control_strength = gr.Slider(
                    label="Control Strength: ", show_label=True, minimum=0, maximum=1.1, value=1, step=0.01
                    )
                with gr.Group():
                    seed = gr.Slider(
                        label="Seed: ", minimum=0, maximum=2147483647, step=1, value=551793204
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                
                with gr.Group():
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            label="Guidance scale",
                            minimum=1,
                            maximum=12,
                            step=0.1,
                            value=12,
                        )
                        num_inference_steps = gr.Slider(
                            label="Number of inference steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=50,
                        )
                with gr.Row(elem_id="Image"):
                    with gr.Tabs(elem_classes=["feedback1"]):
                        with gr.TabItem("User-specified Mask Image (Optional)"):
                            input_mask = gr.Image(type="numpy", label="Mask Image", height=640)
            
        with gr.Column():
            with gr.Tabs(elem_classes=["feedback"]):
                with gr.TabItem("Outputs"):
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True)
    with gr.Row():
        def process_example(input_image, prompt, input_mask, original_image, selected_points,result_gallery): #
            return input_image, prompt, input_mask, original_image, [], result_gallery
        example = gr.Examples(
            label="Input Example",
            examples=image_examples,
            inputs=[input_image, prompt, input_mask, original_image, selected_points,result_gallery],
            outputs=[input_image, prompt, input_mask, original_image, selected_points],
            fn=process_example,
            run_on_click=True,
            examples_per_page=10
        )

    # once user upload an image, the original image is stored in `original_image`
    def store_img(img):
        # image upload is too slow
        if min(img.shape[0], img.shape[1]) > 512:
            img = resize_image(img, 512)
        if max(img.shape[0], img.shape[1])*1.0/min(img.shape[0], img.shape[1])>2.0:
            raise gr.Error('image aspect ratio cannot be larger than 2.0')
        return img, img, [], None  # when new image is uploaded, `selected_points` should be empty

    input_image.upload(
        store_img,
        [input_image],
        [input_image, original_image, selected_points]
    )

    # user click the image to get points, and show the points on the image
    def segmentation(img, sel_pix):
        # online show seg mask
        points = []
        labels = []
        for p, l in sel_pix:
            points.append(p)
            labels.append(l)
        mobile_predictor.set_image(img if isinstance(img, np.ndarray) else np.array(img))
        with torch.no_grad():
            masks, _, _ = mobile_predictor.predict(point_coords=np.array(points), point_labels=np.array(labels), multimask_output=False)

        output_mask = np.ones((masks.shape[1], masks.shape[2], 3))*255
        for i in range(3):
                output_mask[masks[0] == True, i] = 0.0

        mask_all = np.ones((masks.shape[1], masks.shape[2], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            mask_all[masks[0] == True, i] = color_mask[i]
        masked_img = img / 255 * 0.3 + mask_all * 0.7
        masked_img = masked_img*255
        ## draw points
        for point, label in sel_pix:
            cv2.drawMarker(masked_img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        return masked_img, output_mask
    
    def get_point(img, sel_pix, point_type, evt: gr.SelectData):
        if point_type == 'foreground':
            sel_pix.append((evt.index, 1))   # append the foreground_point
        elif point_type == 'background':
            sel_pix.append((evt.index, 0))    # append the background_point
        else:
            sel_pix.append((evt.index, 1))    # default foreground_point

        if isinstance(img, int):
            image_name = image_examples[img][0]
            img = cv2.imread(image_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # online show seg mask
        masked_img, output_mask = segmentation(img, sel_pix)
        return masked_img.astype(np.uint8), output_mask
    
    input_image.select(
        get_point,
        [original_image, selected_points, radio],
        [input_image, original_mask],
    )

    # undo the selected point
    def undo_points(orig_img, sel_pix):
        # draw points
        output_mask = None
        if len(sel_pix) != 0:
            if isinstance(orig_img, int):   # if orig_img is int, the image if select from examples
                temp = cv2.imread(image_examples[orig_img][0])
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            else:
                temp = orig_img.copy()
            sel_pix.pop()
            # online show seg mask
            if len(sel_pix) !=0:
                temp, output_mask = segmentation(temp, sel_pix)
            return temp.astype(np.uint8), output_mask
        else:
            gr.Error("Nothing to Undo")
    
    undo_button.click(
        undo_points,
        [original_image, selected_points],
        [input_image, original_mask]
    )

    ips=[input_image, original_image, original_mask, input_mask, selected_points, prompt, negative_prompt, blending, invert_mask, control_strength, seed, randomize_seed, guidance_scale, num_inference_steps]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name="0.0.0.0",share=False,server_port=12345)
