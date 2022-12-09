# Multiresolution Textual Inversion

[![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1UFhEKhW0RCLvUY_ZouqrF3wpubLs0Woh?usp=sharing)
![](https://img.shields.io/badge/pytorch-green)

![](https://github.com/giannisdaras/multires_textual_inversion/blob/main/images/example_dog.png?raw=true)


Multiresolution Textual Inversion extends Textual Inversion to learn pseudo-words that represent a concept at different resolutions. This allows us to generate images that use the concept with different levels of detail and also to manipulate different resolutions using language.
Once learned, the user can generate images at different levels of agreement to the original concept: "*A photo of <S(0)>*" produces the exact object while the prompt: "*A photo of <S(0.8)>*" only matches the rough outlines and colors. 
Our framework allows us to generate images that use different resolutions of an image (e.g. details, textures, styles)  as separate pseudo-words that can be composed in various  ways. 


## Get started

The fastest way to get started is to use the [Colab](https://colab.research.google.com/drive/1UFhEKhW0RCLvUY_ZouqrF3wpubLs0Woh?usp=sharing). Below, we show to get started locally and we show some of the results.

### Project Installation

To get started:

1. Run: `pip install -r requirements.txt` to install the python dependencies.
2. Get Huggingface token:
    `huggingface-cli login`

### Generate samples with Textual Inversion

We provide a couple of models trained with Textual Inversion. If you want to see how to train your own models, skip to the next section. To download the pre-trained models, run the following commands:

```
gdown --id 1HhksfGmQh6xAiS2MIi6kMe430iHl2PGj && unzip textual_inversion_outputs && mv outputs textual_inversion_outputs
gdown --id 1u8bBM85ncM2D6lusMPG-PIBSN6fktCec && unzip jane.zip && mv jane/ textual_inversion_outputs
```


Once you have downloaded (or trained) the models for the learned concepts, you can use language to generate images of your concept at different resolutions. First, initialize the Multiresolution image generation pipeline and load the trained concepts, as shown below:

```
from diffusers import MultiResPipeline, load_learned_concepts
import torch
model_id = "runwayml/stable-diffusion-v1-5"
pipe = MultiResPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True)
pipe = pipe.to("cuda") 

string_to_param_dict = load_learned_concepts(pipe, "textual_inversion_outputs/")
```


Now that we have the pipeline ready, we can use three different sampling schemes: i) Fully Resolution-Dependent Sampling, ii) Semi Resolution-Dependent Sampling and iii) Fixed Resolution Sampling. We  show how to run each sampler and their visual outputs below. For the examples that will follow, we use the concept **jane** that we learned with re-croppings of the following work from the artist J. Perkins:

<center>
<img src="https://github.com/giannisdaras/multires_textual_inversion/blob/main/images/jane/fig4.jpg?raw=true" width="200"/>
</center>

#### Fully Resolution-Dependent Sampling
![](https://github.com/giannisdaras/multires_textual_inversion/blob/main/images/fully.png?raw=true)

To run the Fully Resolution-Dependent sampler, use the format: `<jane[number]>` to refer to the object at your prompt. For all the trained models, the number should be an integer in \[0, 9\], i.e. we learned a set of $10$ embeddings to describe the object (instead of one, as in Textual Inversion).

Example:

```
prompts = []

selected_i = [0, 3, 5, 7, 9]
for i in selected_i:
  prompts.append(f"An image of <jane[{i}]>")
 
images = pipe(prompts, string_to_param_dict, seed=42)
```


#### Semi Resolution-Dependent Sampling
![](https://github.com/giannisdaras/multires_textual_inversion/blob/main/images/semi.png?raw=true)


To run the Semi Resolution-Dependent sampler, use the format: `<jane(number)>` to refer to the object at your prompt. For all the trained models, the number should be an integer in \[0, 9\], i.e. we learned a set of $10$ embeddings to describe the object (instead of one, as in Textual Inversion).

Example:

```
prompts = []

selected_i = [0, 3, 5, 7, 9]
for i in selected_i:
  prompts.append(f"An image of <jane({i})>")
 
images = pipe(prompts, string_to_param_dict, seed=42)
```

#### Fixed Resolution Sampling
![](https://github.com/giannisdaras/multires_textual_inversion/blob/main/images/fixed.png?raw=true)

To run the Fixed Resolution sampler, use the format: `<jane|number|>` to refer to the object at your prompt. For all the trained models, the number should be an integer in \[0, 9\], i.e. we learned a set of $10$ embeddings to describe the object (instead of one, as in Textual Inversion).

Example:

```
prompts = []

selected_i = [0, 3, 5, 7, 9]
for i in selected_i:
  prompts.append(f"An image of <jane|{i}|>")
 
images = pipe(prompts, string_to_param_dict, seed=42)
```

### Generate samples with Dreambooth

Here, we show how to use the learned concepts to generate samples at different resolutions with DreamBooth.

Once you have trained (or downloaded) the learned concepts, you can create images at different resolutions, as shown below:

```
from diffusers import DreamBoothMultiResPipeline
import torch
model_id = "runwayml/stable-diffusion-v1-5"
pipe = DreamBoothMultiResPipeline.from_pretrained("dreambooth_outputs/multires_800/stan-smith", use_auth_token=True)
pipe = pipe.to("cuda")
image = pipe("An image of a <S(0)>", string_to_param_dict)[0]
loc = f"out_image.png"
image.save(loc)
```



## Train (your own) concepts

1. Run: `pip install -r requirements.txt` to install the python dependencies.
2. Get Huggingface token:
    `huggingface-cli login`
3. Create a datasets folder: `mkdir datasets`.
4. Put the images of the concept you want to learn in a folder under the datasets folder. Alternatively, you can run `python scrape_images.py` to download images from some of the most popular concepts. We also provide some datasets used in the Textual Inversion paper and our paper that you can download with the following commands: 
    ```
    gdown --id 1SDdCsKAMplUbWu1FO7hkc_QvGCuNlUqn
    mv datasets.zip datasets && unzip datasets/datasets.zip
    ```
5. Train the model to the new concept! You can either use Textual Inversion or Dreambooth.

    * Training with Textual Inversion:
        ```
        export MODEL_NAME="runwayml/stable-diffusion-v1-5"
        export CONCEPT_NAME="jane"
        accelerate launch train_textual_inversion.py \
            --pretrained_model_name_or_path=$MODEL_NAME  \
            --train_data_dir=datasets/$CONCEPT_NAME \
            --learnable_property="object" \
            --placeholder_token="S" \
            --initializer_token="painting" \
            --output_dir=textual_inversion_outputs/$CONCEPT_NAME \
            --resolution=512 \
            --train_batch_size=1 \
            --gradient_accumulation_steps=4 \
            --max_train_steps=3000 \
            --learning_rate=5.0e-04 \
            --scale_lr \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0
        ```
    * Training with Dreambooth:
        ```
        export MODEL_NAME="runwayml/stable-diffusion-v1-5"
        export CONCEPT_NAME="hitokomoru-style-nao"

        accelerate launch train_dreambooth.py \
        --pretrained_model_name_or_path=$MODEL_NAME  \
        --instance_data_dir=datasets/$CONCEPT_NAME \
        --output_dir=dreambooth_outputs/multires_100/$CONCEPT_NAME \
        --instance_prompt="S" \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=4 --gradient_checkpointing \
        --use_8bit_adam \
        --learning_rate=5e-6 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --max_train_steps=100
        ```


## References

If you find this work useful, please consider citing the following papers:

```
@misc{daras2022multires,
      url = {https://arxiv.org/abs/2211.17115},
      author = {Giannis Daras and Alexandros G. Dimakis},
      title = {Multiresolution Textual Inversion},
      publisher = {arXiv},
      year = {2022},
      primaryClass={cs.CV}
}
```

```
@misc{gal2022textual,
      doi = {10.48550/ARXIV.2208.01618},
      url = {https://arxiv.org/abs/2208.01618},
      author = {Gal, Rinon and Alaluf, Yuval and Atzmon, Yuval and Patashnik, Or and Bermano, Amit H. and Chechik, Gal and Cohen-Or, Daniel},
      title = {An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion},
      publisher = {arXiv},
      year = {2022},
      primaryClass={cs.CV}
}
```

