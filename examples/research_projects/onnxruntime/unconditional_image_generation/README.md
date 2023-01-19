## Training examples

Creating a training image set is [described in a different document](https://huggingface.co/docs/datasets/image_process#image-datasets).

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the example folder  and run
```bash
pip install -r requirements.txt
```


And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

#### Use ONNXRuntime to accelerate training

In order to leverage onnxruntime to accelerate training, please use train_unconditional_ort.py

The command to train a DDPM UNet model on the Oxford Flowers dataset with onnxruntime:

```bash
accelerate launch train_unconditional_ort.py \
  --dataset_name="huggan/flowers-102-categories" \
  --resolution=64 \
  --output_dir="ddpm-ema-flowers-64" \
  --use_ema \
  --train_batch_size=16 \
  --num_epochs=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=fp16
  ```

Please contact Prathik Rao (prathikr), Sunghoon Choi (hanbitmyths), Ashwini Khade (askhade), or Peng Wang (pengwa) on github with any questions.