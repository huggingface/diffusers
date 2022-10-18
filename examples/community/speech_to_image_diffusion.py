from diffusers import DiffusionPipeline
from transformers import WhisperModel,pipeline
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

audio_sample = ds[3]

text = audio_sample["text"].lower()
speech_data = audio_sample["audio"]["array"]
speech_file = audio_sample["file"]

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

text_prompt = pipe(speech_file)['text']

pipediff = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

pipediff = pipediff.to(device)

pipeline_output = pipediff(text_prompt)
print(pipeline_output.images[0])
plt.imshow(pipeline_output.images[0])