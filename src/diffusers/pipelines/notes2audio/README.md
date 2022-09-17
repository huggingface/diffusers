# TODO Follow the stable diffusion pipeline card


Goal of the the implementation : 

```python 

from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("magenta/notes2audio_base_with_context")

midi_setup_file = "path/to/midi_file.midi" 
pipeline(midi_setup_file).sample[0]



```
