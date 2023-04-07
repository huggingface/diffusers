from diffusers import StableDiffusionPipeline
import torch
import os
import json
from math import ceil, sqrt
from PIL import Image

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model

def save_image(pipeline, prompt, path):
    output = pipeline(prompt=prompt)
    image = output.images[0]
    nsfw = output.nsfw_content_detected
    image.save(path)
    return nsfw

def concat_images_in_square_grid(folder_path, prompt, output_path='output.png'):
    # Get a list of all .png files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') and prompt in f]

    # Calculate the number of rows and columns for the grid
    total_images = len(image_files)
    grid_size = ceil(sqrt(total_images))
    rows, cols = grid_size, grid_size

    # Open the first image to get its dimensions
    first_image = Image.open(os.path.join(folder_path, image_files[0]))
    image_width, image_height = first_image.size

    # Create a blank image for the grid
    grid_width, grid_height = image_width * cols, image_height * rows
    grid = Image.new('RGB', (grid_width, grid_height))

    # Iterate through the images and add them to the grid
    for index, image_file in enumerate(image_files):
        image = Image.open(os.path.join(folder_path, image_file))
        x = (index % cols) * image_width
        y = (index // cols) * image_height
        grid.paste(image, (x, y))

    # Save the output image
    grid.save(output_path)