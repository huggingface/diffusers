from .engine import Engine

class VAEModel(Engine):
    def __init__(self, engine_path: str, stream = None):
        super().__init__(engine_path, stream)
        self.latent_channels = 16
        self.scaling_factor = 0.3611
        self.shift_factor = 0.1159

        # Load engine before
        self.load_engine()

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, self.latent_channels, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }
