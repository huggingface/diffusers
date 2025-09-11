from .engine import Engine


class VAEModel(Engine):
    def __init__(self, engine_path: str, stream=None, build=False):
        super().__init__(engine_path, stream)
        self.latent_channels = 16
        self.scaling_factor = 0.3611
        self.shift_factor = 0.1159
        self.compression_factor = 8

        if not build:
            # Load engine before
            self.load_engine()

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, self.latent_channels, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_input_profile(
        self,
        opt_batch_size=1,
        opt_image_height=1024,
        opt_image_width=1024,
        min_batch=1,
        max_batch=8,
        min_height=512,
        max_height=1280,
        min_width=512,
        max_width=1280,
        static_batch=True,
        dynamic_shape=True,
    ):
        min_batch = opt_batch_size if static_batch else min_batch
        max_batch = opt_batch_size if static_batch else max_batch

        latent_height = opt_image_height // self.compression_factor
        latent_width = opt_image_width // self.compression_factor
        min_latent_height = min_height // self.compression_factor if dynamic_shape else latent_height
        min_latent_width = min_width // self.compression_factor if dynamic_shape else latent_width
        max_latent_height = max_height // self.compression_factor if dynamic_shape else latent_height
        max_latent_width = max_width // self.compression_factor if dynamic_shape else latent_width

        return {
            "latent": [
                (min_batch, self.latent_channels, min_latent_height, min_latent_width),
                (opt_batch_size, self.latent_channels, latent_height, latent_width),
                (max_batch, self.latent_channels, max_latent_height, max_latent_width),
            ]
        }
