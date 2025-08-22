from .engine import Engine

class FluxTransformerModel(Engine):
    def __init__(self, engine_path: str, stream = None):
        super().__init__(engine_path, stream)
        self.in_channels = 64

        # Load engine before
        self.load_engine()

    def get_shape_dict(self,batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        shape_dict = {
            "hidden_states": (batch_size, (latent_height // 2) * (latent_width // 2), 64),
            "encoder_hidden_states": (batch_size, 512, 4096),
            "pooled_projections": (batch_size, 768),
            "timestep": (batch_size,),
            "img_ids": ((latent_height // 2) * (latent_width // 2), 3),
            "txt_ids": (512, 3),
            "latent": (batch_size, (latent_height // 2) * (latent_width // 2), 64),
            "guidance": (batch_size,),
        }

        return shape_dict