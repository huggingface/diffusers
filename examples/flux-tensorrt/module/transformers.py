from .engine import Engine


class FluxTransformerModel(Engine):
    def __init__(self, engine_path: str, stream=None, build=False):
        super().__init__(engine_path, stream)
        self.in_channels = 64
        self.joint_attention_dim = 4096
        self.pooled_projection_dim = 768
        self.text_maxlen = 512
        self.compression_factor = 8

        if not build:
            # Load engine before
            self.load_engine()

    def get_shape_dict(self, batch_size, image_height, image_width):
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

        input_profile = {
            "hidden_states": [
                (min_batch, (min_latent_height // 2) * (min_latent_width // 2), self.in_channels),
                (opt_batch_size, (latent_height // 2) * (latent_width // 2), self.in_channels),
                (max_batch, (max_latent_height // 2) * (max_latent_width // 2), self.in_channels),
            ],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.joint_attention_dim),
                (opt_batch_size, self.text_maxlen, self.joint_attention_dim),
                (max_batch, self.text_maxlen, self.joint_attention_dim),
            ],
            "pooled_projections": [
                (min_batch, self.pooled_projection_dim),
                (opt_batch_size, self.pooled_projection_dim),
                (max_batch, self.pooled_projection_dim),
            ],
            "timestep": [(min_batch,), (opt_batch_size,), (max_batch,)],
            "img_ids": [
                ((min_latent_height // 2) * (min_latent_width // 2), 3),
                ((latent_height // 2) * (latent_width // 2), 3),
                ((max_latent_height // 2) * (max_latent_width // 2), 3),
            ],
            "txt_ids": [(self.text_maxlen, 3), (self.text_maxlen, 3), (self.text_maxlen, 3)],
        }

        input_profile["guidance"] = [(min_batch,), (opt_batch_size,), (max_batch,)]
        return input_profile
