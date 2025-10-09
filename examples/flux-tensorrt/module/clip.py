from .engine import Engine


class CLIPModel(Engine):
    def __init__(self, engine_path: str, stream=None, build=False):
        super().__init__(engine_path, stream)
        self.text_maxlen = 77
        self.embedding_dim = 768
        self.keep_pooled_output = True

        if not build:
            # Load engine before
            self.load_engine()

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }
        if self.keep_pooled_output:
            output["pooled_embeddings"] = (batch_size, self.embedding_dim)
        return output

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

        return {
            "input_ids": [
                (min_batch, self.text_maxlen),
                (opt_batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }
