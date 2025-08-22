from .engine import Engine

class CLIPModel(Engine):
    def __init__(self, engine_path: str, stream = None):
        super().__init__(engine_path, stream)
        self.text_maxlen = 77
        self.embedding_dim = 768
        self.keep_pooled_output = True

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