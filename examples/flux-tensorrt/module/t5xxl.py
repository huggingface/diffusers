from .engine import Engine

class T5XXLModel(Engine):
    def __init__(self, engine_path: str, stream = None):
        super().__init__(engine_path, stream)
        self.text_maxlen = 512
        self.d_model = 4096

        # Load engine before
        self.load_engine()

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.d_model),
        }
        return output