import torch


class TextualInversionMixin:
    r"""
    Mixin class for adding textual inversion tokens and embeddings to the tokenizer and text encoder with method:
    - [`~TextualInversionMixin.load_textual_inversion_embeddings`]
    - [`~TextualInversionMixin.add_textual_inversion_embedding`]

    Class attributes:
    - **textual_inversion_tokens** (`List[str]`): list of tokens added to the tokenizer's vocabulary and the text
      encoder's embedding matrix
    """
    textual_inversion_tokens = []

    def load_textual_inversion_embeddings(self, embeddings):
        r"""
        Loads textual inversion embeddings.

        Receives a dictionary with the following keys:
        - `token`: name of the token to be added to the tokenizers' vocabulary
        - `embedding`: path to the embedding of the token to be added to the text encoder's embedding matrix

        Alternatively, it can receive a list of pathes to embedding dictionaries, where the keys are the tokens and the
        values are the embeddings. In that case, it will iterate through the list and add the tokens and embeddings to
        the tokenizer's vocabulary and the text encoder's embedding matrix.

        Iters through the dictionary and adds the token to the tokenizer's vocabulary and the embedding to the text
        encoder's embedding matrix.
        """

        if isinstance(embeddings, dict):
            for token, embedding_path in embeddings.items():
                # check if token in tokenizer vocab
                # if yes, raise exception
                if token in self.tokenizer.get_vocab():
                    raise ValueError(
                        f"Token {token} already in tokenizer vocabulary. Please choose a different token name."
                    )

                embedding_dict = torch.load(embedding_path)
                embedding = list(embedding_dict.values())[0]

                self.add_textual_inversion_embedding(token, embedding)

        elif isinstance(embeddings, list):
            for embedding_path in embeddings:
                embedding_dict = torch.load(embedding_path)
                token = list(embedding_dict.keys())[0]
                embedding = embedding_dict[token]

                # check if token in tokenizer vocab
                # if yes, raise exception
                if token in self.tokenizer.get_vocab():
                    raise ValueError(
                        f"Token {token} already in tokenizer vocabulary. Please choose a different token name."
                    )
                self.add_textual_inversion_embedding(token, embedding)

    def add_textual_inversion_embedding(self, token, embedding):
        r"""
        Adds a token to the tokenizer's vocabulary and an embedding to the text encoder's embedding matrix.
        """
        # check if token in tokenizer vocab
        # if yes, raise exception
        if token in self.tokenizer.get_vocab():
            raise ValueError(f"Token {token} already in tokenizer vocabulary. Please choose a different token name.")

        embedding = embedding.to(self.text_encoder.device)
        embedding = embedding.to(self.text_encoder.dtype)

        self.tokenizer.add_tokens([token])

        token_id = self.tokenizer.convert_tokens_to_ids(token)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer) + 1)
        self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding

        self.textual_inversion_tokens.append(token)
