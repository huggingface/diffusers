import torch


class TextualInversionMixin:
    def load_textual_inversion_embeddings(self, embeddings):
        r"""
        Loads textual inversion embeddings. Receives a dictionary with the following keys:
        - `token`: name of the token to be added to the tokenizers' vocabulary
        - `embedding`: path to the embedding of the token to be added to the text encoder's embedding matrix

        Iters through the dictionary and adds the token to the tokenizer's vocabulary and the embedding to the text
        encoder's embedding matrix.
        """
        for token, embedding_path in embeddings.items():
            # check if token in tokenizer vocab
            # if yes, raise exception
            if token in self.tokenizer.get_vocab():
                raise ValueError(
                    f"Token {token} already in tokenizer vocabulary. Please choose a different token name."
                )

            # load embedding from embedding path then convert it to self.text_encoder's device and dtype
            embedding = torch.load(embedding_path)
            embedding = embedding.to(self.text_encoder.device)
            embedding = embedding.to(self.text_encoder.dtype)

            self.tokenizer.add_tokens([token])

            token_id = self.tokenizer.convert_tokens_to_ids(token)
            self.text_encoder.resize_token_embeddings(len(self.tokenizer) + 1)
            self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding
