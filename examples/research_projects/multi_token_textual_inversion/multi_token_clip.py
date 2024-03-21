"""
The main idea for this code is to provide a way for users to not need to bother with the hassle of multiple tokens for a concept by typing
a photo of <concept>_0 <concept>_1 ... and so on
and instead just do
a photo of <concept>
which gets translated to the above. This needs to work for both inference and training.
For inference,
the tokenizer encodes the text. So, we would want logic for our tokenizer to replace the placeholder token with
it's underlying vectors
For training,
we would want to abstract away some logic like
1. Adding tokens
2. Updating gradient mask
3. Saving embeddings
to our Util class here.
so
TODO:
1. have tokenizer keep track of concept, multiconcept pairs and replace during encode call x
2. have mechanism for adding tokens x
3. have mech for saving emebeddings x
4. get mask to update x
5. Loading tokens from embedding x
6. Integrate to training x
7. Test
"""

import copy
import random

from transformers import CLIPTokenizer


class MultiTokenCLIPTokenizer(CLIPTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_map = {}

    def try_adding_tokens(self, placeholder_token, *args, **kwargs):
        num_added_tokens = super().add_tokens(placeholder_token, *args, **kwargs)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

    def add_placeholder_tokens(self, placeholder_token, *args, num_vec_per_token=1, **kwargs):
        output = []
        if num_vec_per_token == 1:
            self.try_adding_tokens(placeholder_token, *args, **kwargs)
            output.append(placeholder_token)
        else:
            output = []
            for i in range(num_vec_per_token):
                ith_token = placeholder_token + f"_{i}"
                self.try_adding_tokens(ith_token, *args, **kwargs)
                output.append(ith_token)
        # handle cases where there is a new placeholder token that contains the current placeholder token but is larger
        for token in self.token_map:
            if token in placeholder_token:
                raise ValueError(
                    f"The tokenizer already has placeholder token {token} that can get confused with"
                    f" {placeholder_token}keep placeholder tokens independent"
                )
        self.token_map[placeholder_token] = output

    def replace_placeholder_tokens_in_text(self, text, vector_shuffle=False, prop_tokens_to_load=1.0):
        """
        Here, we replace the placeholder tokens in text recorded in token_map so that the text_encoder
        can encode them
        vector_shuffle was inspired by https://github.com/rinongal/textual_inversion/pull/119
        where shuffling tokens were found to force the model to learn the concepts more descriptively.
        """
        if isinstance(text, list):
            output = []
            for i in range(len(text)):
                output.append(self.replace_placeholder_tokens_in_text(text[i], vector_shuffle=vector_shuffle))
            return output
        for placeholder_token in self.token_map:
            if placeholder_token in text:
                tokens = self.token_map[placeholder_token]
                tokens = tokens[: 1 + int(len(tokens) * prop_tokens_to_load)]
                if vector_shuffle:
                    tokens = copy.copy(tokens)
                    random.shuffle(tokens)
                text = text.replace(placeholder_token, " ".join(tokens))
        return text

    def __call__(self, text, *args, vector_shuffle=False, prop_tokens_to_load=1.0, **kwargs):
        return super().__call__(
            self.replace_placeholder_tokens_in_text(
                text, vector_shuffle=vector_shuffle, prop_tokens_to_load=prop_tokens_to_load
            ),
            *args,
            **kwargs,
        )

    def encode(self, text, *args, vector_shuffle=False, prop_tokens_to_load=1.0, **kwargs):
        return super().encode(
            self.replace_placeholder_tokens_in_text(
                text, vector_shuffle=vector_shuffle, prop_tokens_to_load=prop_tokens_to_load
            ),
            *args,
            **kwargs,
        )
