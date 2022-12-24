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

"""
import torch
import random
from transformers import CLIPTokenizer
import os
import copy

class EmbeddingUtil:
    def __init__(self, tokenizer, text_encoder):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
    def add_tokens(self, placeholder_token, num_vec_per_token=1, initializer_token=None):
        """
        Add tokens to the tokenizer and set the initial value of token embeddings
        """
        placeholder_tokens=self.tokenizer.add_tokens(placeholder_token, num_vec_per_token)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        placeholder_token_ids = self.tokenizer.encode(placeholder_tokens, add_special_tokens=False)
        if initializer_token:
            token_ids = self.tokenizer.encode(initializer_token, add_special_tokens=False)
            for i, placeholder_token_id in enumerate(placeholder_token_ids):
                    token_embeds[placeholder_token_id] = token_embeds[token_ids[i * len(token_ids)//num_vec_per_token]]
        else:
            for i, placeholder_token_id in enumerate(placeholder_token_ids):
                    token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[placeholder_token_id])
    
    def save_progress(self, accelerator, output_dir):
        for placeholder_token in self.tokenizer.token_map:
            placeholder_tokens = " ".join(self.tokenizer.token_map[placeholder_token])
            placeholder_token_ids = self.tokenizer.encode(placeholder_tokens)
            learned_embeds = accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[placeholder_token_ids]
            learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
            torch.save(learned_embeds_dict, os.path.join(output_dir, f"learned_embeds_{placeholder_token}.bin"))
    def get_mask(self):
        # Get the mask of the weights that won't change
        mask = torch.ones(len(self.tokenizer))
        for placeholder_token in self.tokenizer.token_map:
            placeholder_tokens = " ".join(self.tokenizer.token_map[placeholder_token])
            placeholder_token_ids = self.tokenizer.encode(placeholder_tokens)
            for i in range(len(placeholder_token_ids)):
                mask = mask & (torch.arange(len(self.tokenizer)) != placeholder_token_ids[i])
        return mask

class MultiTokenCLIPTokenizer(CLIPTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_map = {}


    def try_adding_tokens(self, placeholder_token):
        num_added_tokens = super().add_tokens(placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
    def add_tokens(self, placeholder_token, num_vec_per_token=1):
        output = []
        if num_vec_per_token == 1:
            super().try_adding_tokens(placeholder_token)
            output.append(placeholder_token)
        else:
            output = []
            for i in range(num_vec_per_token):
                ith_token = placeholder_token+f'_{i}'
                super().try_adding_tokens(ith_token)
                output.append(ith_token)
        self.token_map[placeholder_token] = output
        return " ".join(output)
    def encode(self, text, *args, token_shuffle=False, **kwargs):
        """
        Here, we replace the placeholder tokens in text recorded in token_map so that the text_encoder
        can encode them
        token_shuffle was inspired by https://github.com/rinongal/textual_inversion/pull/119
        where shuffling tokens were found to force the model to learn the concepts more descriptively.
        """
        for placeholder_token in self.token_map:
            if placeholder_token in self.token_map:
                tokens = self.token_map[placeholder_token]
                if token_shuffle:
                    tokens = copy.copy(tokens)
                    random.shuffle(tokens)
                text = text.replace(placeholder_token, " ".join(tokens))
        return super().encode(text, *args, **kwargs)