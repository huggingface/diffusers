import pytest

from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin


class _TextualInversionPromptLoader(TextualInversionLoaderMixin):
    """Only the prompt conversion is under test, so a bare mixin host is enough here."""


def _add_embedding(tokenizer, token: str, num_vectors: int):
    added_tokens = [token] + [f"{token}_{i}" for i in range(1, num_vectors)]
    tokenizer.add_tokens(added_tokens)


@pytest.fixture
def clip_tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")


def test_multi_vector_token_with_uppercase_letters_is_expanded(clip_tokenizer):
    _add_embedding(clip_tokenizer, "MyToken", 3)

    prompt = _TextualInversionPromptLoader().maybe_convert_prompt("a photo of MyToken", clip_tokenizer)

    assert prompt == "a photo of MyToken MyToken_1 MyToken_2"
    assert clip_tokenizer.tokenize(prompt)[-3:] == ["mytoken", "mytoken_1", "mytoken_2"]


def test_multi_vector_token_is_expanded_when_the_prompt_uses_another_casing(clip_tokenizer):
    _add_embedding(clip_tokenizer, "neg", 3)

    prompt = _TextualInversionPromptLoader().maybe_convert_prompt("Neg, blurry", clip_tokenizer)

    assert prompt == "neg neg_1 neg_2, blurry"


def test_single_vector_token_is_left_alone(clip_tokenizer):
    _add_embedding(clip_tokenizer, "solo", 1)

    prompt = _TextualInversionPromptLoader().maybe_convert_prompt("a photo of Solo", clip_tokenizer)

    assert prompt == "a photo of Solo"
