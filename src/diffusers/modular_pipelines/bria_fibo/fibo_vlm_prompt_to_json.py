import json
import math
import textwrap
from typing import Any, Dict, Iterable, List, Optional

import torch
from boltons.iterutils import remap
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen3VLForConditionalGeneration

from .. import ComponentSpec, InputParam, ModularPipelineBlocks, OutputParam, PipelineState


def parse_aesthetic_score(record: dict) -> str:
    ae = record["aesthetic_score"]
    if ae < 5.5:
        return "very low"
    elif ae < 6:
        return "low"
    elif ae < 7:
        return "medium"
    elif ae < 7.6:
        return "high"
    else:
        return "very high"


def parse_pickascore(record: dict) -> str:
    ps = record["pickascore"]
    if ps < 0.78:
        return "very low"
    elif ps < 0.82:
        return "low"
    elif ps < 0.87:
        return "medium"
    elif ps < 0.91:
        return "high"
    else:
        return "very high"


def prepare_clean_caption(record: dict) -> str:
    def keep(p, k, v):
        is_none = v is None
        is_empty_string = isinstance(v, str) and v == ""
        is_empty_dict = isinstance(v, dict) and not v
        is_empty_list = isinstance(v, list) and not v
        is_nan = isinstance(v, float) and math.isnan(v)
        if is_none or is_empty_string or is_empty_list or is_empty_dict or is_nan:
            return False
        return True

    try:
        scores = {}
        if "pickascore" in record:
            scores["preference_score"] = parse_pickascore(record)
        if "aesthetic_score" in record:
            scores["aesthetic_score"] = parse_aesthetic_score(record)

        clean_caption_dict = remap(record, visit=keep)

        # Set aesthetics scores
        if "aesthetics" not in clean_caption_dict:
            if len(scores) > 0:
                clean_caption_dict["aesthetics"] = scores
        else:
            clean_caption_dict["aesthetics"].update(scores)

        # Dumps clean structured caption as minimal json string (i.e. no newlines\whitespaces seps)
        clean_caption_str = json.dumps(clean_caption_dict)
        return clean_caption_str
    except Exception as ex:
        print("Error: ", ex)
        raise ex


def _collect_images(messages: Iterable[Dict[str, Any]]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                continue
            image_value = item.get("image")
            if isinstance(image_value, Image.Image):
                images.append(image_value)
            else:
                raise ValueError("Expected PIL.Image for image content in messages.")
    return images


def _strip_stop_sequences(text: str, stop_sequences: Optional[List[str]]) -> str:
    if not stop_sequences:
        return text.strip()
    cleaned = text
    for stop in stop_sequences:
        if not stop:
            continue
        index = cleaned.find(stop)
        if index >= 0:
            cleaned = cleaned[:index]
    return cleaned.strip()


class TransformersEngine(torch.nn.Module):
    """Inference wrapper using Hugging Face transformers."""

    def __init__(
        self,
        model: str,
        *,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super(TransformersEngine, self).__init__()
        default_processor_kwargs: Dict[str, Any] = {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1024 * 28 * 28,
        }
        processor_kwargs = {**default_processor_kwargs, **(processor_kwargs or {})}
        model_kwargs = model_kwargs or {}

        self.processor = AutoProcessor.from_pretrained(model, **processor_kwargs)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model,
            dtype=torch.bfloat16,
            **model_kwargs,
        )
        self.model.eval()

        tokenizer_obj = self.processor.tokenizer
        if tokenizer_obj.pad_token_id is None:
            tokenizer_obj.pad_token = tokenizer_obj.eos_token
        self._pad_token_id = tokenizer_obj.pad_token_id
        eos_token_id = tokenizer_obj.eos_token_id
        if isinstance(eos_token_id, list) and eos_token_id:
            self._eos_token_id = eos_token_id
        elif eos_token_id is not None:
            self._eos_token_id = [eos_token_id]
        else:
            raise ValueError("Tokenizer must define an EOS token for generation.")

    def dtype(self) -> torch.dtype:
        return self.model.dtype

    def device(self) -> torch.device:
        return self.model.device

    def _to_model_device(self, value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        target_device = getattr(self.model, "device", None)
        if target_device is None or target_device.type == "meta":
            return value
        if value.device == target_device:
            return value
        return value.to(target_device)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> str:
        tokenizer = self.processor.tokenizer
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_inputs: Dict[str, Any] = {
            "text": [prompt_text],
            "padding": True,
            "return_tensors": "pt",
        }
        images = _collect_images(messages)
        if images:
            processor_inputs["images"] = images
        inputs = self.processor(**processor_inputs)
        inputs = {key: self._to_model_device(value) for key, value in inputs.items()}

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "eos_token_id": self._eos_token_id,
            "pad_token_id": self._pad_token_id,
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Processor did not return input_ids; cannot compute new tokens.")
        new_token_ids = generated_ids[:, input_ids.shape[-1] :]
        decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        if not decoded:
            return ""
        text = decoded[0]
        stripped_text = _strip_stop_sequences(text, stop)
        json_prompt = json.loads(stripped_text)
        return json_prompt


def generate_json_prompt(
    vlm_processor: AutoModelForCausalLM,
    top_p: float,
    temperature: float,
    max_tokens: int,
    stop: List[str],
    image: Optional[Image.Image] = None,
    prompt: Optional[str] = None,
    structured_prompt: Optional[str] = None,
):
    if image is None and structured_prompt is None:
        # only got prompt
        task = "generate"
        editing_instructions = None
    elif image is None and structured_prompt is not None and prompt is not None:
        # got structured prompt and prompt
        task = "refine"
        editing_instructions = prompt
    elif image is not None and structured_prompt is None and prompt is not None:
        # got image and prompt
        task = "refine"
        editing_instructions = prompt
    elif image is not None and structured_prompt is None and prompt is None:
        # only got image
        task = "inspire"
        editing_instructions = None
    else:
        raise ValueError("Invalid input")

    messages = build_messages(
        task,
        image=image,
        prompt=prompt,
        structured_prompt=structured_prompt,
        editing_instructions=editing_instructions,
    )

    generated_prompt = vlm_processor.generate(
        messages=messages, top_p=top_p, temperature=temperature, max_tokens=max_tokens, stop=stop
    )
    cleaned_json_data = prepare_clean_caption(generated_prompt)
    return cleaned_json_data


def build_messages(
    task: str,
    *,
    image: Optional[Image.Image] = None,
    refine_image: Optional[Image.Image] = None,
    prompt: Optional[str] = None,
    structured_prompt: Optional[str] = None,
    editing_instructions: Optional[str] = None,
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []

    if task == "inspire":
        user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": "<inspire>"})
    elif task == "generate":
        text_value = (prompt or "").strip()
        formatted = f"<generate>\n{text_value}"
        user_content.append({"type": "text", "text": formatted})
    else:  # refine
        if refine_image is None:
            base_prompt = (structured_prompt or "").strip()
            edits = (editing_instructions or "").strip()
            formatted = textwrap.dedent(f"""<refine> Input: {base_prompt} Editing instructions: {edits}""").strip()
            user_content.append({"type": "text", "text": formatted})
        else:
            user_content.append({"type": "image", "image": refine_image})
            edits = (editing_instructions or "").strip()
            formatted = textwrap.dedent(f"""<refine> Editing instructions: {edits}""").strip()
            user_content.append({"type": "text", "text": formatted})

    messages: List[Dict[str, Any]] = []
    messages.append({"role": "user", "content": user_content})
    return messages


class BriaFiboVLMPromptToJson(ModularPipelineBlocks):
    model_name = "BriaFibo"

    def __init__(self, model_id):
        super().__init__()
        self.engine = TransformersEngine(model_id)
        self.engine.model.to("cuda")

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return []

    @property
    def inputs(self) -> List[InputParam]:
        prompt_input = InputParam(
            "prompt",
            type_hint=str,
            required=False,
            description="Prompt to use",
        )
        image_input = InputParam(
            name="image", type_hint=Image.Image, required=False, description="image for inspiration mode"
        )
        json_prompt_input = InputParam(
            name="json_prompt", type_hint=str, required=False, description="JSON prompt to use"
        )
        sampling_top_p_input = InputParam(
            name="sampling_top_p", type_hint=float, required=False, description="Sampling top p", default=0.9
        )
        sampling_temperature_input = InputParam(
            name="sampling_temperature",
            type_hint=float,
            required=False,
            description="Sampling temperature",
            default=0.2,
        )
        sampling_max_tokens_input = InputParam(
            name="sampling_max_tokens", type_hint=int, required=False, description="Sampling max tokens", default=4096
        )
        return [
            prompt_input,
            image_input,
            json_prompt_input,
            sampling_top_p_input,
            sampling_temperature_input,
            sampling_max_tokens_input,
        ]

    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return []

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "json_prompt",
                type_hint=str,
                description="JSON prompt by the VLM",
            )
        ]

    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        prompt = block_state.prompt
        image = block_state.image
        json_prompt = block_state.json_prompt
        block_state.json_prompt = generate_json_prompt(
            vlm_processor=self.engine,
            image=image,
            prompt=prompt,
            structured_prompt=json_prompt,
            top_p=block_state.sampling_top_p,
            temperature=block_state.sampling_temperature,
            max_tokens=block_state.sampling_max_tokens,
            stop=["<|im_end|>", "<|end_of_text|>"],
        )
        self.set_block_state(state, block_state)

        return components, state
