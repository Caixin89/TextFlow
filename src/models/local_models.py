import logging

import torch
from qwen_vl_utils import process_vision_info
from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                          LlavaNextForConditionalGeneration,
                          LlavaNextProcessor, MllamaForConditionalGeneration,
                          Qwen2VLForConditionalGeneration, pipeline)

from config import config

max_new_tokens = config["model_config"]["max_new_tokens"]
do_sample = config["model_config"]["do_sample"]
temperature = config["model_config"]["temperature"]
top_k = config["model_config"]["top_k"]
top_p = config["model_config"]["top_p"]


def load_local_model(model_name):
    logger = logging.getLogger(__name__)
    model_id = config["model_version"].get(model_name)

    # Load LLM and tokenizer
    if model_name in [
        "Llama-3.1-8B",
        "Llama-3.1-70B",
        "Mixtral-8x22B",
        "Phi-3.5-mini",
        "Phi-3.5-MoE",
        "Qwen2.5-7B",
        "Qwen2.5-14B",
        "Qwen2.5-32B",
        "Qwen2.5-72B",
    ]:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        trust_remote_code = model_name in ["Phi-3.5-mini", "Phi-3.5-MoE"]
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
        )
        return model, tokenizer
    # Load VLM and processor
    else:
        if model_name == "Llava-v1.6-110b":
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="auto"
            )
            processor = LlavaNextProcessor.from_pretrained(model_id)
        elif model_name in ["Llama-3.2-11B", "Llama-3.2-90B"]:
            model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(model_id)
        elif model_name in ["Qwen2-VL-7B", "Qwen2-VL-72B"]:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(model_id)
        else:
            logger.error(f"Local model {model_name} is not supported.")
            raise ValueError(f"Local model {model_name} is not supported.")
        return model, processor


def generate_local_response(model_name, model, tokenizer, messages, image=None):
    logger = logging.getLogger(__name__)
    model_id = config["model_version"].get(model_name)
    processor = tokenizer

    if not image:
        if model_name in ["Qwen2.5-7B", "Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-72B"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=None if temperature == 0 else temperature,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]
        elif model_name in ["Phi-3.5-mini", "Phi-3.5-MoE"]:
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

            generation_args = {
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
                "do_sample": do_sample,
                "top_p": top_p,
                "temperature": (None if temperature == 0 else temperature),
            }

            output = pipe(messages, **generation_args)
            response = output[0]["generated_text"]
        elif model_name in ["Llama-3.1-8B", "Llama-3.1-70B", "Mixtral-8x22B"]:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=None if temperature == 0 else temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

            response = outputs[0][input_ids.shape[-1] :]
            response = tokenizer.decode(response, skip_special_tokens=True)
        elif model_name in ["Llama-3.2-11B", "Llama-3.2-90B", "Llava-v1.6-110b"]:
            input_text = processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = processor(
                text=input_text, add_special_tokens=False, return_tensors="pt"
            ).to(model.device)

            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=None if temperature == 0 else temperature,
            )
            decoded_output = processor.decode(output[0], skip_special_tokens=True)
            start_keyword = "assistant"
            start_pos = decoded_output.find(start_keyword) + len(start_keyword)
            response = decoded_output[start_pos:].strip()
        elif model_name in ["Qwen2-VL-7B", "Qwen2-VL-72B"]:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=None,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=None if temperature == 0 else temperature,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            response = output_text[0]
        else:
            logger.error(f"Model {model_name} has not been implemented.")
            raise ValueError("Unsupported model.")
    # Messages include image and text
    else:
        if model_name in ["Llama-3.2-11B", "Llama-3.2-90B", "Llava-v1.6-110b"]:
            input_text = processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = processor(
                image, input_text, add_special_tokens=False, return_tensors="pt"
            ).to(model.device)

            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=None if temperature == 0 else temperature,
            )
            decoded_output = processor.decode(output[0], skip_special_tokens=True)
            start_keyword = "assistant"
            start_pos = decoded_output.find(start_keyword) + len(start_keyword)
            response = decoded_output[start_pos:].strip()
        elif model_name in ["Qwen2-VL-7B", "Qwen2-VL-72B"]:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=None if temperature == 0 else temperature,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            response = output_text[0]
        else:
            logger.error(f"Model {model_name} has not been implemented.")
            raise ValueError(f"Model {model_name} has not been implemented.")
    return response


def generate_local_response_tool_use(
    model_name, model, tokenizer, messages, representation
):
    logger = logging.getLogger(__name__)
    logger.error(f"Model {model_name} has not been impelmented for tool use.")
    raise ValueError(f"Model {model_name} has not been impelmented for tool use.")
