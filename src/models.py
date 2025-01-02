import json
import logging
import sys

import torch
from anthropic import Anthropic
from openai import OpenAI
from qwen_vl_utils import process_vision_info
from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                          LlavaNextForConditionalGeneration,
                          LlavaNextProcessor, MllamaForConditionalGeneration,
                          Qwen2VLForConditionalGeneration, pipeline)

from config import config
from mermaid_parser import Mermaid2Flowchart

# Constants
GPT4O_MODELS = ["gpt-4o", "gpt-4o-mini"]
# Open source Large Language Models (LLMs)
LLAMA3_1_MODELS = ["Llama-3.1-8B", "Llama-3.1-70B"]
MIXTRAL_MODELS = ["Mixtral-8x22B"]
PHI_MODELS = ["Phi-3.5-mini", "Phi-3.5-MoE"]
QWEN2_5_MODELS = ["Qwen2.5-7B", "Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-72B"]
LLMS = LLAMA3_1_MODELS + MIXTRAL_MODELS + PHI_MODELS + QWEN2_5_MODELS
# Open source Vision Language Models (VLMs)
LLAMA3_2_MODELS = ["Llama-3.2-11B", "Llama-3.2-90B"]
LLAVA_MODELS = ["llava-v1.6-110b"]
QWEN2_VL_MODELS = ["Qwen2-VL-7B", "Qwen2-VL-72B"]

CONFIG = config["model_config"]


# Load LLM with tokenizer or VLM with processor
def load_model_and_tokenizer(model_name):
    logger = logging.getLogger(__name__)
    model_map = {
        "claude-3-5-sonnet": "claude-3-5-sonnet",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        # Open-source LLMs
        "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Llama-3.1-70B": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "Mixtral-8x22B": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "Phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
        "Phi-3.5-MoE": "microsoft/Phi-3.5-MoE-instruct",
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5-14B": "Qwen/Qwen2.5-14B-Instruct",
        "Qwen2.5-32B": "Qwen/Qwen2.5-32B-Instruct",
        "Qwen2.5-72B": "Qwen/Qwen2.5-72B-Instruct",
        # Open-source VLMs
        "Llama-3.2-11B": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-90B": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "llava-v1.6-110b": "llava-hf/llava-next-110b-hf",
        "Qwen2-VL-7B": "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen2-VL-72B": "Qwen/Qwen2-VL-72B-Instruct",
    }
    model_id = model_map.get(model_name)

    if not model_id:
        logger.error(f"Model {model_name} has not been implemented.")
        raise ValueError("Unsupported model.")

    logger.info(f"Loading model {model_name}...")
    for arg, value in CONFIG.items():
        logger.info(f"{arg}: {value}")

    if model_name == "claude-3-5-sonnet":
        client = Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=config["api_keys"]["ANTHROPIC_API_KEY"]
        )
        return client, None
    elif model_name in GPT4O_MODELS:
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=config["api_keys"]["OPENAI_API_KEY"]
        )
        return client, None
    # Load LLM and tokenizer
    elif model_name in LLMS:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        trust_remote_code = model_name in PHI_MODELS
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
        )
        return model, tokenizer
    # Load VLM and processor
    else:
        if model_name in LLAVA_MODELS:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="auto"
            )
            processor = LlavaNextProcessor.from_pretrained(model_id)
        elif model_name in LLAMA3_2_MODELS:
            model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(model_id)
        elif model_name in QWEN2_VL_MODELS:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(model_id)
        else:
            logger.error(f"Model {model_name} has not been implemented.")
            raise ValueError("Unsupported model.")
        return model, processor


def load_messages(model_name, prompt, image=None):
    # Text only messages (Note that VLMs can be used as LLMs without image input)
    if not image:
        templates = {
            # Close sour LLMs
            "claude-3-5-sonnet": [
                {"role": "user", "content": prompt},
            ],
            "gpt-4o": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "gpt-4o-mini": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            # Open source LLMs
            "Llama-3.1-8B": [
                {
                    "role": "system",
                    "content": "Cutting Knowledge Date: December 2023\nToday Date: 23 July 2024\n\nYou are a helpful assistant",
                },
                {"role": "user", "content": prompt},
            ],
            "Llama-3.1-70B": [
                {
                    "role": "system",
                    "content": "Cutting Knowledge Date: December 2023\nToday Date: 23 July 2024\n\nYou are a helpful assistant",
                },
                {"role": "user", "content": prompt},
            ],
            "Mixtral-8x22B": [{"role": "user", "content": prompt}],
            "Phi-3.5-mini": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            "Phi-3.5-MoE": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            "Qwen2.5-7B": [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "Qwen2.5-14B": [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "Qwen2.5-32B": [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "Qwen2.5-72B": [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            # Open source VLMs
            "Llama-3.2-11B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "Llama-3.2-90B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "llava-v1.6-110b": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "Qwen2-VL-7B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "Qwen2-VL-72B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }
    # Messages include both image and text
    else:
        templates = {
            # Close source VLMs
            "claude-3-5-sonnet": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "gpt-4o": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            "gpt-4o-mini": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            # Open source VLMs
            "Llama-3.2-11B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "Llama-3.2-90B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "llava-v1.6-110b": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                },
            ],
            "Qwen2-VL-7B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "Qwen2-VL-72B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }
    return templates.get(model_name)


def generate_response(model_name, model, tokenizer, prompt, image=None):
    logger = logging.getLogger(__name__)
    # tokenizer is used by open source LLMs while processor is used by open source VLMs
    processor = tokenizer
    messages = load_messages(model_name, prompt, image)

    # Models via api calls
    if model_name == "claude-3-5-sonnet":
        client = model
        message = client.messages.create(
            model=config["model_version"][model_name],
            max_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"],
            system="",
            messages=messages,
        )
        response = message.content[0].text
        return response
    elif model_name in GPT4O_MODELS:
        client = model
        completion = client.chat.completions.create(
            model=config["model_version"][model_name],
            max_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"],
            messages=messages,
        )
        response = completion.choices[0].message.content
        return response

    # Models run locally
    # Text only messages
    if not image:
        if model_name in QWEN2_5_MODELS:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=CONFIG["max_new_tokens"],
                do_sample=CONFIG["do_sample"],
                top_p=CONFIG["top_p"],
                top_k=CONFIG["top_k"],
                temperature=(
                    None if CONFIG["temperature"] == 0 else CONFIG["temperature"]
                ),
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]
        elif model_name in PHI_MODELS:
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

            generation_args = {
                "max_new_tokens": CONFIG["max_new_tokens"],
                "return_full_text": False,
                "do_sample": CONFIG["do_sample"],
                "top_p": CONFIG["top_p"],
                "temperature": (
                    None if CONFIG["temperature"] == 0 else CONFIG["temperature"]
                ),
            }

            output = pipe(messages, **generation_args)
            response = output[0]["generated_text"]
        elif model_name in LLAMA3_1_MODELS + MIXTRAL_MODELS:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                input_ids,
                max_new_tokens=CONFIG["max_new_tokens"],
                do_sample=CONFIG["do_sample"],
                top_p=CONFIG["top_p"],
                temperature=(
                    None if CONFIG["temperature"] == 0 else CONFIG["temperature"]
                ),
                pad_token_id=tokenizer.eos_token_id,
            )

            response = outputs[0][input_ids.shape[-1] :]
            response = tokenizer.decode(response, skip_special_tokens=True)
        elif model_name in LLAMA3_2_MODELS + LLAVA_MODELS:
            input_text = processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = processor(
                text=input_text, add_special_tokens=False, return_tensors="pt"
            ).to(model.device)

            output = model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_new_tokens"],
                do_sample=CONFIG["do_sample"],
                top_p=CONFIG["top_p"],
                temperature=(
                    None if CONFIG["temperature"] == 0 else CONFIG["temperature"]
                ),
            )
            decoded_output = processor.decode(output[0], skip_special_tokens=True)
            start_keyword = "assistant"
            start_pos = decoded_output.find(start_keyword) + len(start_keyword)
            response = decoded_output[start_pos:].strip()
        elif model_name in QWEN2_VL_MODELS:
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
                max_new_tokens=CONFIG["max_new_tokens"],
                do_sample=CONFIG["do_sample"],
                top_p=CONFIG["top_p"],
                top_k=CONFIG["top_k"],
                temperature=(
                    None if CONFIG["temperature"] == 0 else CONFIG["temperature"]
                ),
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
        if model_name in LLAMA3_2_MODELS + LLAVA_MODELS:
            input_text = processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = processor(
                image, input_text, add_special_tokens=False, return_tensors="pt"
            ).to(model.device)

            output = model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_new_tokens"],
                do_sample=CONFIG["do_sample"],
                top_p=CONFIG["top_p"],
                temperature=(
                    None if CONFIG["temperature"] == 0 else CONFIG["temperature"]
                ),
            )
            decoded_output = processor.decode(output[0], skip_special_tokens=True)
            start_keyword = "assistant"
            start_pos = decoded_output.find(start_keyword) + len(start_keyword)
            response = decoded_output[start_pos:].strip()
        elif model_name in QWEN2_VL_MODELS:
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
                max_new_tokens=CONFIG["max_new_tokens"],
                do_sample=CONFIG["do_sample"],
                top_p=CONFIG["top_p"],
                top_k=CONFIG["top_k"],
                temperature=(
                    None if CONFIG["temperature"] == 0 else CONFIG["temperature"]
                ),
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
    return response


def load_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_number_of_nodes",
                "description": "Returns the number of nodes in the flowchart.",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_number_of_edges",
                "description": "Returns the number of edges in the flowchart.",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_direct_successors",
                "description": "Returns the direct successors (outgoing connections) of the given node by its description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_description": {
                            "type": "string",
                            "description": "The description of the given node.",
                        },
                    },
                    "required": ["node_description"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_direct_predecessors",
                "description": "Returns the direct predecessors (incoming connections) of the given node by its description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_description": {
                            "type": "string",
                            "description": "The description of the given node.",
                        },
                    },
                    "required": ["node_description"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_shortest_path_length",
                "description": "Returns the number of edges in the shortest path between two nodes based on their descriptions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_node_description": {
                            "type": "string",
                            "description": "The description of the start node.",
                        },
                        "end_node_description": {
                            "type": "string",
                            "description": "The description of the end node.",
                        },
                    },
                    "required": ["start_node_description", "end_node_description"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_max_indegree",
                "description": "Returns the maximum indegree (number of incoming edges) for any node.",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_max_outdegree",
                "description": "Returns the maximum outdegree (number of outgoing edges) for any node.",
            },
        },
    ]
    return tools


def load_functions_code():
    return """
def get_number_of_nodes():
    return flowchart.get_number_of_nodes()

def get_number_of_edges():
    return flowchart.get_number_of_edges()

def get_direct_successors(node_description):
    return flowchart.get_direct_successors(node_description)

def get_direct_predecessors(node_description):
    return flowchart.get_direct_predecessors(node_description)

def get_shortest_path_length(node1, node2):
    return flowchart.get_shortest_path_length(node1, node2)

def get_max_indegree():
    return flowchart.get_max_indegree()

def get_max_outdegree():
    return flowchart.get_max_outdegree()
"""


def test_env():
    exec_code = """
import os
result = os.getcwd()
    """
    try:
        namespace = {}
        exec(exec_code, {}, namespace)  # Isolate execution
        result = namespace.get("result")
        print("Current Working Directory:", result)
        return result
    except Exception as e:
        print("Exception occurred:", e)
        return None


def test_env():
    exec_code = """
from flowchart import Flowchart

flowchart = Flowchart()

flowchart.add_node("A", "Start", "ellipse")
flowchart.add_node("B", "Moisten the Stain with Water", "box")
flowchart.add_node("C", "Apply Lemon Juice", "box")
flowchart.add_node("D", "Apply Salt to Stain", "box")
flowchart.add_node("E", "Is the stain still visible?", "diamond")
flowchart.add_node("F", "Rinse the Stain", "box")
flowchart.add_node("G", "Is the stain nearly gone?", "diamond")
flowchart.add_node("H", "Dry with Towel", "box")
flowchart.add_node("I", "Apply More Lemon Juice", "box")
flowchart.add_node("J", "Sun-Dry the Fabric", "box")
flowchart.add_node("K", "End", "ellipse")

flowchart.add_edge("A", "B")
flowchart.add_edge("B", "C")
flowchart.add_edge("C", "D")
flowchart.add_edge("D", "E")
flowchart.add_edge("E", "F", "Yes")
flowchart.add_edge("F", "E")
flowchart.add_edge("E", "G", "No")
flowchart.add_edge("G", "H", "Yes")
flowchart.add_edge("H", "I")
flowchart.add_edge("I", "J")
flowchart.add_edge("J", "K")
flowchart.add_edge("G", "H", "No")

def get_number_of_nodes():
    return flowchart.get_number_of_nodes()

def get_number_of_edges():
    return flowchart.get_number_of_edges()

def get_direct_successors(node_description):
    return flowchart.get_direct_successors(node_description)

def get_direct_predecessors(node_description):
    return flowchart.get_direct_predecessors(node_description)

def get_shortest_path_length(node1, node2):
    return flowchart.get_shortest_path_length(node1, node2)

def get_max_indegree():
    return flowchart.get_max_indegree()

def get_max_outdegree():
    return flowchart.get_max_outdegree()

result = get_number_of_nodes()
print(result)
"""

    try:
        exec(exec_code, globals())  # Use global namespace
        result = globals().get("result")  # Retrieve 'result' from global namespace
        return result
    except Exception as e:
        print("Exception occurred:", e)
        return None


def generate_response_tool_use(model_name, model, prompt, representation):
    logger = logging.getLogger(__name__)
    messages = load_messages(model_name, prompt)

    # Tool use is only implemented for gpt-4o and gpt-4o-mini.
    # But it can be esaily extend to any LLMs that support tool use.
    if model_name in ["gpt-4o", "gpt-4o-mini"]:
        tools = load_tools()
        client = model
        completion = client.chat.completions.create(
            model=config["model_version"][model_name],
            max_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"],
            messages=messages,
            tools=tools,
        )
        response = completion.choices[0].message.content
        # Response directly without tool use
        if response is not None:
            return response
        # Use tools
        else:
            # Convert mermaid code into executable python graph object
            converter = Mermaid2Flowchart(representation)
            python_code = converter.convert()
            # Add pre-defined functions
            functions_code = load_functions_code()
            python_code += functions_code

            # Initialize list to hold each function call result message
            function_call_result_messages = []

            # Loop through each tool call in the response
            # LLM may return multiple function calls
            for tool_call in completion.choices[0].message.tool_calls:
                print(tool_call)
                func_name = tool_call.function.name
                arguments_str = tool_call.function.arguments

                # The function takes no argument, such as get_number_of_nodes.
                if arguments_str == "{}":
                    argument = None
                    value = None
                else:
                    # Save the arguments' name and value pairs into the dictionary
                    arguments_dict = json.loads(arguments_str)

                    # The function takes one argument, such as get_direct_successors
                    if len(arguments_dict) == 1:
                        argument, value = next(iter(arguments_dict.items()))
                    # The function takes two arguments, such as get_shortest_path_length
                    else:
                        (argument, value), (argument2, value2) = list(
                            arguments_dict.items()
                        )[:2]

                # Generate the appropriate Python code for each function call
                if func_name == "get_number_of_nodes":
                    exec_code = python_code + "\nresult = get_number_of_nodes()"
                elif func_name == "get_number_of_edges":
                    exec_code = python_code + "\nresult = get_number_of_edges()"
                elif func_name == "get_direct_successors":
                    value = value.replace('"', '\\"')
                    exec_code = (
                        python_code + f'\nresult = get_direct_successors("{value}")'
                    )
                elif func_name == "get_direct_predecessors":
                    value = value.replace('"', '\\"')
                    exec_code = (
                        python_code + f'\nresult = get_direct_predecessors("{value}")'
                    )
                elif func_name == "get_shortest_path_length":
                    value = value.replace('"', '\\"')
                    value2 = value2.replace('"', '\\"')
                    exec_code = (
                        python_code
                        + f'\nresult = get_shortest_path_length("{value}", "{value2}")'
                    )
                elif func_name == "get_max_indegree":
                    exec_code = python_code + "\nresult = get_max_indegree()"
                elif func_name == "get_max_outdegree":
                    exec_code = python_code + "\nresult = get_max_outdegree()"

                # Execute and get result
                try:
                    exec(exec_code, globals())  # Use global namespace
                    result = globals().get(
                        "result"
                    )  # Retrieve 'result' from global namespace
                except Exception as e:
                    result = None
                    logger.error(f"Error on sample:\n{prompt}")
                    logger.error(
                        f"Exception: {e} occurred while executing:\n{exec_code}"
                    )

                # Append the function call result message
                if func_name == "get_number_of_nodes":
                    function_call_result_message = {
                        "role": "tool",
                        "content": json.dumps({"number_of_nodes": result}),
                        "tool_call_id": tool_call.id,
                    }
                elif func_name == "get_number_of_edges":
                    function_call_result_message = {
                        "role": "tool",
                        "content": json.dumps({"number_of_edges": result}),
                        "tool_call_id": tool_call.id,
                    }
                elif func_name == "get_direct_successors":
                    function_call_result_message = {
                        "role": "tool",
                        "content": json.dumps(
                            {argument: value, "direct_successors": result}
                        ),
                        "tool_call_id": tool_call.id,
                    }
                elif func_name == "get_direct_predecessors":
                    function_call_result_message = {
                        "role": "tool",
                        "content": json.dumps(
                            {argument: value, "direct_predecessors": result}
                        ),
                        "tool_call_id": tool_call.id,
                    }
                elif func_name == "get_shortest_path_length":
                    function_call_result_message = {
                        "role": "tool",
                        "content": json.dumps(
                            {
                                argument: value,
                                argument2: value2,
                                "shortest_path_length": result,
                            }
                        ),
                        "tool_call_id": tool_call.id,
                    }
                elif func_name == "get_max_indegree":
                    function_call_result_message = {
                        "role": "tool",
                        "content": json.dumps({"max_indegree": result}),
                        "tool_call_id": tool_call.id,
                    }
                elif func_name == "get_max_outdegree":
                    function_call_result_message = {
                        "role": "tool",
                        "content": json.dumps({"max_outdegree": result}),
                        "tool_call_id": tool_call.id,
                    }

                function_call_result_messages.append(function_call_result_message)

            completion = client.chat.completions.create(
                model=config["model_version"][model_name],
                max_tokens=CONFIG["max_new_tokens"],
                temperature=CONFIG["temperature"],
                messages=[
                    *messages,  # Prior messages (prompt)
                    completion.choices[0].message,  # Returned function calls
                    *function_call_result_messages,  # Function calls' results
                ],
            )
            response = completion.choices[0].message.content
            return response

    else:
        logger.error(f"Model {model_name} has not been impelmented for tool use.")
        raise ValueError("Unsupported model for tool use.")


def generate_evaluation_response(model_name, model, prompt):
    logger = logging.getLogger(__name__)
    messages = load_messages(model_name, prompt, image=None)

    # GPT-4o is used as evaluator
    client = model
    completion = client.chat.completions.create(
        model=config["model_version"][model_name],
        max_tokens=CONFIG["max_new_tokens"],
        temperature=0.2,
        n=3,  # Generate 3 responses
        messages=messages,
    )
    # Extract all three responses
    responses = [choice.message.content for choice in completion.choices]
    return responses
