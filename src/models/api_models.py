import json
import logging
import time
import re

from anthropic import Anthropic
from openai import OpenAI

from config import config
from models.mermaid_parser import Mermaid2Flowchart
from models.prompt_utils import load_tools, load_tools_code
from utils import get_base_model_name

max_new_tokens = config["model_config"]["max_new_tokens"]
temperature = config["model_config"]["temperature"]

def is_openrouter_model(model_name):
    return model_name.startswith("openrouter/")

def get_model_id(model_name):
    logger = logging.getLogger(__name__)
    model_id = config["model_version"].get(get_base_model_name(model_name))
    if not model_id:
        model_id = model_name.removeprefix("openrouter/")
        logger.warning(f"Model version for {model_name} not found, using {model_id} as model ID.")
    return model_id

def load_api_model(model_name):
    logger = logging.getLogger(__name__)
    api_keys = config["api_keys"]

    if model_name == "claude-3-5-sonnet":
        client = Anthropic(api_key=api_keys["ANTHROPIC_API_KEY"])
    elif model_name in ["gpt-4o", "gpt-4o-mini"]:
        client = OpenAI(api_key=api_keys["OPENAI_API_KEY"])
    elif is_openrouter_model(model_name):
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_keys["OPENROUTER_API_KEY"])
    else:
        logger.error(f"API model {model_name} is not supported.")
        raise ValueError(f"API model {model_name} is not supported.")

    logger.info(f"Loading model {model_name}...")
    return client


def generate_api_response(model_name, client, messages, image=None):
    logger = logging.getLogger(__name__)
    model_id = get_model_id(model_name)

    if model_name == "claude-3-5-sonnet":
        message = client.messages.create(
            model=model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            system="",
            messages=messages,
        )
        response = message.content[0].text
    elif is_openrouter_model(model_name) or model_name in ["gpt-4o", "gpt-4o-mini"]:
        completion = client.chat.completions.create(
            model=model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=messages,
        )
        response = completion.choices[0].message.content
    else:
        logger.error(f"Response generation for {model_name} is not implemented.")
        raise ValueError(f"Response generation for {model_name} is not implemented.")

    return response

def generate_api_response_tool_use(model_name, client, messages, representation):
    logger = logging.getLogger(__name__)
    model_id = get_model_id(model_name)

    # Tool use is only implemented for gpt-4o and gpt-4o-mini.
    # But it can be esaily extend to any LLMs that support tool use.
    if get_base_model_name(model_name) in ["gpt-4o", "gpt-4o-mini"]:
        tools = load_tools()
        completion = client.chat.completions.create(
            model=model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
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
            python_code += load_tools_code()

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
                    logger.error(f"Error on sample:\n{messages}")
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
                model=model_id,
                max_tokens=max_new_tokens,
                temperature=temperature,
                messages=[
                    *messages,  # Prior messages (prompt)
                    completion.choices[0].message,  # Returned function calls
                    *function_call_result_messages,  # Function calls' results
                ],
            )
            response = completion.choices[0].message.content
    else:
        logger.error(f"Model {model_name} has not been impelmented for tool use.")
        raise ValueError(f"Model {model_name} has not been impelmented for tool use.")

    return response


def generate_api_evaluation_response(model_name, client, messages, seed):
    logger = logging.getLogger(__name__)
    model_id = get_model_id(model_name)

    logger.debug("Evaluator model: %s ", model_id)

    start = time.time()
    try:
        completion = client.chat.completions.create(
            model=model_id,
            max_tokens=max_new_tokens,
            temperature=0,
            messages=messages,
            seed=seed,
            extra_body={
                "provider": {"only": ["openai", "mistral", "anthropic"]},
                "require_parameters": True,
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "judgement",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "verdict": {"type": "string", "description": "\"Correct\" | \"Incorrect\""},
                            "explanation": {"type": "string", "description": "1–3 sentences explaining your decision."},
                        },
                        "required": ["verdict", "explanation"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        raw = completion.choices[0].message.content
        response_in_json = json.loads(raw)
    except json.JSONDecodeError:
        try:
            # Only attempt to extract JSON from fenced code blocks (```json or ```)
            fence_re = re.compile(r'```(?:json)?\s*(\{.*\}|\[.*\])\s*```', re.I | re.S)
            m = fence_re.search(raw)
            response_in_json = json.loads(m.group(1))
        except Exception as e:
            elapsed = time.time() - start
            logger.error("Evaluation request failed after %.2fs: JSON decode error: %s", elapsed, e)
            logger.error("Raw response: %s", raw)
            raise
    except Exception as e:
        elapsed = time.time() - start
        logger.error("Evaluation request failed after %.2fs: %s", elapsed, e)
        raise

    elapsed = time.time() - start
    logger.info("Evaluation request completed in %.2fs (model=%s)", elapsed, model_id)
    return response_in_json
