import json
import logging
import time
import re

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
        
    if is_openrouter_model(model_name):
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_keys["OPENROUTER_API_KEY"])
    else:
        client = OpenAI(api_key=api_keys["OPENAI_API_KEY"])

    logger.info(f"Loading model {model_name}...")
    return client


def generate_api_response(model_name, client, messages, image=None):
    logger = logging.getLogger(__name__)
    model_id = get_model_id(model_name)

    completion = client.chat.completions.create(
        model=model_id,
        max_tokens=max_new_tokens,
        temperature=temperature,
        messages=messages,
    )
    response = completion.choices[0].message.content

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


def generate_api_evaluation_response(model_name, client, messages, seed, max_retries: int = 3):
    logger = logging.getLogger(__name__)
    model_id = get_model_id(model_name)
    logger.debug("Evaluator model: %s", model_id)

    fence_re = re.compile(r'```(?:json)?\s*(\{.*\}|\[.*\])\s*```', re.I | re.S)
    any_json_re = re.compile(r'(\{.*\}|\[.*\])', re.S)

    start = time.time()
    raw = None
    last_exc = None

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model_id,
                max_tokens=max_new_tokens,
                temperature=0,
                messages=messages,
                seed=seed,
                extra_body={"provider": {"only": ["openai", "mistral", "anthropic"]}},
            )

            # assume content is always a string
            raw = completion.choices[0].message.content or ""

            # 1) Try direct JSON (response is raw JSON text)
            try:
                response_in_json = json.loads(raw)
                elapsed = time.time() - start
                logger.info("Evaluation request completed in %.2fs (model=%s)", elapsed, model_id)
                return response_in_json
            except Exception:
                pass

            # 2) Try fenced JSON block ```json ... ``` or ```
            m = fence_re.search(raw)
            if m:
                candidate = m.group(1)
                response_in_json = json.loads(candidate)
                elapsed = time.time() - start
                logger.info("Evaluation request completed in %.2fs (model=%s)", elapsed, model_id)
                return response_in_json

            # 3) Fallback: find first {...} or [...] anywhere in text
            m2 = any_json_re.search(raw)
            if m2:
                candidate = m2.group(1)
                response_in_json = json.loads(candidate)
                elapsed = time.time() - start
                logger.info("Evaluation request completed in %.2fs (model=%s)", elapsed, model_id)
                return response_in_json

            # nothing parsed
            raise json.JSONDecodeError("No JSON found in response", raw, 0)

        except Exception as e:
            last_exc = e
            elapsed = time.time() - start
            logger.warning("Attempt %d/%d failed after %.2fs: %s", attempt, max_retries, elapsed, e)
            if attempt == max_retries:
                logger.error("Evaluation request failed after %.2fs: %s", elapsed, e)
                logger.error("Raw response (if any): %s", raw)
                raise
            time.sleep(2 ** (attempt - 1))

    raise last_exc
