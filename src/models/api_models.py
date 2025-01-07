import json
import logging

from anthropic import Anthropic
from openai import OpenAI

from config import config
from models.mermaid_parser import Mermaid2Flowchart
from models.prompt_utils import load_tools, load_tools_code

max_new_tokens = config["model_config"]["max_new_tokens"]
temperature = config["model_config"]["temperature"]


def load_api_model(model_name):
    logger = logging.getLogger(__name__)
    api_keys = config["api_keys"]

    if model_name == "claude-3-5-sonnet":
        client = Anthropic(api_key=api_keys["ANTHROPIC_API_KEY"])
    elif model_name in ["gpt-4o", "gpt-4o-mini"]:
        client = OpenAI(api_key=api_keys["OPENAI_API_KEY"])
    else:
        logger.error(f"API model {model_name} is not supported.")
        raise ValueError(f"API model {model_name} is not supported.")

    logger.info(f"Loading model {model_name}...")
    return client


def generate_api_response(model_name, client, messages, image=None):
    logger = logging.getLogger(__name__)
    model_id = config["model_version"].get(model_name)

    if model_name == "claude-3-5-sonnet":
        message = client.messages.create(
            model=model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            system="",
            messages=messages,
        )
        response = message.content[0].text
    elif model_name in ["gpt-4o", "gpt-4o-mini"]:
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
    model_id = config["model_version"].get(model_name)

    # Tool use is only implemented for gpt-4o and gpt-4o-mini.
    # But it can be esaily extend to any LLMs that support tool use.
    if model_name in ["gpt-4o", "gpt-4o-mini"]:
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


def generate_api_evaluation_response(model_name, client, messages):
    logger = logging.getLogger(__name__)
    model_id = config["model_version"].get(model_name)

    # GPT-4o is used as evaluator
    completion = client.chat.completions.create(
        model=model_id,
        max_tokens=max_new_tokens,
        temperature=0.2,
        n=3,  # Generate 3 responses
        messages=messages,
    )
    # Extract all three responses
    responses = [choice.message.content for choice in completion.choices]
    return responses
