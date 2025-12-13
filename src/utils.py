import base64
import io
import re

from PIL import Image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_mermaid_code(string):
    mermaid_pattern = r"```mermaid\s+([\s\S]*?)```"

    match = re.search(mermaid_pattern, string)

    if match:
        # Extract the mermaid code block
        mermaid_code = match.group(1).strip()
        return mermaid_code
    else:
        return string


def extract_graphviz_code(string):
    mermaid_pattern = r"```dot\s+([\s\S]*?)```"

    match = re.search(mermaid_pattern, string)

    if match:
        # Extract the mermaid code block
        mermaid_code = match.group(1).strip()
        return mermaid_code
    else:
        return string


def extract_plantuml_code(string):
    mermaid_pattern = r"```plantuml\s+([\s\S]*?)```"

    match = re.search(mermaid_pattern, string)

    if match:
        # Extract the mermaid code block
        mermaid_code = match.group(1).strip()
        return mermaid_code
    else:
        return string


def extract_representation(string):
    if "```mermaid" in string:
        return extract_mermaid_code(string)
    elif "```dot" in string:
        return extract_graphviz_code(string)
    elif "```plantuml" in string:
        return extract_plantuml_code(string)
    else:
        return string


def majority_vote(decisions):
    return max(set(decisions), key=decisions.count)


def get_base_model_name(model_name):
    return model_name.split("/")[-1]
