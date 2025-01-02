import base64
import io
import re

from PIL import Image


def encode_image_openai(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Claude support image size up to 8000x8000 pixels
# Resize image before encoding the image
def encode_image_anthropic(image_path):
    with Image.open(image_path) as img:
        # Get current dimensions
        width, height = img.size

        # Check if either dimension is larger than 8000
        if width > 8000 or height > 8000:
            # Calculate the scaling factor while maintaining aspect ratio
            scaling_factor = 8000 / max(width, height)
            new_size = (int(width * scaling_factor), int(height * scaling_factor))

            # Resize the image while keeping the aspect ratio
            img = img.resize(new_size)

        # Convert the image to bytes for base64 encoding
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

    # Return the base64 encoded string
    return base64.b64encode(img_bytes).decode("utf-8")


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


def majority_vote(*decisions):
    return max(set(decisions), key=decisions.count)


def shuffle_mermaid(mermaid_str):
    """Shuffle the lines of a Mermaid diagram string except for the first line."""
    lines = mermaid_str.strip().split("\n")
    if len(lines) <= 1:
        return mermaid_str  # If only one line, return as is

    first_line = lines[0]
    rest_lines = lines[1:]
    random.seed(config["experiment"]["seed"])
    random.shuffle(rest_lines)  # Shuffle the remaining lines

    return "\n".join([first_line] + rest_lines)


def reverse_mermaid(mermaid_str):
    """Reverse the lines of a Mermaid diagram string except for the first line."""
    lines = mermaid_str.strip().split("\n")
    if len(lines) <= 1:
        return mermaid_str  # If only one line, return as is

    first_line = lines[0]
    rest_lines = lines[1:]
    rest_lines.reverse()  # Reverse the remaining lines

    return "\n".join([first_line] + rest_lines)
