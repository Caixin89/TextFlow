from config import config
from models.api_models import (generate_api_evaluation_response,
                               generate_api_response,
                               generate_api_response_tool_use, 
                               load_api_model)
from models.prompt_utils import load_messages
from utils import encode_image

CONFIG = config["model_config"]


class ModelWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = load_api_model(model_name)

    def generate_response(self, prompt, image_path=None, representation=None):
        image = encode_image(image_path) if image_path else None
        tool_use = representation is not None
        messages = load_messages(prompt, image)

        if tool_use:
            return generate_api_response_tool_use(
                self.model_name, self.model, messages, representation
            )
        else:
            return generate_api_response(
                self.model_name, self.model, messages, image
            )

    def generate_evaluation_response(self, prompt, seed):
        messages = load_messages(prompt)
        return generate_api_evaluation_response(self.model_name, self.model, messages, seed)
