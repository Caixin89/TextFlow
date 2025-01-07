from config import config
from models.api_models import (generate_api_evaluation_response,
                               generate_api_response,
                               generate_api_response_tool_use, load_api_model)
from models.local_models import (generate_local_response,
                                 generate_local_response_tool_use,
                                 load_local_model)
from models.prompt_utils import load_messages
from utils import encode_image

CONFIG = config["model_config"]


class ModelWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_api_model = model_name in [
            "claude-3-5-sonnet",
            "gpt-4o",
            "gpt-4o-mini",
        ]

        if self.is_api_model:
            self.model = load_api_model(model_name)
        else:
            self.model, self.tokenizer = load_local_model(model_name)

    def generate_response(self, prompt, image_path=None, representation=None):
        image = encode_image(image_path, self.model_name) if image_path else None
        tool_use = representation is not None
        messages = load_messages(self.model_name, prompt, image)

        if self.is_api_model:
            if tool_use:
                return generate_api_response_tool_use(
                    self.model_name, self.model, messages, representation
                )
            else:
                return generate_api_response(
                    self.model_name, self.model, messages, image
                )
        else:
            if tool_use:
                generate_local_response_tool_use(
                    self.model_name,
                    self.model,
                    self.tokenizer,
                    messages,
                    representation,
                )
            else:
                return generate_local_response(
                    self.model_name, self.model, self.tokenizer, messages, image
                )

    def generate_evaluation_response(self, prompt):
        messages = load_messages(self.model_name, prompt)
        return generate_api_evaluation_response(self.model_name, self.model, messages)
