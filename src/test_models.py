from models import ModelWrapper


def main():

    # 1 Test text only prompt
    # 1.1 Models via API calls
    # model_name = "claude-3-5-sonnet"
    model_name = "gpt-4o"
    # model_name = "gpt-4o-mini"

    # 1.2 Models run locally
    # model_name = "Llama-3.1-8B"
    # model_name = "Llama-3.1-70B"
    # model_name = "Mixtral-8x22B"
    # model_name = "Phi-3.5-mini"
    # model_name = "Phi-3.5-MoE"
    # model_name = "Qwen2.5-7B"
    # model_name = "Qwen2.5-14B"
    # model_name = "Qwen2.5-32B"
    # model_name = "Qwen2.5-72B"
    # model_name = "Llama-3.2-11B"
    # model_name = "Llama-3.2-90B"
    # model_name = "Llava-v1.6-110b"
    # model_name = "Qwen2-VL-7B"
    # model_name = "Qwen2-VL-72B"

    model = ModelWrapper(model_name)
    prompt = "what is 2+2?"
    response = model.generate_response(prompt)
    print(response)

    # # 2 Test text with image prompt
    # # 2.1 Models via API calls
    # # model_name = "claude-3-5-sonnet"
    # model_name = "gpt-4o"
    # # model_name = "gpt-4o-mini"

    # # 2.2 Models run locally
    # # model_name = "Llama-3.2-11B"
    # # model_name = "Llama-3.2-90B"
    # # model_name = "Llava-v1.6-110b"
    # # model_name = "Qwen2-VL-7B"
    # # model_name = "Qwen2-VL-72B"

    # model = ModelWrapper(model_name)
    # prompt = "Generate the Mermaid code for the provided flowchart."
    # image_path = "data/flowvqa/images/instruct00025.png"
    # response = model.generate_response(prompt, image_path)
    # print(response)


if __name__ == "__main__":
    main()
