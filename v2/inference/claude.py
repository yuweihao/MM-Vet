import json
import time
import os
import base64
import requests
import argparse
import anthropic


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class Claude:
    def __init__(self, api_key, 
                 model="claude-3-5-sonnet-20240620", temperature=0.0,
                 max_tokens=512, system=None):
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=api_key,
        )
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def get_response(self, image_folder, prompt="What's in this image?"):
        messages = []
        content = []
        queries = prompt.split("<IMG>")
        img_num = 0
        for query in queries:
            query = query.strip()
            if query == "":
                continue
            if query.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(image_folder, query)
                base64_image = encode_image(image_path)
                image_format = "png" if image_path.endswith('.png') else "jpeg"
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{image_format}",
                            "data": base64_image,
                        }
                    }
                )
                img_num += 1
            else:
                content.append(
                    {
                        "type": "text",
                        "text": query
                    },
                )

        messages.append({
            "role": "user",
            "content": content,
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.system:
            payload["system"] = self.system
        
        response = self.client.messages.create(**payload)
        response_text = response.content[0].text
        return response_text


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmvetv2_path",
        type=str,
        default="/path/to/mm-vet-v2",
        help="Download mm-vet.zip and `unzip mm-vet.zip` and change the path here",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--anthropic_api_key", type=str, default=None,
        help="refer to https://platform.openai.com/docs/quickstart?context=python"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="claude-3-5-sonnet-20240620",
        help="Claude model name",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    model_name = args.model_name
    if os.path.exists(args.result_path) is False:
        os.makedirs(args.result_path)
    results_path = os.path.join(args.result_path, f"{model_name}.json")
    image_folder = os.path.join(args.mmvetv2_path, "images")
    meta_data = os.path.join(args.mmvetv2_path, "mm-vet-v2.json")


    if args.anthropic_api_key:
        ANTHROPIC_API_KEY = args.anthropic_api_key
    else:
        ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

    if ANTHROPIC_API_KEY is None:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable or pass it as an argument")
    
    claude = Claude(ANTHROPIC_API_KEY, model=model_name)

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    with open(meta_data, "r") as f:
        data = json.load(f)

    for id in data:
        if id in results:
            continue
        prompt = data[id]["question"].strip()
        print(id)
        print(f"Prompt: {prompt}")
        response = claude.get_response(image_folder, prompt)
        print(f"Response: {response}")
        results[id] = response
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
