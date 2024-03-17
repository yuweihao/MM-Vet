"""
Usage:
python claude.py --mmvet_path /path/to/mm-vet --anthropic_api_key <api_key>
"""
import os
import argparse
import anthropic
from utils import evaluate_on_mmvet, encode_image


class Claude:
    def __init__(self, api_key, 
                 model="claude-3-opus-20240229", temperature=0.0,
                 max_tokens=512, system=None):
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=api_key,
        )
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def get_response(self, image_path, prompt="What's in this image?"):
        base64_image = encode_image(image_path)
        image_format = "png" if image_path.endswith('.png') else "jpeg"

        messages = []
        content = [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{image_format}",
                    "data": base64_image,
                }
            }

        ]

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
        return response_text.strip()


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmvet_path",
        type=str,
        default="/path/to/mm-vet",
        help="Download mm-vet.zip and `unzip mm-vet.zip` and change the path here",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--anthropic_api_key", type=str, default=None,
        help="refer to https://docs.anthropic.com/claude/reference/getting-started-with-the-api"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="claude-3-opus-20240229",
        help="Claude model name",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()

    # prepare the model
    if args.anthropic_api_key:
        ANTHROPIC_API_KEY = args.anthropic_api_key
    else:
        ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

    if ANTHROPIC_API_KEY is None:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable or pass it as an argument")
    
    model = Claude(ANTHROPIC_API_KEY, model=args.model_name)

    # evalute on mm-vet
    evaluate_on_mmvet(args, model)
