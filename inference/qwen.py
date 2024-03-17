"""
Usage:
python qwen.py --mmvet_path /path/to/mm-vet --dashscope_api_key <api_key>
"""
import os
import argparse
from utils import evaluate_on_mmvet
from http import HTTPStatus
import dashscope

class Qwen:
    def __init__(self, model='qwen-vl-max'):
        self.model = model
    
    def get_response(self, image_path, prompt="What's in this image?"):
        messages = []
        content = [
            {
                "text": prompt,
            },
            {
                "image": f"file://{image_path}"
            }
        ]

        messages.append({
            "role": "user",
            "content": content,
        })

        payload = {
            "model": self.model,
            "messages": messages,
        }

        response = dashscope.MultiModalConversation.call(**payload)
        if response.status_code == HTTPStatus.OK:
            rps = response['output']['choices'][0]['message']['content']
            for rp in rps:
                if 'text' in rp:
                    response_text = rp['text']
                    return response_text.strip()
        else:
            print(response.code)  # The error code.
            print(response.message)  # The error message.
            return ""


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
        "--dashscope_api_key", type=str, default=None,
        help="refer to https://docs.anthropic.com/claude/reference/getting-started-with-the-api"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen-vl-max",
        help="Qwen model name",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()

    # prepare the model
    if args.dashscope_api_key:
        DASHSCOPE_API_KEY = args.dashscope_api_key
    else:
        DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

    if DASHSCOPE_API_KEY is None:
        raise ValueError("Please set the DASHSCOPE_API_KEY environment variable or pass it as an argument")
    
    dashscope.api_key = DASHSCOPE_API_KEY
    model = Qwen(model=args.model_name)

    # evalute on mm-vet
    evaluate_on_mmvet(args, model)
