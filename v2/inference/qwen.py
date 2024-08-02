import json
import time
import os
import base64
import requests
import argparse
from utils import evaluate_on_mmvetv2
from http import HTTPStatus
import dashscope


class Qwen:
    def __init__(self, model='qwen-vl-max'):
        self.model = model
    
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
                content.append(
                    {
                        "image": f"file://{image_path}"
                    }
                )
                img_num += 1
            else:
                content.append(
                    {
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
        "--dashscope_api_key", type=str, default=None,
        help="refer to https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start"
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

    if args.dashscope_api_key:
        DASHSCOPE_API_KEY = args.dashscope_api_key
    else:
        DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

    if DASHSCOPE_API_KEY is None:
        raise ValueError("Please set the DASHSCOPE_API_KEY environment variable or pass it as an argument")
    
    model = Qwen(model=args.model_name)

    evaluate_on_mmvetv2(args, model)




