"""
Usage:
python gpt4v.py --mmvet_path /path/to/mm-vet --openai_api_key <api_key>
"""

import json
import time
import os
import base64
import requests
import argparse


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class EvalGPT4V:
    def __init__(self, api_key, model="gpt-4-vision-preview", image_detail="auto",
                 system_text="You are a helpful assistant. Generate a short and concise response to the following image text pair."):
        self.api_key = api_key
        self.model = model
        self.image_detail = image_detail
        self.system_text = system_text
        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        self.url = "https://api.openai.com/v1/chat/completions"
    
    def get_response(self, image_path, prompt="What's in this image?"):
        base64_image = encode_image(image_path)
        image_format = "data:image/png;base64" if 'png' in image_path else "data:image/jpeg;base64"
        messages = []
        if self.system_text is not None or self.system_text != "":
            messages.append({
                "role": "system", 
                "content": [
                {
                    "type": "text",
                    "text": self.system_text,
                },
                ]
            })
        messages.append({
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"{image_format},{base64_image}",
                    "detail": self.image_detail,
                }
                }
            ]
        })

        payload = {
        "model": self.model,
        "messages": messages,
        "max_tokens": 300,
        }

        response_text, retry, response_json, regular_time = '', 0, None, 30
        while len(response_text) < 1:
            retry += 1
            time.sleep(1)
            try:
                response = requests.post(self.url, headers=self.headers, json=payload)
                response_json = response.json()
                # print(response_json)
            except Exception as e:
                print(e)
                time.sleep(regular_time)
                continue
            if response.status_code != 200:
                print(response.headers,response.content)
                print(image_path)
                print(f"The response status code for is {response.status_code} (Not OK)")
                time.sleep(regular_time)
                continue
            if 'choices' not in response_json:
                time.sleep(regular_time)
                continue
            response_text = response_json["choices"][0]["message"]["content"]
        return response_text
    

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
        "--openai_api_key", type=str, default=None,
        help="refer to https://platform.openai.com/docs/quickstart?context=python"
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="gpt-4-vision-preview",
        help="GPT model name",
    )
    parser.add_argument(
        "--image_detail",
        type=str,
        default="auto",
        help="Refer to https://platform.openai.com/docs/guides/vision/low-or-high-fidelity-image-understanding",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    model_name = args.gpt_model
    if os.path.exists(args.result_path) is False:
        os.makedirs(args.result_path)
    results_path = os.path.join(args.result_path, f"{model_name}_detail-{args.image_detail}_test.json")
    image_folder = os.path.join(args.mmvet_path, "images")
    meta_data = os.path.join(args.mmvet_path, "mm-vet.json")

    with open(meta_data, 'r') as f:
        data = json.load(f)

    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    data_num = len(data)

    if args.openai_api_key:
        OPENAI_API_KEY = args.openai_api_key
    else:
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    if OPENAI_API_KEY is None:
        raise ValueError("Please set the OPENAI_API_KEY environment variable or pass it as an argument")

    gpt4v = EvalGPT4V(OPENAI_API_KEY, model=model_name, image_detail=args.image_detail)

    for i in range(len(data)):
        id = f"v1_{i}"
        if id in results:
            continue
        imagename = data[id]['imagename']
        img_path = os.path.join(image_folder, imagename)
        prompt = data[id]['question']
        prompt = prompt.strip()
        print("\n", id)
        print(f"Image: {imagename}")
        print(f"Prompt: {prompt}")
        response = gpt4v.get_response(img_path, prompt)                
        print(f"Response: {response}")
        results[id] = response        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)