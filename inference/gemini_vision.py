"""
Please refer to https://ai.google.dev/tutorials/python_quickstart to get the API key

Install with `pip install -q -U google-generativeai`,
Then `python gemini_vision.py --mmvet_path /path/to/mm-vet --google_api_key YOUR_API_KEY`
"""

import os
import time
from pathlib import Path
import json
import google.generativeai as genai
import argparse

class EvalGemini:
    def __init__(self, model="gemini-pro-vision"):
        self.model = genai.GenerativeModel(model)

    def generate_text(self, image_path, prompt) -> str:
        # Query the model
        text = ""
        while len(text) < 1:
            try:
                image_path = Path(image_path)
                image = {
                    'mime_type': f'image/{image_path.suffix[1:].replace("jpg", "jpeg")}',
                    'data': image_path.read_bytes()
                }
                response = self.model.generate_content(
                    [
                        # Add an example image
                        image,
                        # Add an example query
                        prompt,
                    ]
                )
                try:
                    text = response.text
                except:
                    text = " "
            except Exception as error:
                print(error)
                print('Sleeping for 10 seconds')
                time.sleep(10)
        return text


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
        "--google_api_key", type=str, default=None,
        help="refer to https://ai.google.dev/tutorials/python_quickstart"
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-pro-vision",
        help="Gemini model name",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    model_name = args.gemini_model
    if os.path.exists(args.result_path) is False:
        os.makedirs(args.result_path)
    results_path = os.path.join(args.result_path, f"{model_name}.json")
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

    if args.google_api_key:
        GOOGLE_API_KEY = args.google_api_key
    else:
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    if GOOGLE_API_KEY is None:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable or pass it as an argument")

    genai.configure(api_key=GOOGLE_API_KEY)
    gemini = EvalGemini(model=model_name)

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
        response = gemini.generate_text(img_path, prompt)
        response = response.strip()           
        print(f"Response: {response}")
        results[id] = response        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)