import os
import time
from pathlib import Path
import argparse
import json
import google.generativeai as genai
from utils import evaluate_on_mmvetv2


class Gemini:
    def __init__(self, model="gemini-1.5-pro"):
        self.model = genai.GenerativeModel(model)

    def get_response(self, image_folder, prompt="What's in this image?") -> str:

        content = []
        queries = prompt.split("<IMG>")
        img_num = 0
        for query in queries:
            if query.endswith((".jpg", ".png", ".jpeg")):
                image_path = Path(os.path.join(image_folder, query))
                image = {
                    'mime_type': f'image/{image_path.suffix[1:].replace("jpg", "jpeg")}',
                    'data': image_path.read_bytes()
                }
                img_num += 1
                content.append(image)
            else:
                content.append(query)

        if img_num > 16:
            return ""
        # Query the model
        text = ""
        while len(text) < 1:
            try:
                response = self.model.generate_content(
                    content
                )
                try:
                    text = response.text
                except:
                    text = " "
            except Exception as error:
                print(error)
                print('Sleeping for 10 seconds')
                time.sleep(10)
        return text.strip()


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmvetv2_path",
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
        "--model_name",
        type=str,
        default="gemini-1.5-pro",
        help="Gemini model name",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()

    if args.google_api_key:
        GOOGLE_API_KEY = args.google_api_key
    else:
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    if GOOGLE_API_KEY is None:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable or pass it as an argument")

    genai.configure(api_key=GOOGLE_API_KEY)
    model = Gemini(model=args.model_name)

    evaluate_on_mmvetv2(args, model)

    
