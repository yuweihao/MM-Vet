import json
import time
import os
import base64
import requests


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class EvalGPT4V:
    def __init__(self, api_key, model="gpt-4-vision-preview", detail="auto",
                 content="You are a helpful assistant. Generate a short and concise response to the following image text pair."):
        self.api_key = api_key
        self.model = model
        self.detail = detail
        self.content = content
        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        self.url = "https://api.openai.com/v1/chat/completions"
    
    def get_response(self, image_path, prompt="What's in this image?"):
        base64_image = encode_image(image_path)
        image_format = "data:image/png;base64" if 'png' in image_path else "data:image/jpeg;base64"
        payload = {
        "model": self.model,
        "messages": [
            {
                "role": "system", 
                "content": [
                {
                    "type": "text",
                    "text": self.content
                },
                ]
            },
            {
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
                    "detail": self.detail,
                }
                }
            ]
            }
        ],
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
        return response_json["choices"][0]["message"]["content"]
    


if __name__ == "__main__":
    # OpenAI API Key
    OPENAI_API_KEY = "YOUR_API_KEY"
    model_name = "gpt-4-vision-preview"
    detail = "high"
    # change the path to your own path
    results_path = f'../results/{model_name}_detail-{detail}.json' # path to save the results
    image_folder = f"/path/to/mm-vet/images" 
    meta_data = "/path/to/mm-vet/mm-vet.json" 

    with open(meta_data, 'r') as f:
        data = json.load(f)

    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    data_num = len(data)

    gpt4v = EvalGPT4V(OPENAI_API_KEY, model=model_name, detail=detail)

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