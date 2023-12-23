import json
import os
import time
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image

class EvalGemini:
    def __init__(self, project_id, location, model="gemini-pro-vision"):
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        self.multimodal_model = GenerativeModel("gemini-pro-vision")

    def generate_text(self, image_path, prompt) -> str:
        # Query the model
        text = ""
        while len(text) < 1:
            try:
                response = self.multimodal_model.generate_content(
                    [
                        # Add an example image
                        Part.from_image(Image.load_from_file(image_path)),
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


if __name__ == "__main__":
    project_id = "GCP_PROJECT_ID"
    location = 'GCP_LOCATION'
    model_name = "gemini-pro-vision"
    # change the path to your own path
    results_path = f'../results/{model_name}.json' # path to save the results
    image_folder = "/Users/yuweihao/od/code/mm-vet/images" 
    meta_data = "/Users/yuweihao/od/code/mm-vet/mm-vet.json"

    with open(meta_data, 'r') as f:
        data = json.load(f)

    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    data_num = len(data)

    gemini = EvalGemini(project_id, location, model=model_name)

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
