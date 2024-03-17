import os
import json
import base64


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def evaluate_on_mmvet(args, model):
    if os.path.exists(args.result_path) is False:
        os.makedirs(args.result_path)
    results_path = os.path.join(args.result_path, f"{args.model_name}.json")
    image_folder = os.path.join(args.mmvet_path, "images")
    meta_data = os.path.join(args.mmvet_path, "mm-vet.json")

    with open(meta_data, 'r') as f:
        data = json.load(f)

    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    for i in range(len(data)):
        id = f"v1_{i}"
        if id in results:
            continue
        imagename = data[id]['imagename']
        img_path = os.path.join(image_folder, imagename)
        prompt = data[id]['question']
        print(f"\n{id}")
        print(f"Image: {imagename}")
        print(f"Prompt: {prompt}")
        response = model.get_response(img_path, prompt)
        print(f"Response: {response}")
        results[id] = response        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)