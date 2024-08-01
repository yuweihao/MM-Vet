import os
import json
import base64


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def evaluate_on_mmvetv2(args, model):
    if os.path.exists(args.result_path) is False:
        os.makedirs(args.result_path)

    model_name = args.model_name.replace('/', '--')
    results_path = os.path.join(args.result_path, f"{model_name}.json")
    image_folder = os.path.join(args.mmvetv2_path, "images")
    meta_data = os.path.join(args.mmvetv2_path, "mm-vet-v2.json")

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    with open(meta_data, "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        id = f"v2_{i}"
        if id in results:
            continue
        prompt = data[id]["question"].strip()
        print(id)
        print(f"Prompt: {prompt}")
        try: 
            response = model.get_response(image_folder, prompt)
        except:
            response = ""
        print(f"Response: {response}")
        results[id] = response
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)